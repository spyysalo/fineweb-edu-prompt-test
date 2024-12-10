#!/usr/bin/env python3

# Attempt to replicate the process to generate the FineWeb-edu Llama 3
# annotations
# https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations

# Attempts to do this batched ran up against
# https://github.com/meta-llama/llama/issues/380. None of the suggested
# solutions worked.

import sys
import re
import json

from argparse import ArgumentParser
from functools import wraps
from logging import warning

from datasets import load_dataset
from transformers import pipeline


SYSTEM_MESSAGE = 'You are a helpful assistant'

SCORE_RE = re.compile(r'Educational score: ([0-5])\b')

# Generation args from
# https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations/discussions/3#6660891c1552bdf08cadd9c3:

GENERATION_ARGS = {
    'temperature': 0.6,
    'top_p': 0.95,
    'top_k': 50,
    'repetition_penalty': 1,
}


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--key', default='prompt')
    ap.add_argument('--max-new-tokens', type=int, default=1000)
    ap.add_argument('model')
    ap.add_argument('infile')
    ap.add_argument('outfile')
    return ap


def load_pipeline(args):
    pipe = pipeline(
        'text-generation',
        args.model,
        device_map='auto',
        torch_dtype='auto'
    )
    if pipe.tokenizer.pad_token_id is None:
        eos_token_id = pipe.tokenizer.eos_token_id
        warning(f'Setting pad_token_id to eos_token_id:{eos_token_id}')
        pipe.tokenizer.pad_token_id = eos_token_id
        pipe.model.generation_config.pad_token_id = eos_token_id
    return pipe


def load_data(args):
    return load_dataset(
        'json',
        data_files=args.infile,
        split='train'
    )


def apply_template(tokenizer, prompt):
    messages =  [
        { 'role': 'system', 'content': SYSTEM_MESSAGE },
        { 'role': 'user', 'content': prompt },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def format_prompts(dataset, pipe, args):
    def add_input_field(example):
        example['input'] = apply_template(pipe.tokenizer, example[args.key])
        return example
    return dataset.map(add_input_field)


def predict(dataset, pipe, args):
    output = pipe(
        dataset['input'],
        max_new_tokens=args.max_new_tokens,
        **GENERATION_ARGS,
    )
    output = [o[0]['generated_text'] for o in output]
    for i, o in zip(dataset['input'], output):
        assert o.startswith(i)
    output = [o[len(i):] for i, o in zip(dataset['input'], output)]
    return dataset.add_column('output', output)


def get_score(output):
    m = SCORE_RE.search(output)
    if not m:
        return None
    else:
        return int(m.group(1))


def add_scores(dataset):
    def add_score(example):
        example['score'] = get_score(example['output'])
        return example
    return dataset.map(add_score)


def main(argv):
    args = argparser().parse_args(argv[1:])

    pipe = load_pipeline(args)
    dataset = load_data(args)
    dataset = format_prompts(dataset, pipe, args)
    dataset = predict(dataset, pipe, args)
    dataset = add_scores(dataset)
    dataset.to_json(args.outfile)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
