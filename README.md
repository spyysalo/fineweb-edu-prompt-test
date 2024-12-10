# fineweb-edu-prompt-test

Quick test of FineWeb-edu prompting

## Sample data

`sample_annotations.jsonl` is a random sample of 1000 lines from
`HuggingFaceFW/fineweb-edu-llama3-annotations`

## Quickstart

Assign scores with Llama 3.1 8B Instruct and compare them with the scores
in the original data.

```
python3 score.py meta-llama/Llama-3.1-8B-Instruct sample_annotations.jsonl sample_outputs.jsonl
python3 compare_scores.py sample_annotations.jsonl sample_outputs.jsonl
```
