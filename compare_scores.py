#!/usr/bin/env python3

import sys
import json

from argparse import ArgumentParser
from logging import warning


def argparser():
    ap = ArgumentParser()
    ap.add_argument('jsonl1')
    ap.add_argument('jsonl2')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    matches, mismatches = 0, 0
    with open(args.jsonl1) as f1, open(args.jsonl2) as f2:
        for l1, l2 in zip(f1, f2):
            d1, d2 = json.loads(l1), json.loads(l2)
            if d1['text'] != d2['text']:
                raise ValueError('text mismatch')
            s1, s2 = d1['score'], d2['score']
            if s1 is None or s2 is None:
                warning(f'skip pair with None ({s1}, {s2})')
                continue
            s1, s2 = int(s1), int(s2)
            print(s1, s2)
            if s1 == s2:
                matches += 1
            else:
                mismatches +=1
    print(f'{matches} matches, {mismatches} mismatches', file=sys.stderr)


if __name__ == '__main__':
    sys.exit(main(sys.argv))

