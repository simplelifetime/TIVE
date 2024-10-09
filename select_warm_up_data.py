# we sample warm-up data to train a reference model.
# The numbers of data points from each task are equal.
import json
from collections import defaultdict
import random
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", add_help=True)
    parser.add_argument("--source_path", type=str, default=None)
    parser.add_argument('--target_path', type=str, default=None)
    parser.add_argument("--sample_ratio", type=float, default=0.08)
    args = parser.parse_args()
    
    source_data = json.load(open(args.source_path, 'r'))
    
    # categorize each data point to its corrsponding task dataset
    ds2data = defaultdict(list)
    for d in source_data:
        ds2data[d['dataset']].append(d)
    
    sample_number = len(source_data) * args.sample_ratio / len(ds2data)
    
    # sample new warm-up data
    new_data = []
    for k, v in ds2data.items():
        new_data.extend(random.sample(v, int(sample_number)))
        
    json.dump(new_data, open(args.target_path, 'w'), indent=4)

