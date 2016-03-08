#!/usr/bin/env python
import sys 
import os
import logging
import argparse
import simplejson as json

parser = argparse.ArgumentParser(description='init finetune directory.')
parser.add_argument('dir', nargs='?', default='.')
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)

outd = args.dir[0]

if outd != '.':
    os.mkdir(outd)

params = {
        "template": "fcn",
        "backend": "LMDB",
        "train_source": "db/train",
        "train_batch": 1,
        "val_source": "db/val",
        "val_batch": 1,
        "val_batches": 19,
        "num_output": 2,
        "val_interval": 1000,
        "display_interval": 1000,
        "snapshot_interval": 1000,
        "max_iter": 15000,
        "device": "GPU",
}

params_json = json.dumps(params, sort_keys=False, indent=4 * ' ')
open(os.path.join(outd, 'config.json'), 'w').write(params_json)

