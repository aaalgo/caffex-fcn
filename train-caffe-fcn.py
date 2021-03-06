#!/usr/bin/env python
import sys 
import os
import re
import shutil
import logging
#import argparse
import subprocess
import simplejson as json

#parser = argparse.ArgumentParser(description='')
#parser.add_argument('input', nargs=1)
#parser.add_argument('output', nargs=1)
#parser.add_argument('tmp', nargs=1)
#args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)

if len(sys.argv) < 4:
    print "usage:  train-caffe.py <model> <tmp> <input> [<input> ...]"
    sys.exit(1)

output = sys.argv[1]
tmp = sys.argv[2]
lists = sys.argv[3:]

lines = []

MAX_R = 2500

for p in lists:
    with open(p, 'r') as f:
        lines.extend(f.readlines())
        pass
    pass

NL = len(lines)
print '%d lines.' % NL
REP = (MAX_R + NL - 1) / NL
print '%d replicates.' % REP
FOLD = 8 

if os.path.exists(output):
    logging.error("%s already exists" % output)
    sys.exit(1)
if os.path.exists(tmp):
    logging.error("%s already exists" % tmp)
    sys.exit(1)

bin_dir = os.path.abspath(os.path.dirname(__file__))

subprocess.check_call("%s %s" % (os.path.join(bin_dir, "finetune-init.py"), tmp), shell=True)

#cat_cmd = 'cat ' + ' '.join(lists) + ' > %s/list' % tmp
#subprocess.check_call(cat_cmd, shell=True)
with open(os.path.join(tmp, 'list'), 'w') as f:
    for l in lines:
        f.write(l)

subprocess.check_call("%s -f %d -R %d --list %s/list --output %s/db --cache %s/cache" % (os.path.join(bin_dir, "import-images"), FOLD, REP, tmp, tmp, tmp), shell=True)
subprocess.check_call("cd %s; %s" % (tmp, os.path.join(bin_dir, "finetune-generate.py")), shell=True)
subprocess.check_call("cd %s; ./train.sh 2>&1 | tee train.log" % tmp, shell=True)
subprocess.check_call("cd %s; grep solver.cpp train.log > solver.log" % tmp, shell=True)

perf = []
best_snap = -1
best_acc = 0
with open(os.path.join(tmp, "solver.log")) as f:
    # look for lines like below
    # I0127 11:24:15.227892 27000 solver.cpp:340] Iteration 552, Testing net (#0)
    # I0127 11:24:15.283869 27000 solver.cpp:408]     Test net output #0: accuracy = 0.975
    re1 = re.compile("Iteration (\d+), Testing")
    re2 = re.compile("accuracy = (.+)")
    while True:
        l = f.readline()
        if not l:
            break
        m = re1.search(l)
        if m:
            it = int(m.group(1))
            l = f.readline()
            m = re2.search(l)
            acc = float(m.group(1))
            perf.append((it, acc))
            if acc > best_acc:
                best_acc = acc
                best_snap = it
            pass
        pass
    pass
# now we have performance data
print perf
# find best one
print "BEST ACCURACY", best_acc, "AT SNAPSHOT", best_snap
        
os.mkdir(output)
shutil.copy(os.path.join(tmp, "deploy.prototxt"),
            os.path.join(output, "caffe.model"))
shutil.copy(os.path.join(tmp, "snapshots", "fcn_iter_%d.caffemodel" % best_snap),
            os.path.join(output, "caffe.params"))
with open(os.path.join(output, "blobs"), "w") as f:
    f.write("prob\n")

# generate lists
#
