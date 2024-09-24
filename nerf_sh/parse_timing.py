"""
A utility to parse timings files, which are saved automatically during training
(checkpoint_dir/timings.txt).
If you do not stop & restart training, this allows for measuring the training time.
"""
import argparse
from datetime import datetime

parser = argparse.ArgumentParser();
parser.add_argument("file", type=str);
parser.add_argument("--times", "-t", type=int, default=[], nargs='+');
args = parser.parse_args();

f = open(args.file, 'r')
lines = f.readlines()
lines = [line.strip() for line in lines]
lines = [line.split() for line in lines if len(line)]
lines = {int(line[0]) : datetime.fromisoformat(line[1]) for line in lines}

if not args.times:
    print(list(lines.keys()))

for t in args.times:
    print((lines[t] - lines[0]).total_seconds() / 3600)
