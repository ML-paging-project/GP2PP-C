#!/usr/bin/python
# !/usr/bin/env python


trace_names = ['astar', 'bwaves', 'bzip', 'cactusadm',
               'gems', 'lbm', 'leslie3d', 'libq', 'mcf',
               'milc', 'omnetpp', 'sphinx3', 'xalanc']

for t in trace_names:
    count = len(open(r"datasets/" + t + '_train.csv', 'r').readlines())
    tt = t
    while len(tt) < 12:
        tt += ' '
    print(tt + ':', count)
