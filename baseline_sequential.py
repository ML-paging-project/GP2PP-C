#!/usr/bin/python
# !/usr/bin/env python


from random import shuffle

from tool import read_crc_seq
from collections import OrderedDict
from lru import run_lru


class Trace:
    def __init__(self, tname):
        self.name = tname
        self.seq = read_crc_seq('datasets/{0}_test.csv'.format(tname))
        self.running_time = 0
        self.finish_time = 0

    def __eq__(self, other):
        return self.running_time == other.running_time

    def __le__(self, other):
        return self.running_time < other.running_time

    def __gt__(self, other):
        return self.running_time > other.running_time


def sequential_schedule():
    trace_names = ['astar', 'bwaves', 'bzip', 'cactusadm',
                   'gems', 'lbm', 'leslie3d', 'libq', 'mcf',
                   'milc', 'omnetpp', 'sphinx3', 'xalanc']
    test_traces = []
    k = 128
    s = 100

    for nm in trace_names:
        tc = Trace(nm)
        test_traces.append(tc)
        cache = OrderedDict()
        pointer = 0
        while pointer < len(tc.seq):
            end, rem = run_lru(cache, k, tc.seq,
                               pointer, k * s, 1, s)
            if rem > 0:
                tc.running_time += (k * 3 * s - rem) / 3 / s
            else:
                tc.running_time += k
            for ii in range(pointer, end):
                if tc.seq[ii] not in cache.keys():
                    cache[tc.seq[ii]] = True
                cache.move_to_end(tc.seq[ii], last=False)
                if len(cache.keys()) > k:
                    cache.popitem(last=True)
            pointer = end
        print(nm, tc.running_time)

    print('///////////////////////////////')
    test_traces = sorted(test_traces)
    for ii in range(len(test_traces)):
        if ii == 0:
            test_traces[ii].finish_time = test_traces[ii].running_time
        else:
            test_traces[ii].finish_time = test_traces[ii].running_time + test_traces[ii - 1].finish_time
        print(test_traces[ii].name, test_traces[ii].finish_time, test_traces[ii].running_time)

    max_ct = 0
    mean_ct = 0
    for _ in range(100):
        total = 0
        shuffle(test_traces)
        for ii in range(len(test_traces)):
            if ii == 0:
                test_traces[ii].finish_time = test_traces[ii].running_time
            else:
                test_traces[ii].finish_time = test_traces[ii].running_time + test_traces[ii - 1].finish_time
            total += test_traces[ii].finish_time
        max_ct += test_traces[12].finish_time / 100
        mean_ct += total / 13 / 100
    print(mean_ct, max_ct)


sequential_schedule()
