#!/usr/bin/python
# !/usr/bin/env python


import math
from collections import OrderedDict
from random import shuffle

import xgboost as xgb
from tool import read_crc_seq, format_box


class Trace:
    def __init__(self, tn, k):
        self.seq = read_crc_seq('datasets/{0}_test.csv'.format(tn))
        self.box_start_pointer = 0
        self.box_end_pointer = 0
        self.box_start_stack = OrderedDict()
        self.box_end_stack = OrderedDict()
        self.box_height = k
        self.box_width = k
        self.memory_impact = 0
        self.completion_time = 0


def find_next(t, k, miss_cost):
    t.box_start_pointer = t.box_end_pointer
    t.box_start_stack = OrderedDict()
    for pid in t.box_end_stack.keys():
        t.box_start_stack[pid] = True
        t.box_start_stack.move_to_end(pid, last=True)
    t.box_height = k
    t.box_width = t.box_height
    t = format_box(t, miss_cost)
    return t


def round_robin():
    trace_names = ['astar', 'bwaves', 'bzip', 'cactusadm',
                   'gems', 'lbm', 'leslie3d', 'libq', 'mcf',
                   'milc', 'omnetpp', 'sphinx3', 'xalanc']
    shuffle(trace_names)
    test_traces = {}
    k = 128
    s = 100
    model = xgb.Booster()
    model.load_model('xgb_model_k128_b5_w256_s100_d20_r50')
    window = 256
    packed_traces = {}
    finished = {}
    running_time = 0
    available_memory = 128

    # initiate
    for name in trace_names:
        test_traces[name] = Trace(name, k)
        test_traces[name] = format_box(test_traces[name], s)

    while len(finished.keys()) != len(trace_names):
        while len(packed_traces.keys()) + len(finished) < len(trace_names):
            # find the one to be packed
            next_trace = ''
            min_mi = math.inf
            for name in trace_names:
                if (name not in packed_traces) and \
                        (name not in finished) and \
                        test_traces[name].box_height <= available_memory and \
                        test_traces[name].memory_impact < min_mi:
                    min_mi = test_traces[name].memory_impact
                    next_trace = name

            if next_trace == '':
                break

            # pack
            next_box_height = test_traces[next_trace].box_height
            next_box_width = test_traces[next_trace].box_width
            packed_traces[next_trace] = {'mem_size': next_box_height,
                                         'end_time': running_time + next_box_width,
                                         'last': test_traces[next_trace].box_end_pointer >=
                                                 len(test_traces[next_trace].seq)}
            test_traces[next_trace].memory_impact += next_box_height * next_box_width
            available_memory -= next_box_height
            if test_traces[next_trace].box_end_pointer < len(test_traces[next_trace].seq):
                test_traces[next_trace] = find_next(test_traces[next_trace], k, s)

        # find next decision time
        next_time_point = math.inf
        for trace in packed_traces.keys():
            if packed_traces[trace]['end_time'] < next_time_point:
                next_time_point = packed_traces[trace]['end_time']
        running_time = next_time_point

        for trace in list(packed_traces):
            if packed_traces[trace]['end_time'] == next_time_point:
                available_memory += packed_traces[trace]['mem_size']
                test_traces[trace].completion_time = next_time_point
                if packed_traces[trace]['last']:
                    finished[trace] = True
                packed_traces.pop(trace)

    avg_complete = 0
    max_complete = 0
    for name in trace_names:
        # print(name + ',' + str(test_traces[name].completion_time))
        avg_complete += test_traces[name].completion_time / 13
        max_complete = max(max_complete, test_traces[name].completion_time)
    # print(avg_complete)
    # print(max_complete)
    return avg_complete, max_complete


print('Round robin')
repeat = 100
avg_complete = 0
max_complete = 0
for t in range(repeat):
    print('RR', t)
    a, m = round_robin()
    avg_complete += a / repeat
    max_complete += m / repeat
print(avg_complete)
print(max_complete)
