#!/usr/bin/python
# !/usr/bin/env python


import math
from collections import OrderedDict
from tool import read_crc_seq, format_box


class Trace:
    def __init__(self, tn, k):
        self.seq = read_crc_seq('datasets/{0}_test.csv'.format(tn))
        self.box_start_pointer = 0
        self.box_end_pointer = 0
        self.box_start_stack = OrderedDict()
        self.box_end_stack = OrderedDict()
        self.box_height = int(k / 13)
        self.box_width = int(k / 13)
        self.completion_time = 0


def find_next(t, k, miss_cost, trace_number):
    t.box_start_pointer = t.box_end_pointer
    t.box_start_stack = OrderedDict()
    for pid in t.box_end_stack.keys():
        t.box_start_stack[pid] = True
        t.box_start_stack.move_to_end(pid, last=True)
    t.box_height = int(k / trace_number)
    t.box_width = t.box_height
    t = format_box(t, miss_cost)
    return t


trace_names = ['astar', 'bwaves', 'bzip', 'cactusadm',
               'gems', 'lbm', 'leslie3d', 'libq', 'mcf',
               'milc', 'omnetpp', 'sphinx3', 'xalanc']
test_traces = {}
k = 128
s = 100
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
        for name in trace_names:
            if (name not in packed_traces) and (name not in finished):
                next_trace = name
                break
        if next_trace == '':
            break

        # pack
        next_box_height = test_traces[next_trace].box_height
        next_box_width = test_traces[next_trace].box_width
        packed_traces[next_trace] = {'mem_size': next_box_height,
                                     'end_time': running_time + next_box_width,
                                     'last': test_traces[next_trace].box_end_pointer >=
                                             len(test_traces[next_trace].seq)}
        available_memory -= next_box_height
        if test_traces[next_trace].box_end_pointer < len(test_traces[next_trace].seq):
            test_traces[next_trace] = find_next(test_traces[next_trace], k, s,
                                                len(trace_names) - len(finished))

    # find next decision time
    next_time_point = math.inf
    for trace in packed_traces.keys():
        if packed_traces[trace]['end_time'] < next_time_point:
            next_time_point = packed_traces[trace]['end_time']
    running_time = next_time_point
    phasing = False
    for trace in list(packed_traces):
        if packed_traces[trace]['end_time'] == next_time_point:
            available_memory += packed_traces[trace]['mem_size']
            test_traces[trace].completion_time = next_time_point
            if packed_traces[trace]['last']:
                finished[trace] = True
                phasing = True
            packed_traces.pop(trace)

    # phasing
    if phasing:
        for name in trace_names:
            if name not in finished.keys():
                test_traces[name].box_height = int(k / (len(trace_names) - len(finished)))
                test_traces[name] = format_box(test_traces[name], miss_cost=s)

avg_complete = 0
for name in trace_names:
    print(name + ',' + str(test_traces[name].completion_time))
    avg_complete += test_traces[name].completion_time
print(avg_complete / 13)
