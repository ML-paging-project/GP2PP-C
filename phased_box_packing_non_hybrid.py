#!/usr/bin/python
# !/usr/bin/env python


import math
from collections import OrderedDict
import xgboost as xgb
from xgboost_module import get_feature_vector
from tool import read_crc_seq, format_box


class Trace:
    def __init__(self, tn, k):
        self.seq = read_crc_seq('datasets/{0}_test.csv'.format(tn))
        self.box_start_pointer = 0
        self.box_end_pointer = 0
        self.box_start_stack = OrderedDict()
        self.box_end_stack = OrderedDict()
        self.box_height = int(k / 16)
        self.box_width = int(k / 16)
        self.memory_impact = 0
        self.completion_time = 0
        self.det_counter = [1, 0, 0, 0, 0]
        self.box_kind = 0


def find_next_by_deterministic(t, k, miss_cost):
    if t.box_end_pointer >= len(t.seq):
        print('mmmmmmmmmmmmmmmmmmmmmmmmmmm')
    t.box_start_pointer = t.box_end_pointer
    t.box_start_stack = OrderedDict()
    for pid in t.box_end_stack.keys():
        t.box_start_stack[pid] = True
        t.box_start_stack.move_to_end(pid, last=True)
    if t.box_kind == (len(t.det_counter) - 1):
        t.box_kind = 0
    elif t.det_counter[t.box_kind] % 4 == 0:
        t.box_kind += 1
    else:
        t.box_kind = 0

    t.det_counter[t.box_kind] += 1
    t.box_height = k / (2 ** (len(t.det_counter) - t.box_kind - 1))
    t.box_width = t.box_height
    t = format_box(t, miss_cost)
    return t


def find_next_by_oracle(t, k, miss_cost, phase, model, window_size):
    if t.box_end_pointer >= len(t.seq):
        print('mmmmmmmmmmmmmmmmmmmmmmmmmmm')
    t.box_start_pointer = t.box_end_pointer
    t.box_start_stack = OrderedDict()
    for pid in t.box_end_stack.keys():
        t.box_start_stack[pid] = True
        t.box_start_stack.move_to_end(pid, last=True)
    if phase == 5:
        t.box_height = k
    else:
        fea = xgb.DMatrix([get_feature_vector(t.seq, t.box_start_pointer, window_size)])
        t.box_height = k / (2 ** model.predict(fea)[0])
    t.box_width = t.box_height
    t = format_box(t, miss_cost)
    return t


def no_hybrid_phased_packing(method):
    # if method = 'oracle' we run xgb parallel paging
    # else we run deterministic parallel paging
    trace_names = ['astar', 'bwaves', 'bzip', 'cactusadm',
                   'gems', 'lbm', 'leslie3d', 'libq', 'mcf',
                   'milc', 'omnetpp', 'sphinx3', 'xalanc']
    test_traces = {}
    k = 128
    s = 100
    phase = 1
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
        if method == 'oracle':
            fea = xgb.DMatrix([get_feature_vector(test_traces[name].seq, 0, window)])
            test_traces[name].box_height = k / (2 ** model.predict(fea)[0])
            test_traces[name].box_width = test_traces[name].box_height
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
                if method == 'oracle':
                    test_traces[next_trace] = find_next_by_oracle(test_traces[next_trace], k, s,
                                                                  phase, model, window)
                else:
                    test_traces[next_trace] = find_next_by_deterministic(test_traces[next_trace], k, s)

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

        # phasing
        if 8 >= len(trace_names) - len(finished.keys()) > 4 and phase != 2:
            phase = 2
            if method == 'oracle':
                model.load_model('xgb_model_k128_b4_w256_s100_d20_r50')
                for name in trace_names:
                    fea = xgb.DMatrix([get_feature_vector(test_traces[name].seq,
                                                          test_traces[name].box_start_pointer,
                                                          window)])
                    test_traces[name].box_height = k / (2 ** model.predict(fea)[0])
                    test_traces[name] = format_box(test_traces[name], miss_cost=s)
            else:
                for name in trace_names:
                    test_traces[name].det_counter = [1, 0, 0, 0]
                    test_traces[name].box_kind = 0
                    test_traces[name].box_height = k / 8
                    test_traces[name] = format_box(test_traces[name], miss_cost=s)

        if 4 >= len(trace_names) - len(finished.keys()) > 2 and phase != 3:
            phase = 3
            if method == 'oracle':
                model.load_model('xgb_model_k128_b3_w256_s100_d50_r99')
                for name in trace_names:
                    fea = xgb.DMatrix([get_feature_vector(test_traces[name].seq,
                                                          test_traces[name].box_start_pointer,
                                                          window)])
                    test_traces[name].box_height = k / (2 ** model.predict(fea)[0])
                    test_traces[name] = format_box(test_traces[name], miss_cost=s)
            else:
                for name in trace_names:
                    test_traces[name].det_counter = [1, 0, 0]
                    test_traces[name].box_kind = 0
                    test_traces[name].box_height = k / 4
                    test_traces[name] = format_box(test_traces[name], miss_cost=s)

        if len(trace_names) - len(finished.keys()) == 2 and phase != 4:
            phase = 4
            if method == 'oracle':
                model.load_model('xgb_model_k128_b2_w256_s100_d90_r99')
                for name in trace_names:
                    fea = xgb.DMatrix([get_feature_vector(test_traces[name].seq,
                                                          test_traces[name].box_start_pointer,
                                                          window)])
                    test_traces[name].box_height = k / (2 ** model.predict(fea)[0])
                    test_traces[name] = format_box(test_traces[name], miss_cost=s)
            else:
                for name in trace_names:
                    test_traces[name].det_counter = [1, 0]
                    test_traces[name].box_kind = 0
                    test_traces[name].box_height = k / 2
                    test_traces[name] = format_box(test_traces[name], miss_cost=s)

        if len(trace_names) - len(finished.keys()) == 1 and phase != 5:
            phase = 5
            for name in trace_names:
                test_traces[name].det_counter = [1]
                test_traces[name].box_height = k
                test_traces[name] = format_box(test_traces[name], miss_cost=s)

    print('k =', k)
    avg_complete = 0
    for name in trace_names:
        print(name + ',' + str(test_traces[name].completion_time))
        avg_complete += test_traces[name].completion_time
    print(avg_complete / 13)


print('Phased packing with xgb only')
no_hybrid_phased_packing('oracle')
print()
print('Phased packing with double box only')
no_hybrid_phased_packing('double box')
