#!/usr/bin/python
# !/usr/bin/env python


import collections
import math
from collections import OrderedDict
import xgboost as xgb
from xgboost_module import get_feature_vector, test_xgboost
from tool import read_crc_seq, format_box
from other_algorithms import deterministic_algorithm


class Trace:
    def __init__(self, tname, k, number_of_box_kinds,
                 models, miss_cost, window_size):
        self.seq = read_crc_seq('datasets/{0}_test.csv'.format(tname))
        self.box_start_pointer = 0
        self.box_end_pointer = 0
        self.box_start_stack = OrderedDict()
        self.box_end_stack = OrderedDict()

        feature = xgb.DMatrix([get_feature_vector(self.seq, 0, window_size)])
        self.box_height = k / (2 ** models[0].predict(feature)[0])
        self.box_width = self.box_height

        self.memory_impact = 0
        self.completion_time = 0
        _, _, self.ml_req2mi = test_xgboost(models[0],
                                            self.seq,
                                            k,
                                            miss_cost,
                                            window_size)
        _, _, self.det_req2box, self.det_req2counter, self.det_req2mi = \
            deterministic_algorithm(self.seq, k, number_of_box_kinds, miss_cost)

        self.det_box_kind = self.det_req2box[0]
        self.det_counter = []
        self.check_point = 0
        self.run_oracle = True


def find_next_box(t, k, miss_cost, phase, models,
                  number_of_box_kinds, window_size,
                  check_length=10000, phase_change=False):
    t.box_start_pointer = t.box_end_pointer
    t.box_start_stack = OrderedDict()
    for pid in t.box_end_stack.keys():
        t.box_start_stack[pid] = True
        t.box_start_stack.move_to_end(pid, last=True)

    if phase == 4:
        t.box_height = k
    else:
        if int(t.box_start_pointer / check_length) > t.check_point or phase_change:
            t.check_point = int(t.box_start_pointer / check_length)
            if (t.run_oracle or phase_change) and \
                    t.ml_req2mi[t.box_start_pointer - 1] >= \
                    t.det_req2mi[t.box_start_pointer - 1]:
                t.det_box_kind = t.det_req2box[t.box_start_pointer - 1]
                t.det_counter = t.det_req2counter[t.box_start_pointer - 1]
            t.run_oracle = \
                t.ml_req2mi[t.box_start_pointer - 1] < t.det_req2mi[t.box_start_pointer - 1]

        if t.run_oracle:
            feature = xgb.DMatrix([get_feature_vector(t.seq, t.box_start_pointer, window_size)])
            oracle = models[phase].predict(feature)[0]
            t.box_height = k / (2 ** oracle)
        else:
            print('pick deterministic')
            if t.det_box_kind == number_of_box_kinds - 1:
                t.det_box_kind = 0
            elif t.det_counter[t.det_box_kind] % 4 == 0:
                t.det_box_kind = t.det_box_kind + 1
            else:
                t.det_box_kind = 0
            t.det_counter[t.det_box_kind] = t.det_counter[t.det_box_kind] + 1
            t.box_height = k / (2 ** (number_of_box_kinds - t.det_box_kind - 1))
        t.box_width = t.box_height
    t = format_box(t, miss_cost)
    return t


def run_hybrid_parallel1():
    trace_names = ['astar', 'bwaves', 'bzip', 'cactusadm',
                   'gems', 'lbm', 'leslie3d', 'libq', 'mcf',
                   'milc', 'omnetpp', 'sphinx3', 'xalanc']
    test_traces = {}
    k = 128
    number_of_box_kinds = 5
    s = 100
    phase = 0
    models = []
    for _ in range(4):
        model = xgb.Booster()
        models.append(model)
    models[0].load_model('xgb_model_k128_b5_w256_s100_d20_r50')
    models[1].load_model('xgb_model_k128_b4_w256_s100_d20_r50')
    models[2].load_model('xgb_model_k128_b3_w256_s100_d50_r99')
    models[3].load_model('xgb_model_k128_b2_w256_s100_d90_r99')
    window = 256
    packed_traces = {}
    finished = {}
    running_time = 0
    available_memory = 128

    print(models)
    # initiate
    for name in trace_names:
        test_traces[name] = Trace(name, k, number_of_box_kinds, models, s, window)
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
                test_traces[next_trace] = find_next_box(test_traces[next_trace], k, s,
                                                        phase, models, number_of_box_kinds, window)

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
        phase_change = False
        if 8 >= len(trace_names) - len(finished.keys()) > 4 and phase != 1:
            phase = 1
            phase_change = True
        if 4 >= len(trace_names) - len(finished.keys()) > 2 and phase != 2:
            phase = 2
            phase_change = True
        if len(trace_names) - len(finished.keys()) == 2 and phase != 3:
            phase = 3
            phase_change = True
        if len(trace_names) - len(finished.keys()) == 1 and phase != 4:
            phase = 4
            phase_change = True
        if phase_change:
            number_of_box_kinds -= 1
            for name in trace_names:
                test_traces[name].box_end_pointer = test_traces[name].box_start_pointer
                test_traces[name].box_end_stack = collections.OrderedDict()
                for pid in test_traces[name].box_start_stack:
                    test_traces[name].box_end_stack[pid] = True
                    test_traces[name].box_end_stack.move_to_end(pid, last=True)
                if phase < 4:
                    _, _, test_traces[name].ml_req2mi = test_xgboost(models[phase],
                                                                     test_traces[name].seq,
                                                                     k,
                                                                     s,
                                                                     window)
                    _, _, test_traces[name].det_req2box, test_traces[name].det_req2counter, test_traces[
                        name].det_req2mi = \
                        deterministic_algorithm(test_traces[name].seq, k, number_of_box_kinds, s)
                test_traces[name] = find_next_box(test_traces[name], k, s,
                                                  phase, models, number_of_box_kinds, window, phase_change)

    avg_complete = 0
    for name in trace_names:
        print(name + ',' + str(test_traces[name].completion_time))
        avg_complete += test_traces[name].completion_time
    print(avg_complete / 13)


run_hybrid_parallel1()
