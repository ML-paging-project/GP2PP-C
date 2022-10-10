#!/usr/bin/python
# !/usr/bin/env python


import math
import matplotlib.pyplot as plt


def read_box_seqs(k, file):
    txt_file = open(file, "r")
    file_content = txt_file.read()
    file_content = file_content.replace('[', '')
    file_content = file_content.replace(']', '')
    file_content = file_content.replace(' ', '')
    content_list = file_content.split(",")
    return [k / (2 ** int(v)) for v in content_list]


def parallel_paging(algorithm):
    REQ_NUM = 1173888
    k = 128
    total_memory = 2 * k
    available_memory = 2 * k

    trace_names = ['astar', 'bwaves', 'bzip', 'cactusadm',
                   'gems', 'lbm', 'leslie3d', 'libq', 'mcf',
                   'milc', 'omnetpp', 'sphinx3', 'xalanc']
    # random.shuffle(trace_names)
    # print(trace_names)
    packed_traces = {}
    finished = []
    memory_impacts = {}
    box_seqs = {}

    running_time = 0
    completion_time = {}
    sequential_green_paging_time = 0
    for name in trace_names:
        memory_impacts[name] = 0
        f = 'boxes-s100d5r10/' + name + '-' + algorithm + '-boxes-s100d5r10.txt'
        box_seqs[name] = read_box_seqs(k, f)
        sequential_green_paging_time += sum(box_seqs[name])

    print('\nWith', algorithm, 'green paging:', sequential_green_paging_time / REQ_NUM, 'per req')

    plot_points = {}

    while len(finished) != len(trace_names) or len(packed_traces.keys()) > 0:
        while available_memory > total_memory / 2 and \
                len(packed_traces.keys()) + len(finished) < len(trace_names):
            next_trace = ''
            min_mi = math.inf
            for name in trace_names:
                if (name not in packed_traces) and \
                        (name not in finished) and memory_impacts[name] < min_mi:
                    min_mi = memory_impacts[name]
                    next_trace = name

            next_box = box_seqs[next_trace][0]
            box_seqs[next_trace].pop(0)
            if len(box_seqs[next_trace]) == 0:
                finished.append(next_trace)
            packed_traces[next_trace] = {'mem_size': next_box,
                                         'end_time': running_time + next_box}
            memory_impacts[next_trace] += (next_box ** 2)
            available_memory -= next_box
            plot_points[running_time] = total_memory - available_memory

        # find next decision time
        next_time_point = math.inf
        run_trace = ''
        for trace in packed_traces.keys():
            if packed_traces[trace]['end_time'] < next_time_point:
                run_trace = trace
                next_time_point = packed_traces[trace]['end_time']
        running_time = next_time_point
        available_memory += packed_traces[run_trace]['mem_size']
        plot_points[running_time] = total_memory - available_memory
        packed_traces.pop(run_trace)
        completion_time[run_trace] = next_time_point

    return running_time, sum(memory_impacts.values()), plot_points, sum(completion_time.values()) / len(
        trace_names), completion_time


t, imp, _, act, _ = parallel_paging('opt')
print('With opt green paging, parallel paging total time:        ',
      str(t), '|avg completion time', act, '\nmemory impact:', imp)

t, imp, pp, act, ct_oracle = parallel_paging('deterministic')
print('With deterministic green paging, parallel paging total time:    ',
      str(t), '|avg completion time', act, '\nmemory impact:', imp)
y = 0
ys = []
for x in range(int(t) + 1):
    if x in pp.keys():
        y = pp[x]
    ys.append(y)
plt.bar(range(len(ys)), ys)

t, imp, pp, act, ct_det = parallel_paging('pure-oracle')
print('With pure-oracle green paging, parallel paging total time:',
      str(t), '|avg completion time', act, '\nmemory impact:', imp)

y = 0
ys = []
for x in range(int(t) + 1):
    if x in pp.keys():
        y = pp[x]
    ys.append(y)
plt.bar(range(len(ys)), ys)
plt.xlabel('time')
plt.ylabel('memory size')
plt.legend(['deterministic', 'XGB'])
plt.show()
better = 0
for trace in ct_oracle.keys():
    if ct_oracle[trace] < ct_det[trace]:
        better += 1

print(better)

'''

# plt.yscale('log')


t, imp, pp = parallel_paging('augmented-deterministic')
print('With augmented-deterministic green paging, parallel paging total time:',
      str(t), '| memory impact:', imp)
'''
