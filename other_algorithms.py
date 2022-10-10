import collections
import math
import random

import networkx as nx

from lru import run_lru


def random_pick(sequence, k, number_of_box_kinds, miss_cost):
    print('running random picking...')
    total_impact = 0
    rounds = 100  # number of rounds to evaluate over
    for _ in range(rounds):
        pointer = 0
        global_lru = collections.OrderedDict()
        while pointer < len(sequence):
            start_pointer = pointer
            box_id = random.randint(0, number_of_box_kinds - 1)
            cache_size = k / (2 ** box_id)
            box_width = miss_cost * cache_size
            # Compartmentalization
            # Load top pages from LRU stack.
            mycache = collections.OrderedDict()
            for pid in global_lru.keys():
                mycache[pid] = True
                mycache.move_to_end(pid, last=True)  ###########
                if len(mycache) == cache_size:
                    break

            pointer = run_lru(mycache, cache_size, sequence, pointer, box_width, 1, miss_cost)
            endpointer = pointer

            # Update global stack
            for x in range(start_pointer, endpointer):
                if sequence[x] in global_lru.keys():
                    global_lru.move_to_end(sequence[x], last=False)
                else:
                    global_lru[sequence[x]] = True
                    global_lru.move_to_end(sequence[x], last=False)

            mi = 3 * miss_cost * cache_size * cache_size
            total_impact = total_impact + mi
    # print(total_impact / rounds)
    return total_impact / rounds


def deterministic_algorithm(sequence, k, number_of_box_kinds, miss_cost):
    print('running deterministic algorithm...')
    req2counter = []
    req2box = []
    req2mi = []
    # deterministic algorithm
    countings = [0 for i in range(number_of_box_kinds)]
    countings[0] = 1
    currentbox = 0
    pointer = 0
    total_impact = 0
    box_seq = []
    global_lru = collections.OrderedDict()
    while pointer < len(sequence):
        start_pointer = pointer
        cache_size = k / (2 ** (number_of_box_kinds - currentbox - 1))
        box_width = miss_cost * cache_size
        # Compartmentalization
        # Load top pages from LRU stack.
        mycache = collections.OrderedDict()
        for pid in global_lru.keys():
            mycache[pid] = True
            mycache.move_to_end(pid, last=True)  ###########
            if len(mycache) == cache_size:
                break

        pointer, _ = run_lru(mycache, cache_size, sequence, pointer, box_width, 1, miss_cost)
        box_seq.append(int(number_of_box_kinds - currentbox - 1))
        endpointer = pointer

        # Update global stack
        for x in range(start_pointer, endpointer):
            if sequence[x] in global_lru.keys():
                global_lru.move_to_end(sequence[x], last=False)
            else:
                global_lru[sequence[x]] = True
                global_lru.move_to_end(sequence[x], last=False)

        mi = 3 * miss_cost * cache_size * cache_size
        total_impact = total_impact + mi

        while len(req2mi) < pointer:
            req2mi.append(total_impact)
            req2box.append(currentbox)
            req2counter.append(countings)

        if currentbox == number_of_box_kinds - 1:
            currentbox = 0
        elif countings[currentbox] % 4 == 0:
            currentbox = currentbox + 1
        else:
            currentbox = 0
        countings[currentbox] = countings[currentbox] + 1

    return total_impact, box_seq, req2box, req2counter, req2mi


############################
# run lru, mark down hits and faults
def LRU(sequence, size):
    stack = collections.OrderedDict()
    marks = []
    for r in sequence:
        if r in stack.keys():
            marks.append(True)  # a hit
        else:
            if len(stack.keys()) == size:  # memory is full
                stack.popitem(last=True)
            marks.append(False)  # a fault
            stack[r] = True
        stack.move_to_end(r, last=False)
    return marks


def opt(sequence, k, number_of_box_kinds, miss_cost):
    print('running offline opt')
    # parameters
    n = len(sequence)
    ############################
    print('building dag...')
    # build dag
    dag = nx.DiGraph()
    edge2box = {}  # to reconstruct the opt box sequence
    for i in range(number_of_box_kinds):
        # loop for every row
        end_index = [-1 for _ in range(n)]
        memory_size = math.floor(k / (2 ** i))
        # run lru on the whole sequence
        # with memory_size = k / (2 ** i)
        is_a_hit = LRU(sequence, memory_size)
        # find the end index for req[0]
        box_width = miss_cost * memory_size
        request_position = 0
        running_time = 0
        while running_time < box_width and request_position < n:  # So burst is allowed for the last req
            if is_a_hit[request_position]:
                running_time += 1
            else:
                running_time += miss_cost
            request_position += 1
        end_index[0] = request_position - 1
        dag.add_edge(0, end_index[0] + 1, weight=memory_size * miss_cost * 3 * memory_size)
        edge2box[(0, end_index[0] + 1)] = i

        # find end indexes for other nodes in row i
        for j in range(1, n):
            if is_a_hit[j - 1]:
                running_time = running_time - 1
            else:
                running_time = running_time - miss_cost
            if running_time >= box_width:
                end_index[j] = end_index[j - 1]
            else:
                request_position = end_index[j - 1] + 1
                while running_time < box_width and request_position < n:
                    if is_a_hit[request_position]:
                        running_time += 1
                    else:
                        running_time += miss_cost
                    request_position += 1
                end_index[j] = request_position - 1

            if (j, end_index[j] + 1) in edge2box:
                dag.remove_edge(j, end_index[j] + 1)  # smaller box is preferred
            dag.add_edge(j, end_index[j] + 1, weight=memory_size * miss_cost * 3 * memory_size)
            edge2box[(j, end_index[j] + 1)] = i
    #########################################

    print('searching the shortest path...')
    nodes_sorted = list(nx.topological_sort(dag))
    build_path = {}
    for v in nodes_sorted:
        build_path[v] = (math.inf, -1)  # (distance to the beginning, predecessor)
    build_path[0] = (0, -1)
    for u in nodes_sorted:
        for v in list(dag.successors(u)):
            if build_path[v][0] > build_path[u][0] + dag.edges[u, v]['weight']:
                build_path[v] = (build_path[u][0] + dag.edges[u, v]['weight'], u)

    box_start_points = [n]
    opt_box_seq = []
    while True:
        box_start_points.append(build_path[box_start_points[-1]][1])
        # box_start_points.append()
        opt_box_seq.append(int(edge2box[(box_start_points[-1], box_start_points[-2])]))
        if box_start_points[-1] == 0:
            break
    opt_impact = build_path[n][0]
    opt_box_seq = list(reversed(opt_box_seq))
    box_start_points = list(reversed(box_start_points))
    # opt_path = list(reversed(box_start_points))
    # d = nx.dijkstra_path_length(dag, source='start', target='end')
    # print(d)
    return opt_impact, opt_box_seq, box_start_points
