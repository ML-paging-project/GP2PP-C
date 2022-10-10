from collections import OrderedDict
from lru import run_lru


def read_crc_seq(file):
    sequence = []
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            if len(line) == 0:
                continue
            data = line.split(',')
            sequence.append(data[0])
    f.close()
    return sequence


def format_box(t, miss_cost):
    cache = OrderedDict()
    cache_size = t.box_height
    start = t.box_start_pointer
    # reload
    for pid in t.box_start_stack.keys():
        cache[pid] = True
        cache.move_to_end(pid, last=True)
        if len(cache) == cache_size:
            break

    end, rem_width = run_lru(cache, cache_size, t.seq,
                             start, cache_size * miss_cost, 1, miss_cost)
    t.box_end_pointer = end
    if rem_width > 0:
        t.box_width = (t.box_height * 3 * miss_cost - rem_width) / 3 / miss_cost
    else:
        t.box_width = t.box_height

    # Update ending stack
    t.box_end_stack = OrderedDict()
    for pid in t.box_start_stack.keys():
        t.box_end_stack[pid] = True
        t.box_end_stack.move_to_end(pid, last=True)
    for x in range(start, end):
        if t.seq[x] in t.box_end_stack.keys():
            t.box_end_stack.move_to_end(t.seq[x], last=False)
        else:
            t.box_end_stack[t.seq[x]] = True
            t.box_end_stack.move_to_end(t.seq[x], last=False)
    return t
