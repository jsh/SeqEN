#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from torch import diagonal, eye, fliplr, index_select, mode, tensor


def consensus(output, ndx, w, device):
    output_length = len(output)
    seq_length = output_length + w - 1
    filter_size = min(seq_length - ndx, ndx + 1)
    if filter_size > w:
        filter_size = w
    r_min = max(0, ndx - w + 1)
    r_max = r_min + filter_size
    r_indices = tensor(range(r_min, r_max), device=device)
    c_min = max(0, ndx - output_length + 1)
    c_max = min(ndx, w - 1) + 1
    c_indices = tensor(range(c_min, c_max), device=device)
    sub_result = index_select(index_select(output, 0, r_indices), 1, c_indices)
    return mode(diagonal(fliplr(fliplr(eye(filter_size, device=device).long()) * sub_result))).values.item()


def consensus_acc(seq, output, w, device):
    n = 0
    consensus_seq = []
    for i, ndx in enumerate(seq):
        consensus_seq.append(consensus(output, i, w, device=device))
        if ndx.item() == consensus_seq[-1]:
            n += 1
    return n / len(seq), consensus_seq
