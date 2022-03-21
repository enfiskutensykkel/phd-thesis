#!/usr/bin/env python3

import re
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 200


def pt2in(pt):
    return float(pt) / 72.0


maxwidth = pt2in(345)


def prepare_axis(ax, yaxis):
    """
    Prepare axis so the look similar for all plots.
    """
    width = 0.8
    color = "#4d4d4d"

    ax.xaxis.tick_bottom()
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_label_position('bottom')

    ax.yaxis.tick_left()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('left')

    for axis in ax.xaxis, ax.yaxis:
        axis.label.set_size(7)
        axis.label.set_weight('medium')
        axis.label.set_family('sans')

    grid = 'y' if not yaxis else 'x'
    ax.grid(axis=grid, ls=':', lw=width, c=color)

    ax.minorticks_off()
    ax.tick_params(axis='both', color=color, width=width)

    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        tick.label.set_family('sans')
        tick.label.set_weight('medium')
        tick.label.set_size(6.5)

    for spine in ax.spines.values():
        spine.set_linewidth(width)
        spine.set_color(color)



def parse_fio_latency(path):
    """
    Parse latency values from FIO file.
    """
    expr = re.compile(r'\s*([^,]+)[,\n]')
    ds = []
    col = 1 # column 1 is the values we are looking for

    with open(path) as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break

            datacols = expr.findall(line)
            value = datacols[col]
            ds.append(float(value) / 1000.0)

    return np.array(ds)


def save_figure(name):
    plt.savefig(f"figures/{name}.pdf", dpi=600, format='pdf', bbox_inches='tight')


def lending_nvme_figure():
    local = []
    remote = []

    for root, dirs, files in os.walk("results/lending-nvme"):
        if "sync_lat.1.log" in files:
            path = root + "/sync_lat.1.log"
            if re.match(r'lending-nvme/borrow-\d+-\d+-\d+', root):
                remote = np.append(remote, parse_fio_latency(path))
            else:
                local = np.append(local, parse_fio_latency(path))

    fig = plt.figure(figsize=(maxwidth, pt2in(200)))

    ax = fig.subplots(1, 1)

    prepare_axis(ax, True)
    ax.set_ylabel("Number of I/O operations")
    ax.set_xlabel("Latency distribution (µs)")


    fig.tight_layout()
    save_figure("lending-nvme")


lending_nvme_figure()
plt.show()
