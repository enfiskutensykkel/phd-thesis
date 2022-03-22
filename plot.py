#!/usr/bin/env python3

import re
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 200


def pt2in(pt):
    return float(pt) / 72.0


maxwidth = pt2in(345)


Palette = namedtuple('Palette', ['fill', 'stroke', 'alt'])


class DataGroup:
    """
    How should data be presented?
    """

    def __init__(self, color, linestyle, marker, hatch, offset=0):
        self._color = color
        self.marker = marker
        self.hatch = hatch

        dot = 1
        dash = 5
        modifier = 3

        self.linestyle = None
        if linestyle == "solid":
            self.linestyle = (0, ())
        else:
            m = re.match(r'^(loosely\s|densely\s)?(dash|dot)*?(dashed|dotted)\s*$', linestyle)

            if not m.group(1) is None:
                if m.group(1).startswith("loosely"):
                    modifier = 10
                elif m.group(1).startswith("densely"):
                    modifier = 1

            sequence = []
            for dot_or_dash in re.findall(r'(dot|dash)', m.string):
                if dot_or_dash == "dot":
                    sequence += [dot, modifier]
                else:
                    sequence += [dash, modifier]

            self.linestyle = (offset, tuple(sequence))


    @property
    def color(self):
        return self._color.alt

    @property
    def fillcolor(self):
        return self._color.fill

    @property
    def edgecolor(self):
        return self._color.stroke


palette = {'orange': Palette('#ffccaa', '#d45500', '#ff7f2a'),
           'yellow': Palette('#ffeeaa', '#d4aa00', '#ddbc36'),
           'blue': Palette('#b0c4de', '#0055d4', '#5599ff'),
           'purple': Palette('#e5d5ff', '#8e1e92', '#da80f0'),
           'green': Palette('#e3f4d7', '#5aa02c', '#8dd35f'),
          }

groups = {'local': DataGroup(palette['orange'], "densely dashdotted", 'D', '////'),
          'remote': DataGroup(palette['blue'], "dashed", 'x', '\\\\\\\\'),
          'gpu': DataGroup(palette['green'], "densely dotted", 'o', '++++'),
         }

darkgray = "#303030"


def prepare_axis(ax, grid_axis='both'):
    """
    Prepare axis so the look similar for all plots.
    """
    width = 0.8
    color = "#4d4d4d"
    gridcolor = "#b0b0b0"

    ax.xaxis.tick_bottom()
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')

    ax.yaxis.tick_left()
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_label_position('left')

    for axis in ax.xaxis, ax.yaxis:
        axis.label.set_size(6.5)
        axis.label.set_weight('medium')
        axis.label.set_family('sans')

    on = 2 * width
    off = 4 * width
    ax.grid(axis=grid_axis, ls='--', lw=width, c=gridcolor, dashes=(on,off,on,off), zorder=0)

    ax.minorticks_off()
    ax.tick_params(axis='both', which='major', color=color, width=width, labelsize=6.5)

    for spine in ax.spines.values():
        spine.set_linewidth(width)
        spine.set_color(color)


def plot_histograms(ax, datasets, groups, labels, range, bins=1000, orientation='vertical', show_median=False):
    nbins = bins

    alpha = "90"

    for idx, dataset in enumerate(datasets):
        group = groups[idx]

        N, bins, containers = ax.hist(dataset, bins=nbins, orientation=orientation, range=range, label=labels[idx], hatch=group.hatch,
                fc=group.fillcolor + alpha, ec=group.edgecolor, lw=.7, color=group.fillcolor + alpha, histtype='stepfilled', zorder=1000)

    ax.legend(labels, loc='upper right', fontsize=7, framealpha=1)

    if show_median:
        func = ax.axvline if orientation == 'vertical' else ax.axhline

        for idx, group in enumerate(groups):
            dataset = datasets[idx]
            median = np.median(dataset)

            func(median, linestyle=group.linestyle, linewidth=1.2, color=darkgray, zorder=1005)


def plot_boxes(ax, datasets, groups, labels, show_median=False):
    if show_median:
        for idx, group in enumerate(groups):
            dataset = datasets[idx]
            median = np.median(dataset)

            ax.axhline(median, linestyle=group.linestyle, linewidth=1, color=darkgray)

    boxplot = ax.boxplot(datasets, showfliers=False, labels=labels, notch=False, patch_artist=True)

    for idx, box in enumerate(boxplot['boxes']):
        group = groups[idx]
        box.set_facecolor(group.fillcolor)
        box.set_edgecolor(group.edgecolor)
        box.set_hatch(group.hatch)

    for idx, median in enumerate(boxplot['medians']):
        group = groups[idx]
        median.set_color(darkgray)
        median.set_linewidth(1.2)
        median.set_linestyle(group.linestyle)

    for idx, whisker in enumerate(boxplot['whiskers']):
        group = groups[idx//2]
        whisker.set_color(group.edgecolor)

    for idx, cap in enumerate(boxplot['caps']):
        group = groups[idx//2]
        cap.set_color(group.edgecolor)
        cap.set_linewidth(1.2)
        cap.set_xdata(cap.get_xdata() + np.array([-.05, .05]))



def parse_fio_log(path, datacol=1):
    """
    Parse values from FIO run log file.
    """
    expr = re.compile(r'\s*([^,]+)[,\n]')
    dataset = []

    with open(path) as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break

            datacols = expr.findall(line)
            value = datacols[datacol]
            dataset.append(float(value) / 1000.0)

    return np.array(dataset)


def parse_latency_bench(path, datacol='time'):
    """
    Parse values from nvm-latency-bench run.
    """

    metadata = {}
    dataset = []

    expr = re.compile(r'\s*([^;]+);')

    with open(path) as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break

            # Try to read header lines
            match = re.match(r'^###', line)
            if not match:
                f.seek(f.tell() - len(line))
                break

        # Read column definitions
        line = f.readline()
        columns = expr.findall(line)

        columnmap = dict((name, i) for i, name in enumerate(columns))
        columnidx = columnmap[datacol]

        # Read data lines
        while True:
            line = f.readline()
            if len(line) == 0:
                break

            data = expr.findall(line)
            dataset.append(float(data[columnidx]))

    return np.array(dataset)



def save_figure(name):
    plt.savefig(f"figures/{name}.pdf", dpi=600, format='pdf', bbox_inches='tight')


def lending_nvme():
    local = []
    remote = []

    for root, dirs, files in os.walk("results/lending-nvme"):
        if "sync_lat.1.log" in files:
            path = root + "/sync_lat.1.log"
            if re.match(r'results/lending-nvme/borrow-\d+-\d+-\d+', root):
                remote = np.append(remote, parse_fio_log(path))
            else:
                local = np.append(local, parse_fio_log(path))

    fig = plt.figure(figsize=(maxwidth, pt2in(175)))

    ax = fig.subplots(1, 1)

    prepare_axis(ax)
    ax.set_ylabel("Number of I/O operations")
    ax.set_xlabel("Latency distribution (µs)")

    grouping = [groups['local'], groups['remote']]
    plot_histograms(ax, [local, remote], grouping, ["Local Baseline", "Device Lending"], (14.5, 20.5), bins=200)

    fig.tight_layout()
    save_figure("lending-nvme")



def smartio_driver_sq_figure():
    local = parse_latency_bench('results/nvme-sq/ssd-queue=local-gpu=local.txt')
    remote = parse_latency_bench('results/nvme-sq/ssd-queue=remote-gpu=local.txt')
    gpu = parse_latency_bench('results/nvme-sq/ssd-queue=c0e00-gpu=local.txt')

    grouping = [groups['local'], groups['remote'], groups['gpu']]
    labels = ["Local RAM", "Remote RAM", "Remote GPU"]
    datasets = [local, remote, gpu]

    fig = plt.figure(figsize=(maxwidth, pt2in(200)))

    hist, box = fig.subplots(1, 2, sharey=True)

    prepare_axis(hist, grid_axis='y')
    hist.set_ylabel("Latency distribution (µs)")
    hist.set_xlabel("Number of I/O operations")
    hist.set_title("Histogram", fontsize=7, weight='medium', family='sans', pad=3)

    plot_histograms(hist, datasets, grouping, labels, (8, 9),
            bins=200, orientation="horizontal", show_median=True)

    prepare_axis(box, grid_axis='y')
    box.yaxis.tick_right()
    box.set_ylabel("Latency distribution (µs)")
    box.yaxis.set_label_position('right')
    box.set_title("Boxplot (outliers removed)", fontsize=7, weight='medium', family='sans', pad=3)

    plot_boxes(box, datasets[::-1], grouping[::-1], [l.replace(" ", "\n") for l in labels[::-1]], show_median=True)

    for tick in box.xaxis.get_major_ticks():
        tick.label.set_weight('bold')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    save_figure("nvme-sq")


lending_nvme()
smartio_driver_sq_figure()
plt.show()
