#!/usr/bin/env python3

import re
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from collections import namedtuple

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 200


scipp_data = """1300,3
1320,164
1340,387
1360,39
1380,76
1400,175
1420,24219
1440,217860
1460,7402107
1480,726029
1500,219091
1520,106858
1540,31043
1560,19714
1580,39411
1600,68654
1620,87174
1640,95791
1660,102404
1680,96257
1700,95016
1720,100485
1740,103312
1760,102195
1780,101099
1800,95017
1820,47429
1840,1996
1860,6765
1880,1452
1900,2307
1920,10614
1940,29050
1960,43068
1980,17153
2000,1292
2020,724
2040,885
2060,283
1000000000,2402"""

def pt2in(pt):
    return float(pt) / 72.0


def unit(size):
    idx = int(np.log2(size) // 10)
    unit = ['', 'k', 'M', 'G'][idx]
    return "{}{}".format(size // (1024 ** idx), unit)



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

            if len(re.findall(r'dot', m.string)) > 0:
                dash = 3

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

groups = {}
groups['local'] = DataGroup(palette['orange'], "solid", 'D', '////')
groups['local-alt'] = DataGroup(palette['yellow'], "densely dashdotted", 'D', '////')
groups['remote'] = DataGroup(palette['blue'], "dashed", 'x', '\\\\\\\\')
groups['remote-alt'] = DataGroup(palette['purple'], "dashed", 'x', '\\\\\\\\')
groups['gpu'] = DataGroup(palette['green'], "densely dotted", 'o', '+++')

maxwidth = pt2in(345)

mediancolor = "#303030"
medianwidth = 1.2


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

    on = 1 * width
    off = 1 * width
    ax.grid(axis=grid_axis, ls='--', lw=width, c=gridcolor, dashes=(on,off,on,off), zorder=0)

    ax.minorticks_off()
    ax.tick_params(axis='both', which='major', color=color, width=width, labelsize=6.5)

    for spine in ax.spines.values():
        spine.set_linewidth(width)
        spine.set_color(color)


def draw_line(ax, point, linestyle, vertical):
    draw_func = ax.axvline if vertical else ax.axhline
    draw_func(point, linestyle=linestyle, linewidth=medianwidth, color=mediancolor, zorder=1005,
              path_effects=[pe.Stroke(linewidth=medianwidth+.4, foreground="#b0b0b0"), pe.Normal()],
             )


def plot_median_line(ax, datasets, groups, labels):

    for idx, dataset in enumerate(datasets):
        x = sorted(dataset.keys())
        y = []

        for pos in x:
            y.append(np.median(dataset[pos]))

        group = groups[idx]
        label = labels[idx]

        marker = group.marker
        marker_ec = group.edgecolor
        marker_fc = group.fillcolor
        marker_ew = medianwidth + .4

        ax.plot(x, y, group.marker, linestyle=group.linestyle, linewidth=medianwidth, color=group.color, label=label,
                path_effects=[pe.Stroke(linewidth=medianwidth+.6, foreground=group.edgecolor), pe.Normal()],
                markersize=5, markeredgecolor=marker_ec, markerfacecolor=marker_fc, markeredgewidth=marker_ew)

        ax.legend(loc='lower right', fontsize=7, framealpha=1, handlelength=4.5)


def plot_histograms(ax, datasets, groups, labels, range, bins=1000, orientation='vertical', show_median=False):
    nbins = bins

    alpha = "90"

    for idx, dataset in enumerate(datasets):
        group = groups[idx]

        N, bins, containers = ax.hist(dataset, bins=nbins, orientation=orientation, range=range, label=labels[idx], hatch=group.hatch,
                fc=group.fillcolor + alpha, ec=group.edgecolor, lw=.7, color=group.fillcolor + alpha, histtype='stepfilled', zorder=1000)

    ax.legend(loc='upper right', fontsize=7, framealpha=1)

    if show_median:
        for idx, group in enumerate(groups):
            dataset = datasets[idx]
            median = np.median(dataset)
            draw_line(ax, median, group.linestyle, orientation == "vertical")


def plot_boxes(ax, datasets, groups, labels, show_median=False, show_outliers=False):
    if show_median:
        for idx, group in enumerate(groups):
            dataset = datasets[idx]
            median = np.median(dataset)
            draw_line(ax, median, group.linestyle, False)

    boxplot = ax.boxplot(datasets, showfliers=show_outliers, labels=labels, notch=False, patch_artist=True, zorder=1000)

    for idx, box in enumerate(boxplot['boxes']):
        group = groups[idx]
        box.set_facecolor(group.fillcolor)
        box.set_edgecolor(group.edgecolor)
        box.set_hatch(group.hatch)
        box.set_linewidth(medianwidth + .2)

    for idx, median in enumerate(boxplot['medians']):
        group = groups[idx]
        median.set_color(group.edgecolor)
        median.set_linewidth(medianwidth + .4)
        median.set_linestyle("solid")

    for idx, whisker in enumerate(boxplot['whiskers']):
        group = groups[idx//2]
        whisker.set_color(group.edgecolor)
        whisker.set_linewidth(medianwidth + .2)

    for idx, cap in enumerate(boxplot['caps']):
        group = groups[idx//2]
        cap.set_color(group.edgecolor)
        cap.set_linewidth(medianwidth + .2)
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


def parse_gpu_bench(path, datacol='bandwidth', gpu=0):
    """
    Parse GPU bandwidth measurements.
    """
    expr = re.compile(r'^\s*(?P<src>\w+|\d+)\s+(?P<dst>\w+|\d+)\s+(?P<mode>DtoH|HtoD|DtoD)\s+(?P<size>\d+)\s+(?P<usecs>\d+\.\d+)\s+(?P<bw>\d+\.\d+)$')
    dataset_dtoh = {}
    dataset_htod = {}

    with open(path) as f:

        # Skip over header lines
        while True:
            line = f.readline()
            if len(line) == 0:
                break

            match = re.match(r'^([^=]+)=(.*)$', line)
            if not match:
                f.seek(f.tell() - len(line))
                break

        # Read column definition
        f.readline()
        line = f.readline()
        columns = line.split()

        # Parse data lines
        while True:
            line = f.readline()
            if len(line) == 0:
                break

            match = expr.match(line)
            if not match:
                continue

            if match.group('src') == str(gpu) and match.group('dst') == "host":
                ds = dataset_dtoh
            elif match.group('dst') == str(gpu) and match.group('src') == "host":
                ds = dataset_htod
            else:
                continue

            size, bw = int(match.group('size')), float(match.group('bw'))

            if size < 4096:
                continue

            if not size in ds:
                ds[size] = []

            ds[size].append(bw)

    for size, data in dataset_htod.items():
        dataset_htod[size] = np.array(data)

    for size, data in dataset_dtoh.items():
        dataset_dtoh[size] = np.array(data)

    return dataset_htod, dataset_dtoh



def parse_latency_bench(path, datacol='time'):
    """
    Parse values from nvm-latency-bench run.
    """

    dataset = []

    expr = re.compile(r'\s*([^;]+);')

    with open(path) as f:
        # Skip over header lines
        while True:
            line = f.readline()
            if len(line) == 0:
                break

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
    local = []
    remote = []
    gpu = []
    for buffer_gpu in "local", "peer":
        local.append(parse_latency_bench(f'results/nvme-sq/ssd-queue=local-gpu={buffer_gpu}.txt'))
        remote.append(parse_latency_bench(f'results/nvme-sq/ssd-queue=remote-gpu={buffer_gpu}.txt'))
        gpu.append(parse_latency_bench(f'results/nvme-sq/ssd-queue=c0e00-gpu={buffer_gpu}.txt'))

    local = np.concatenate(local)
    remote = np.concatenate(remote)
    gpu = np.concatenate(gpu)

    grouping = [groups['local-alt'], groups['remote-alt'], groups['gpu']]
    labels = ["Borrower RAM", "Lender RAM", "Borrowed GPU"]
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
        #tick.label.set_weight('bold')
        tick.label.set_weight('normal')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    save_figure("nvme-sq")


def lending_gpu():
    local1_fname = 'results/zero-overhead/local-vs-remote/bw-topo=local-gpu0=P4000-gpu1=P4000-switch=yes-borrower=petty-borrower_iommu=yes-lender_a=-lender_b=-lender_a_iommu=no-lender_b_iommu=no-p2p=no-dtoh=yes-htod=yes.txt'
    local2_fname = 'results/zero-overhead/local-vs-remote/bw-topo=local-gpu0=P4000-gpu1=P4000-switch=yes-borrower=petty-borrower_iommu=yes-lender_a=-lender_b=-lender_a_iommu=no-lender_b_iommu=no-p2p=no-dtoh=yes-htod=yes.txt'

    remote1_fname = 'results/zero-overhead/local-vs-remote/bw-topo=1L-gpu0=P4000-gpu1=P4000-switch=yes-borrower=petty-borrower_iommu=yes-lender_a=betty-lender_b=-lender_a_iommu=no-lender_b_iommu=no-p2p=no-dtoh=yes-htod=yes.txt'
    remote2_fname = 'results/zero-overhead/local-vs-remote/bw-topo=1L-gpu0=P4000-gpu1=P4000-switch=yes-borrower=petty-borrower_iommu=yes-lender_a=betty-lender_b=-lender_a_iommu=no-lender_b_iommu=no-p2p=no-dtoh=yes-htod=yes.txt'

    local_htod1, local_dtoh1 = parse_gpu_bench(local1_fname, gpu=0)
    #local_htod2, local_dtoh2 = parse_gpu_bench(local2_fname, gpu=1)
    remote_htod1, remote_dtoh1 = parse_gpu_bench(remote1_fname, gpu=0)
    #remote_htod2, remote_dtoh2 = parse_gpu_bench(remote2_fname, gpu=1)

    keys = [(1 << n) for n in range(12, 28)]

    grouping = [groups['local'], groups['remote']]
    labels = ["Local Baseline", "Device Lending"]

    fig = plt.figure(figsize=(maxwidth, pt2in(175)))

    ax = fig.subplots(1, 1)

    prepare_axis(ax)

    ax.set_xscale('log', base=2)
    ax.set_xlim(keys[0] - (keys[0] >> 2), keys[-1] + keys[-2])
    ax.set_xticks(keys)
    ax.set_xticklabels((unit(size) for size in keys), rotation=47)
    ax.set_xlabel("Transfer size (B)")

    ax.set_ylabel("DMA throughput (GB/s)")

    ax.set_ylim(0, 13 * 1024)
    yticks = range(0, 14 * 1024, 2048)
    ax.set_yticks(yticks)
    ax.set_yticklabels((int(i // 1024) for i in yticks))

    plot_median_line(ax, [local_dtoh1, remote_dtoh1], grouping, labels)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    save_figure("lending-gpu")


def scipp():
    buckets = []
    counts = []

    for line in scipp_data.split('\n')[:-1]:
        bucket, count = [int(d) for d in line.split(',')]
        buckets.append(bucket)
        counts.append(count)

    print(buckets[counts.index(max(counts))])

    fig = plt.figure(figsize=(pt2in(300), pt2in(160)))
    ax = fig.subplots(1, 1)

    prepare_axis(ax)

    ax.set_xlim(buckets[0], buckets[-1])
    ax.set_xlabel("Ping-pong latency (nanoseconds)")

    ax.set_ylim(0, 8*10**6)
    ax.set_ylabel("Number of ping-pongs (in millions)")
    ax.set_yticks([i * 10**6 for i in range(9)])
    ax.set_yticklabels(range(9))

    ax.bar(buckets, counts, width=20, color="#2ad4ff", ec="#303030")
    fig.tight_layout()

    plt.savefig("scipp.png", dpi=600, format='png', bbox_inches='tight')


def lending_gpu_slides():
    local1_fname = 'results/zero-overhead/local-vs-remote/bw-topo=local-gpu0=P4000-gpu1=P4000-switch=yes-borrower=petty-borrower_iommu=yes-lender_a=-lender_b=-lender_a_iommu=no-lender_b_iommu=no-p2p=no-dtoh=yes-htod=yes.txt'
    local2_fname = 'results/zero-overhead/local-vs-remote/bw-topo=local-gpu0=P4000-gpu1=P4000-switch=yes-borrower=petty-borrower_iommu=yes-lender_a=-lender_b=-lender_a_iommu=no-lender_b_iommu=no-p2p=no-dtoh=yes-htod=yes.txt'

    remote1_fname = 'results/zero-overhead/local-vs-remote/bw-topo=1L-gpu0=P4000-gpu1=P4000-switch=yes-borrower=petty-borrower_iommu=yes-lender_a=betty-lender_b=-lender_a_iommu=no-lender_b_iommu=no-p2p=no-dtoh=yes-htod=yes.txt'
    remote2_fname = 'results/zero-overhead/local-vs-remote/bw-topo=1L-gpu0=P4000-gpu1=P4000-switch=yes-borrower=petty-borrower_iommu=yes-lender_a=betty-lender_b=-lender_a_iommu=no-lender_b_iommu=no-p2p=no-dtoh=yes-htod=yes.txt'

    local_htod1, local_dtoh1 = parse_gpu_bench(local1_fname, gpu=0)
    #local_htod2, local_dtoh2 = parse_gpu_bench(local2_fname, gpu=1)
    remote_htod1, remote_dtoh1 = parse_gpu_bench(remote1_fname, gpu=0)
    #remote_htod2, remote_dtoh2 = parse_gpu_bench(remote2_fname, gpu=1)

    keys = [(1 << n) for n in range(12, 28)]

    grouping = [groups['local'], groups['remote']]
    labels = ["Local Baseline", "Device Lending"]

    fig = plt.figure(figsize=(pt2in(540), pt2in(160)))

    ax = fig.subplots(1, 1)

    prepare_axis(ax)

    ax.set_xscale('log', base=2)
    ax.set_xlim(keys[0] - (keys[0] >> 2), keys[-1] + keys[-2])
    ax.set_xticks(keys)
    ax.set_xticklabels((unit(size)+"B" for size in keys), rotation=0)
    #ax.set_xlabel("Transfer size (B)")

    ax.set_ylabel("DMA throughput (GB/s)")

    ax.set_ylim(0, 13 * 1024)
    yticks = range(0, 14 * 1024, 2048)
    ax.set_yticks(yticks)
    ax.set_yticklabels((int(i // 1024) for i in yticks))

    plot_median_line(ax, [local_dtoh1, remote_dtoh1], grouping, labels)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)

    plt.savefig("lending.png", dpi=600, format='png', bbox_inches='tight')
    #save_figure("lending-gpu")

#lending_nvme()
#smartio_driver_sq_figure()
#lending_gpu()
#lending_gpu_slides()

scipp()

plt.show()
