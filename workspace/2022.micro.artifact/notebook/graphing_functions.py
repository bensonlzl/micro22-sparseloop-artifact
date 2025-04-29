import pandas as pd
import numpy as np
import scipy.sparse as ss
import yaml, inspect, os, sys, subprocess, pprint, shutil, argparse

import matplotlib.pyplot as plt
import re
from collections import defaultdict


# Energy breakdown 
def parse_energy_summary(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    algorithmic = {
        'total_computes': None,
        'total_energy_per_compute_pJ': None,
        'breakdown': {}
    }
    actual = {
        'total_computes': None,
        'total_energy_per_compute_pJ': None,
        'breakdown': {}
    }

    in_algo = False
    in_actual = False

    for line in lines:
        stripped = line.strip()

        # Header flags
        if stripped.startswith("Algorithmic Computes ="):
            algorithmic['total_computes'] = int(stripped.split('=')[1].strip())
            in_algo = True
            in_actual = False
            continue

        if stripped.startswith("Actual Computes ="):
            actual['total_computes'] = int(stripped.split('=')[1].strip())
            in_algo = False
            in_actual = True
            continue

        # Breakdown parsing
        if in_algo and re.match(r'^.+?=\s+[\d.]+$', stripped):
            key, val = map(str.strip, stripped.split(' = '))
            val = float(val)
            if "<==>" in key:
                continue
            if key == "Total":
                algorithmic['total_energy_per_compute_pJ'] = val
            else:
                algorithmic['breakdown'][key] = val

        elif in_actual and re.match(r'^.+?=\s+[\d.]+$', stripped):
            key, val = map(str.strip, stripped.split(' = '))
            val = float(val)
            if "<==>" in key:
                continue
            if key == "Total":
                actual['total_energy_per_compute_pJ'] = val
            else:
                actual['breakdown'][key] = val

    return algorithmic, actual



def plot_energy_breakdown_multiple(algorithmic_list, actual_list, labels):
    assert len(algorithmic_list) == len(actual_list) == len(labels), "Each algorithmic/actual entry must have a corresponding label."
    num_configs = len(algorithmic_list)

    def extract_data(data_list, exclude_backing=False):
        all_components = set()
        for entry in data_list:
            for k in entry['breakdown'].keys():
                if not (exclude_backing and k == 'BackingStorage'):
                    all_components.add(k)
        components = sorted(all_components)

        data = []
        for entry in data_list:
            data.append([entry['breakdown'].get(comp, 0.0) if comp in entry['breakdown'] else 0.0 for comp in components])
        return components, list(zip(*data))  # Transposed: component-wise, not config-wise

    def plot_chart(ax1, ax2, algo_data, actual_data, components, title_suffix):
        x = list(range(num_configs))
        width = 0.6

        # --- Algorithmic chart ---
        bottoms = [0] * num_configs
        for i, comp_vals in enumerate(algo_data):
            ax1.bar(x, comp_vals, width, bottom=bottoms, label=components[i], color=f"C{i}")
            bottoms = [bottoms[j] + comp_vals[j] for j in range(num_configs)]
        ax1.set_title(f"Algorithmic Energy Breakdown {title_suffix}")
        ax1.set_ylabel("Energy per Compute (pJ)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45)
        ax1.grid(axis='y', linestyle='--', alpha=0.6)

        # --- Actual chart ---
        bottoms = [0] * num_configs
        
        for i, comp_vals in enumerate(actual_data):
            print(f"{i=},{comp_vals=}")
            ax2.bar(x, comp_vals, width, bottom=bottoms, label=components[i], color=f"C{i}")
            bottoms = [bottoms[j] + comp_vals[j] for j in range(num_configs)]
        ax2.set_title(f"Actual Energy Breakdown {title_suffix}")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45)
        ax2.grid(axis='y', linestyle='--', alpha=0.6)

        return components

    # === First plot: with BackingStorage ===
    comp_all, algo_data_all = extract_data(algorithmic_list, exclude_backing=False)
    print(comp_all)
    _, actual_data_all = extract_data(actual_list, exclude_backing=False)

    fig1, (ax1a, ax2a) = plt.subplots(1, 2, figsize=(14, 6))
    plot_chart(ax1a, ax2a, algo_data_all, actual_data_all, comp_all, "(All Components)")
    fig1.legend(comp_all, loc='center right', bbox_to_anchor=(1.12, 0.5))
    fig1.tight_layout()
    plt.show()

    # === Second plot: excluding BackingStorage ===
    comp_excl, algo_data_excl = extract_data(algorithmic_list, exclude_backing=True)
    _, actual_data_excl = extract_data(actual_list, exclude_backing=True)

    fig2, (ax1b, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
    plot_chart(ax1b, ax2b, algo_data_excl, actual_data_excl, comp_excl, "(Excluding BackingStorage)")
    fig2.legend(comp_excl, loc='center right', bbox_to_anchor=(1.12, 0.5))
    fig2.tight_layout()
    plt.show()


    
# Average read/write bandwidth

def parse_bandwidth_stats(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Split the content by component sections
    components = re.split(r"=== (.+?) ===", content)[1:]  # [component_name, block, component_name, block, ...]

    bandwidth_data = {}

    for i in range(0, len(components), 2):
        comp_name = components[i].strip()
        comp_block = components[i+1]

        # Find the Average Read Bandwidth and Average Write Bandwidth blocks
        read_match = re.search(r"Average Read Bandwidth \(per-instance\).*?: ([\d.]+) words/cycle\s+Breakdown \(Data, Format\): \(([\d.]+)%, ([\d.]+)%\)", comp_block)
        write_match = re.search(r"Average Write Bandwidth \(per-instance\).*?: ([\d.]+) words/cycle\s+Breakdown \(Data, Format\): \(([\d.]+)%, ([\d.]+)%\)", comp_block)

        if read_match and write_match:
            read_bw, read_data_pct, read_format_pct = map(float, read_match.groups())
            write_bw, write_data_pct, write_format_pct = map(float, write_match.groups())

            bandwidth_data[comp_name] = {
                'read': {
                    'total': read_bw,
                    'data': read_bw * (read_data_pct / 100),
                    'format': read_bw * (read_format_pct / 100)
                },
                'write': {
                    'total': write_bw,
                    'data': write_bw * (write_data_pct / 100),
                    'format': write_bw * (write_format_pct / 100)
                }
            }
    return bandwidth_data


def plot_bandwidth_separately(bandwidth_data_list, labels=None):
    num_runs = len(bandwidth_data_list)
    components = list(bandwidth_data_list[0].keys())
    bar_width = 0.3

    for comp in components:
        read_data_vals = [bd[comp]['read']['data'] for bd in bandwidth_data_list]
        read_format_vals = [bd[comp]['read']['format'] for bd in bandwidth_data_list]
        write_data_vals = [bd[comp]['write']['data'] for bd in bandwidth_data_list]
        write_format_vals = [bd[comp]['write']['format'] for bd in bandwidth_data_list]

        x = np.arange(num_runs)

        fig, ax = plt.subplots(figsize=(12, 4))

        # Stack bars per run
        r1 = ax.bar(x, read_data_vals, bar_width, label='Read Data', color='tab:blue')
        r2 = ax.bar(x, read_format_vals, bar_width, bottom=read_data_vals, label='Read Format', color='lightblue')
        r3 = ax.bar(x, write_data_vals, bar_width, bottom=np.array(read_data_vals) + np.array(read_format_vals),
                    label='Write Data', color='tab:orange')
        r4 = ax.bar(x, write_format_vals, bar_width,
                    bottom=np.array(read_data_vals) + np.array(read_format_vals) + np.array(write_data_vals),
                    label='Write Format', color='moccasin')

        ax.set_ylabel('Bandwidth (words/cycle)')
        ax.set_title(f'Bandwidth Breakdown for Component: {comp}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels if labels else [f'Run {i+1}' for i in x])
        ax.legend(loc='center right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        
# Scalar reads, writes and updates

def parse_scalar_activity(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Split the content by component sections
    components = re.split(r"=== (.+?) ===", content)[1:]  # [component_name, block, component_name, block, ...]

    scalar_activity = {}

    for i in range(0, len(components), 2):
        comp_name = components[i].strip()
        comp_block = components[i + 1]

        # Extract actual + skipped reads
        actual_reads = re.search(r"Actual scalar reads \(per-instance\)\s*:\s*(\d+)", comp_block)
        skipped_reads = re.search(r"Skipped scalar reads \(per-instance\)\s*:\s*(\d+)", comp_block)

        # Extract actual + skipped writes (fills)
        actual_writes = re.search(r"Actual scalar fills \(per-instance\)\s*:\s*(\d+)", comp_block)
        skipped_writes = re.search(r"Skipped scalar fills \(per-instance\)\s*:\s*(\d+)", comp_block)

        # Extract actual + skipped updates
        actual_updates = re.search(r"Actual scalar updates \(per-instance\)\s*:\s*(\d+)", comp_block)
        skipped_updates = re.search(r"Skipped scalar updates \(per-instance\)\s*:\s*(\d+)", comp_block)

        # Only include if at least one set is found
        if any([actual_reads, actual_writes, actual_updates]):
            scalar_activity[comp_name] = {
                "reads": {
                    "actual": int(actual_reads.group(1)) if actual_reads else 0,
                    "skipped": int(skipped_reads.group(1)) if skipped_reads else 0
                },
                "writes": {
                    "actual": int(actual_writes.group(1)) if actual_writes else 0,
                    "skipped": int(skipped_writes.group(1)) if skipped_writes else 0
                },
                "updates": {
                    "actual": int(actual_updates.group(1)) if actual_updates else 0,
                    "skipped": int(skipped_updates.group(1)) if skipped_updates else 0
                }
            }

    return scalar_activity

def plot_scalar_activity_multiple(scalar_stats_list, labels=None):
    num_runs = len(scalar_stats_list)
    components = list(scalar_stats_list[0].keys())
    num_components = len(components)
    x = np.arange(num_components)
    bar_width = 0.8 / num_runs

    def get_values(op):
        actual = []
        skipped = []
        for run in scalar_stats_list:
            actual.append([run[c][op]['actual'] for c in components])
            skipped.append([run[c][op]['skipped'] for c in components])
        return actual, skipped

    def plot_stacked_grouped(actual_list, skipped_list, title, ylabel):
        fig, ax = plt.subplots(figsize=(12, 5))
        for i in range(num_runs):
            offset = (i - num_runs / 2) * bar_width + bar_width / 2
            ax.bar(x + offset, actual_list[i], bar_width, label=f"{labels[i]} Actual" if labels else f"Run {i+1} Actual", color='tab:green')
            ax.bar(x + offset, skipped_list[i], bar_width, bottom=actual_list[i],
                   label=f"{labels[i]} Skipped" if labels else f"Run {i+1} Skipped", color='lightgreen')

        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45, ha='right')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # Extract and plot each operation
    reads_actual, reads_skipped = get_values('reads')
    writes_actual, writes_skipped = get_values('writes')
    updates_actual, updates_skipped = get_values('updates')

    plot_stacked_grouped(reads_actual, reads_skipped, "Scalar Reads per Component", "Reads (per-instance)")
    plot_stacked_grouped(writes_actual, writes_skipped, "Scalar Writes (Fills) per Component", "Writes (per-instance)")
    plot_stacked_grouped(updates_actual, updates_skipped, "Scalar Updates per Component", "Updates (per-instance)")


def extract_cycles(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    match = re.search(r"Cycles\s*:\s*(\d+)", content)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("Number of cycles not found in the file.")

def plot_cycle_counts(cycle_counts, labels=None):
    x = np.arange(len(cycle_counts))
    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.bar(x, cycle_counts, color='skyblue', edgecolor='black')

    ax.set_ylabel('Cycles')
    ax.set_title('Total Cycles per Run')
    ax.set_xticks(x)
    ax.set_xticklabels(labels if labels else [f'Run {i+1}' for i in x])
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values above bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,}',  # formatted with commas
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
