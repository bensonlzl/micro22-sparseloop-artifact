import os
import yaml, inspect, os, sys, subprocess, pprint, shutil, argparse

from graphing_functions import *

def get_all_files_in_dir(given_dir):
    return sorted([
        os.path.join(
            given_dir,
            file 
        ) for file in os.listdir(given_dir) if 'yaml' in os.path.join(
            given_dir,
            file 
        )
    ])

def run_timeloop(workloads, mappings, config, pe_art, pe_ert, arch, output_dir, experiment_name):
    timeloop_exe = "timeloop-model"
    algorithmic_outputs_list = []
    actual_outputs_list = []
    bandwidth_list = []
    memory_movement_list = []
    cycles_list = []
    labels = []
    for workload,mapping in zip(workloads,mappings):
        lst_of_input_files = [
            workload,
            mapping,
            config,
            pe_art,
            pe_ert,
            arch
        ]
        
        subprocess_cmd = [timeloop_exe, *lst_of_input_files]
       
        status = subprocess.call(subprocess_cmd) 
        print("TIMELOOP-MODEL COMMAND: ", ' '.join(subprocess_cmd))
        
        layer_name = workload.split("/")[-1]
        
        stat_file_name = os.path.join(output_dir, f'{experiment_name}/raw-stats/timeloop-model_{layer_name}.stats.txt')
        
        
        mv_cmd = ['mv','timeloop-model.stats.txt',stat_file_name]
        status = subprocess.call(mv_cmd) 
        
        cycles_list.append(extract_cycles(stat_file_name))
        algorithmic, actual = parse_energy_summary(stat_file_name)
        bandwidth_list.append((parse_bandwidth_stats(stat_file_name)))
        labels.append(layer_name)
        algorithmic_outputs_list.append(algorithmic)
        actual_outputs_list.append(actual)
        memory_movement_list.append(parse_scalar_activity(stat_file_name))
    
    plot_bandwidth_separately(bandwidth_list,labels)
    plot_energy_breakdown_multiple(algorithmic_outputs_list,actual_outputs_list,labels)
    plot_scalar_activity_multiple(memory_movement_list,labels)
    plot_cycle_counts(cycles_list)
    
    
def run_accelergy(arch, components):
    accelergy_exe = "accelergy"
    lst_of_input_files = [
        arch,
        components,
    ]
    subprocess_cmd = [accelergy_exe, *lst_of_input_files]
    
    status = subprocess.call(subprocess_cmd) 
    print("ACCELERGY COMMAND: ", ' '.join(subprocess_cmd))

    
def run_timeloop_mapper(arch, problem, mapper, pe_art, pe_ert, config, constraints):
    timeloop_mapper_exe = "timeloop-mapper"
    lst_of_input_files = [
        arch,
        problem,
        mapper,
        pe_art,
        pe_ert,
        config,
        constraints
    ]
    subprocess_cmd = [timeloop_mapper_exe, *lst_of_input_files]
    status = subprocess.call(subprocess_cmd) 
    print("TIMELOOP-MAPPER COMMAND: ", ' '.join(subprocess_cmd))
    
    
