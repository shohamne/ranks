import os
import time
import concurrent.futures
import subprocess
import torch
import queue
import uci_datasets
import copy

from task_queue_gpu import Task, GpuQueue

# Configuration variables
tasks_per_gpu = 3

datasets = [name for name, (n_observations, n_dimensions) in uci_datasets.all_datasets.items() if 500 < n_observations < 2500]
#datasets = ['gas']

def convert_dict_to_cli_args(dict_args):
    cli_args = []
    for key, value in dict_args.items():
        cli_args.extend([f'--{key}', str(value)])
    return cli_args


def run_regression(args):
    name_args = copy.deepcopy(args)
    del name_args['device']
    csv_name = f"{name_args}.csv"

    if os.path.exists(csv_name):
        print(f"Skipping {csv_name}")
        return

    regression_cmd = ["python", "clean.py"] + \
        convert_dict_to_cli_args(args) + \
            ["--csv_name", csv_name]
    print("\n"," ".join(regression_cmd), "\n")
    subprocess.run(regression_cmd, check=True) #stdout=subprocess.DEVNULL)

# Get the number of available GPUs
num_gpus = torch.cuda.device_count()

# Total number of tasks that can run concurrently
total_concurrent_tasks = num_gpus * tasks_per_gpu

# Create a queue of tasks
gpu_queue = GpuQueue(num_gpus, tasks_per_gpu)

gpu_index = 0
for dataset_index, dataset in enumerate(datasets):
    for kernel in ['rbf']:
        for split in range(5):
            for optimize_logdet in [0, 1]:
                for optimize_logdet2 in [0, 1] if optimize_logdet else [0]:
                    for standard_loss in [0, 1] if not optimize_logdet else [0]:
                        for l2_reg in [0.0]:
                            for lr_shift in [10.0, 0.1]:
                                if standard_loss==1:
                                    lr = 0.01*lr_shift
                                else:
                                    lr = 0.001*lr_shift

                                args = {'dataset': dataset, 
                                        'split': split, 
                                        'optimize_logdet': optimize_logdet,
                                        'optimize_logdet2': optimize_logdet2,
                                        'standard_loss': standard_loss,
                                        'l2_reg': l2_reg,
                                        'kernel': kernel,
                                        'lr':lr}
                                task = Task(id=args, processing_function=run_regression, args=args) 
                                gpu_queue.add_task(task)

print(f"Running {gpu_queue.task_queue.qsize()} tasks on {num_gpus} GPUs")
gpu_queue.process_tasks()  # User calls process_tasks when ready                
gpu_queue.wait_for_completion()
print("All tasks completed")