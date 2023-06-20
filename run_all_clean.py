import time
import concurrent.futures
import subprocess
import torch
import queue
import uci_datasets

from task_queue_gpu import Task, GpuQueue

# Configuration variables
tasks_per_gpu = 2

datasets = [name for name, (n_observations, n_dimensions) in uci_datasets.all_datasets.items() if 500 < n_observations < 10000]
#datasets = ['gas']

def convert_dict_to_cli_args(dict_args):
    cli_args = []
    for key, value in dict_args.items():
        cli_args.extend([f'--{key}', str(value)])
    return cli_args


def run_regression(args):
    regression_cmd = ["python", "clean.py"] + \
        convert_dict_to_cli_args(args) + \
            ["--csv_name", f"{args}.csv"]
    #print("\n"," ".join(regression_cmd), "\n")
    subprocess.run(regression_cmd, check=True)#, stdout=subprocess.DEVNULL)

# Get the number of available GPUs
num_gpus = torch.cuda.device_count()

# Total number of tasks that can run concurrently
total_concurrent_tasks = num_gpus * tasks_per_gpu

# Create a queue of tasks
gpu_queue = GpuQueue(num_gpus, tasks_per_gpu)

gpu_index = 0
for dataset_index, dataset in enumerate(datasets):
    for kernel in ['linear', 'rbf']:
        for optimise_logdet in [0, 1]:
            for split in range(5):
                args = {'dataset': dataset, 
                        'split': split, 
                        'optimise_logdet': optimise_logdet,
                        'kernel': kernel}
                task = Task(id=args, processing_function=run_regression, args=args) 
                gpu_queue.add_task(task)

print(f"Running {gpu_queue.task_queue.qsize()} tasks on {num_gpus} GPUs")
gpu_queue.process_tasks()  # User calls process_tasks when ready                
gpu_queue.wait_for_completion()
print("All tasks completed")