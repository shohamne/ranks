import concurrent.futures
import subprocess
import torch
import queue
import uci_datasets

# Configuration variables
tasks_per_gpu = 3

datasets = [name for name, (n_observations, n_dimensions) in uci_datasets.all_datasets.items() if 1000 < n_observations < 100000]

def run_regression_and_visualization(task):
    dataset, beta, gpu_index = task

    regression_cmd = [
        "python", "regression.py",
        "--dataset", dataset,
        "--hidden_size", "64",
        "--depth", "4",
        "--epochs", "100",
        "--lambdas", "0.1", "1.0", "10.0",
        "--lr", "0.001",
        "--beta", str(beta),
        "--device", f"cuda:{gpu_index}",
        "--csv_name", f"metrics_{dataset}_beta={beta}.csv"
    ]
    print(" ".join(regression_cmd))
    subprocess.run(regression_cmd, check=True)

    visualization_cmd = [
        "python", "visualization.py",
        "--csv_file", f"metrics_{dataset}_beta={beta}.csv",
        "--pdf_file", f"metrics_{dataset}_beta={beta}.pdf"
    ]
    subprocess.run(visualization_cmd, check=True)

# Get the number of available GPUs
num_gpus = torch.cuda.device_count()

# Total number of tasks that can run concurrently
total_concurrent_tasks = num_gpus * tasks_per_gpu

# Create a queue of tasks
tasks = queue.Queue()

gpu_index = 0
for dataset_index, dataset in enumerate(datasets):
    for beta in [0.0, 0.01]:
        tasks.put((dataset, beta, gpu_index))
        gpu_index = (gpu_index + 1) % num_gpus

# Create a pool of workers
with concurrent.futures.ProcessPoolExecutor(max_workers=total_concurrent_tasks) as executor:
    futures = {}
    for _ in range(total_concurrent_tasks):
        gpu = _ % num_gpus
        if not tasks.empty():
            new_future = executor.submit(run_regression_and_visualization, tasks.get())
            futures[new_future] = gpu

    while futures:
        # Wait for a process to complete
        done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)

        for future in done:
            gpu = futures.pop(future)
            try:
                future.result()  # Get the result or raise the exception
            except Exception as e:
                print(f"Task on GPU {gpu} raised an exception: {e}")

            # If there are more tasks, start a new one
            if not tasks.empty():
                new_future = executor.submit(run_regression_and_visualization, tasks.get())
                futures[new_future] = gpu

print("All tasks completed!")
