import concurrent.futures
import subprocess
import torch
import queue
import uci_datasets

# Configuration variables
tasks_per_gpu = 4

datasets = [name for name, (n_observations, n_dimensions) in uci_datasets.all_datasets.items() if 1000 < n_observations < 100000]
#datasets = datasets[:1]
def run_regression_and_visualization(task):
    dataset, beta, start_layer, split, gpu_index = task
    test_name = f"metrics_{dataset}_beta={beta}_start_layer={start_layer}"
    regression_cmd = [
        "python", "regression.py",
        "--dataset", dataset,
        "--hidden_size", "64",
        "--depth", "4",
        "--epochs", "1000",
        "--stop_rank_reg", "1000",
        "--lambdas", "0.1", "1.0", "10.0",
        "--lr", "0.001",
        "--beta", str(beta),
        "--start_layer", str(start_layer),
        "--split", str(split),
        "--device", f"cuda:{gpu_index}",
        "--csv_name", f"{test_name}_split={split}.csv"
    ]

    print(" ".join(regression_cmd))
    subprocess.run(regression_cmd, check=True)

    visualization_cmd = [
        "python", "visualization.py",
        "--test_name", f"{test_name}",
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
    for split in range(10):
        for beta in [0.0, 0.01]:
            for start_layer in ([1, 2, 3] if beta != 0.0 else [3]):
                tasks.put((dataset, beta, start_layer, split, gpu_index))
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
