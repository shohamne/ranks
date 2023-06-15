import concurrent.futures
import queue
import time
import os
from tqdm import tqdm
import functools

class Task:
    def __init__(self, id, processing_function, *args, **kwargs):
        self.id = id
        self.processing_function = processing_function
        self.args = args
        self.kwargs = kwargs

class GpuQueue:
    def __init__(self, num_gpus, tasks_per_gpu):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus*tasks_per_gpu)
        self.task_queue = queue.Queue()
        self.available_gpus = queue.Queue()
        self.completed_tasks = queue.Queue()

        # Initially all GPUs are available tasks_per_gpu times
        for i in range(num_gpus):
            for _ in range(tasks_per_gpu):
                self.available_gpus.put(i)

    def add_task(self, task):
        self.task_queue.put(task)

    def process_tasks(self):
        futures = set()
        with tqdm(total=self.task_queue.qsize(), ncols=70) as pbar:
            while not self.task_queue.empty():
                gpu_index = self.available_gpus.get()  # This will block if no GPUs are available
                task = self.task_queue.get()
                future = self.executor.submit(self.run_task, task, gpu_index)
                future.add_done_callback(functools.partial(self.task_done, task=task, gpu_index=gpu_index, pbar=pbar))
                futures.add(future)
            concurrent.futures.wait(futures)  # Wait for all tasks to complete

    def run_task(self, task, gpu_index):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        return task.processing_function(*task.args, **task.kwargs)

    def task_done(self, future, task, gpu_index, pbar):
        result = future.result()
        self.completed_tasks.put(result)
        pbar.update(1)
        self.available_gpus.put(gpu_index)  # The GPU is now available

    def wait_for_completion(self):
        self.executor.shutdown()

def gpu_task(task_id):
    # replace this with actual processing code
    time.sleep(1)
    return f"Task {task_id} completed"

def main():
    num_gpus = 4  # replace with actual number of GPUs
    tasks_per_gpu = 2  # replace with number of tasks per GPU
    gpu_queue = GpuQueue(num_gpus, tasks_per_gpu)

    for i in range(20):  # create 20 tasks
        task = Task(i, gpu_task, i)
        gpu_queue.add_task(task)

    gpu_queue.process_tasks()  # User calls process_tasks when ready
    gpu_queue.wait_for_completion()

    # Print the results of the completed tasks
    while not gpu_queue.completed_tasks.empty():
        print(gpu_queue.completed_tasks.get())

if __name__ == "__main__":
    main()
