"""
With code from here: https://gist.github.com/mgmarino/2773137
"""

from typing import Tuple, List, Iterable, Any, Optional
from mpi4py import MPI
import traceback
import pandas as pd
import numpy as np
import fire.utils.io as uio


mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

WORKTAG = 0
DIETAG  = 1


class Logger:
    """
    Args:
        fpath (str): path of log file
        log_only (bool, opt): If False, messages will also be printed. 
    """
    def __init__(self, fpath: str, log_only: bool=False, 
                 add_date_default: bool=True):
        self.fpath = fpath
        self.log_only = log_only
        self.add_date_default = add_date_default

    def log(self, msg: str, add_date: Optional[bool]=None):
        add_date = self.add_date_default if add_date is None else add_date
        if add_date:
            msg = str(pd.Timestamp.now()) + "   " + msg
        if not self.log_only:
            print(msg)
        uio.write_lines([msg], fpath=self.fpath, append=True)


class TaskQueue:
    def __init__(self, tasks: list, ids: Optional[list]=None):
        assert isinstance(tasks, list), "tasks must be of type list"
        self.tasks = tasks.copy()
        if ids is None:
            self.ids = list(range(len(tasks)))
        else:
            assert isinstance(ids, list), "if ids is passed it must be of type list"
            assert len(ids) == len(tasks), "tasks and ids are not of same length"
            self.ids = ids.copy()

    def get_next(self) -> Tuple[Any, Any]:
        if len(self.tasks) == 0:
            return None
        else:
            return (self.ids.pop(), self.tasks.pop())

    def __len__(self):
        return len(self.tasks)


def send_to_workers_and_collect_results(queue: TaskQueue, 
                                        logger: Optional[Logger]=None,
                                        max_tries: int=1
                                       ) -> List[Tuple[Any,Any]]:
    """
    Workers are expected to receive whatever TaskQueue.get_next() returns, i.e.
    tuples of int (task_id) and Any (the actual task). The latter is passed to 
    the function that is processing single tasks.

    Args:
        max_tries (int, optional):
            If >1, a task will be sent out again to workers if processing it 
            failed the first time.

    Details:
        The order of workers is shuffled at which they get their tasks. This 
        increases the probability that some worker that failed on a task due to 
        a possibly worker-specific error (e.g. out of memory) doesn't get the
        same task over and over again (and always fails while another worker 
        might be able to process the task successfully).
    """
    results = [] # (task_id, result)
    fails   = [] # (task_id, task)
    status  = MPI.Status()
    n_tasks = len(queue)
    n_workers = mpi_size-1

    if logger is not None:
        logger.log(f" * Got {n_tasks} tasks for {n_workers} workers.")

    # send first batch of tasks. Will break if n_tasks < n_workers
    if logger is not None:
        logger.log(" * Send first batch of tasks.")

    # see details in doc
    shuffled_worker_ids = np.random.choice(range(1, n_workers+1), size=n_workers, 
                                           replace=False)
    for i in shuffled_worker_ids: 
        next_task = queue.get_next() 
        if next_task is None:
            break
        mpi_comm.send(obj=next_task, dest=i, tag=WORKTAG)
    
    # for all tasks that weren't send yet, send them to worker x as soon as worker 
    # x sends a result (and would thus idle)
    if n_tasks > n_workers:
        if logger is not None:
            logger.log(" * More tasks than workers. Waiting for workers to idle.")
        while True:
            next_task = queue.get_next() 
            if next_task is None:
                break
            _collect_results_and_fails(results, fails)
            mpi_comm.send(obj=next_task, dest=status.Get_source(), tag=WORKTAG)
    
    # no more tasks to send, so collect remaining results
    if logger is not None:
        logger.log(" * No more tasks to send. Waiting for tasks to be finished.")
    for _ in range(1, min(n_workers, n_tasks)+1):
        _collect_results_and_fails(results, fails)

    # handle failed task executions
    if len(fails) > 0:
        if logger is not None:
            logger.log(f" * {len(fails)} tasks failed.")
        if max_tries > 1:
            if logger is not None:
                logger.log(f" * Trying again ({max_tries-1} left).")
            retry_queue = TaskQueue(
                tasks=[t for i,t in fails], ids=[i for i,t in fails]
            )
            results.extend(send_to_workers_and_collect_results(
                queue=retry_queue, logger=logger, max_tries=max_tries-1))
        else:
            if logger is not None:
                logger.log(f" * Reached maximum number of tries. Aborting.")
            make_workers_to_stop(logger)
            raise RuntimeError("Reached max number of tries.")
    
    if (len(results) < n_tasks) and (logger is not None):
        logger.log(" ! len(results) < n_tasks")
    return results

def _collect_results_and_fails(results: List[Tuple[int,Any]], 
                               fails: List[Tuple[int,Any]]) -> None:
    """
    Appends results and fails in-place to the respective lists.
    """
    status = MPI.Status()
    task_id, result, success = mpi_comm.recv(source=MPI.ANY_SOURCE, 
                                             tag=MPI.ANY_TAG, status=status)
    if success:
        results.append((task_id, result))
    else:
        # variable `result` contains the task that was sent in the first place
        fails.append((task_id, result)) 

def make_workers_to_stop(logger: Optional[Logger]=None):
    """
    Workers are expected to stop whenever they receive `None` instead of a usual
    task.
    """
    if logger is not None:
        logger.log(" * Making workers to stop.")
    for i in range(1,mpi_size):
        mpi_comm.send(obj=None, dest=i, tag=DIETAG)

def worker(f: callable):
    """
    Handles the communication of the worker with the master. `f` is the function
    that will be called on each task (not `task_id`!). Worker can be stopped 
    by sending `None` and thus also via `make_workers_to_stop`.
    """
    status = MPI.Status()
    # process tasks as long as deliveries are not None
    while True:
        delivery = mpi_comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if delivery is None: 
            break # if None was sent, stop
        task_id, task = delivery

        success = False
        try:
            result  = f(task) # <= the actual work
            success = True
        except:
            result = task
            print(" *** CAUGHT AN EXCEPTION: *** ")
            print(traceback.format_exc())
        mpi_comm.send(obj=(task_id, result, success), dest=0)
