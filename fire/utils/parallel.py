"""
"""

from typing import Callable, Iterable, Mapping, Any, List, Optional

from queue import Queue
from threading import Thread, Event
import time # sleep

import fire.utils.etc as uetc


def process_tasks(tasks: Iterable[Any], f: Callable, f_kwargs={}, 
                  n_workers: int=4, interrupt_timeout=10, 
                  progress=True) -> List[Any]:
    n = len(tasks)
    results = [None] * n # init results list

    # set up queue
    q = Queue(maxsize=0) # infinite max queue size
    
    # fill queue with tasks, i.e. urls and corresponding targets
    for i, t in enumerate(tasks):
        q.put((i, t))

    # progress display
    progr = uetc.ProgressDisplay(ntotal=n, disable=not progress).start_timer()

    # init worker threads
    workers = []
    for _ in range(n_workers):
        wrkr = Thread(target= simple_worker,
                      kwargs={"f": f, "q": q, "progr": progr,
                              "results": results, 
                              "task_kwargs": f_kwargs})
        wrkr.setDaemon(True) # allows to exit a hanging thread
        workers.append(wrkr)

    # let util function do the queue processing 
    # (unlike q.join(), this one allows KeyboardInterrupt)
    process_queue(q, workers, interrupt_timeout=interrupt_timeout)

    # end progress display
    progr.stop()

    return results

def process_queue(q: Queue, threads: Iterable[Thread], verbose: bool=True, 
                  interrupt_timeout: float=10.0) -> None:
    """
    Kind of like Queue.join(), but allows KeyboardInterrupts. 
    
    Target functions in all threads must have an argument 
    `stop_event: threading.Event` and must check regularly if 
    `stop_event.is_set()` is True, and if so, stop their work as soon as 
    possible. This allows the KeyboardInterrupt to do what it is meant for.
    
    Args:
        q (queue.Queue): The queue to be processed completely. 
        threads (iterable of threading.Thread): Iterable, e.g. list, containing 
            Thread objects WHICH ARE NOT STARTED YET. They probably should be
            daemonic as well. (?)
        interrupt_timeout (float): Number of seconds to wait for threads
            to finish their work, when Ctrl-C is pressed. 
            
    Details:
        This function will add one more thread to the ones passed via 
        `threads`, which monitors whether the queue is empty. `Queue.join()` 
        normally does this, but can't be used here since it does not allow for 
        KeyboardInterrupts to work.
    """
    # init event that will signal all threads to stop at some point
    stop_event = Event()
    
    try:
        # start thread that will monitor the queue
        monitor_thread = Thread(target=_signal_when_queue_is_empty, 
                                kwargs={"q": q, "stop_event": stop_event})
        monitor_thread.setDaemon(True)
        monitor_thread.start()
        
        # pass stop_event to user defined threads/workers and start them
        for worker in threads:
            worker._kwargs["stop_event"] = stop_event
            worker.start()
            
        # wait until any thread sets the stop_event to True 
        # (monitor_thread will do this once the queue is empty)
        stop_event.wait()
        
    except KeyboardInterrupt:
        if verbose:
            print("\nCrtl-C pressed. Stopping... "
                  f"(may take {round(interrupt_timeout, 2)} seconds)")
        stop_event.set()  # inform all threads to stop now
        still_alive_threads = _wait_for_threads_to_terminate(
            threads, timeout=interrupt_timeout)
        if verbose:
            if still_alive_threads > 0:
                print(f"\n{still_alive_threads} threads are still running.")
            else:
                print("\nAll threads stopped cleanly.")
            
    return

def simple_worker(f: Callable, q: Queue, stop_event: Event, 
                  results: List[str], task_kwargs={},
                  progr: Optional[uetc.ProgressDisplay]=None) -> None:
    """
    [summary]

    Parameters
    ----------
    f : Callable
        function with signature `f(task: Any, **task_kwargs)` -> Any.
    q : Queue
        has to be filled with (i,task) tuples. `task` will be passed to `f`.
    stop_event : Event
        [description]
    results : List[str]
        [description]
    task_kwargs : dict, optional
        [description], by default {}
    progr : Optional[uetc.ProgressDisplay], optional
        [description], by default None
    """
    while not q.empty():
        # get unfinished task from tuple
        i, task = q.get()
        
        # process task
        results[i] = f(task, **task_kwargs)
        
        # notify queue that task has been processed
        q.task_done()
        
        # update progress bar
        if progr is not None:
            progr.update_and_print(1)
        
        if stop_event.is_set():
            return
        
def _signal_when_queue_is_empty(q: Queue, stop_event: Event, sleep: float=1.0
                               ) -> None:
    """
    Monitors a queue and signals all other threads that have access to 
    `stop_event` when the queue is empty. The main process is intended to wait 
    for `stop_event.is_set()` to turn True to resume. Required in 
    `process_queue`. 

    Args:
        q (Queue): Queue to monitor.
        stop_event (Event): Event instance that is shared between all threads,
            in order to broadcast when the threads should stop their work. 
        sleep (float): Number of seconds between check if queue is empty.
    """
    while True:
        if q.empty():
            q.join()
            stop_event.set()
            return
        if stop_event.is_set():
            return
        time.sleep(sleep)


def _wait_for_threads_to_terminate(threads: Iterable[Thread], 
                                   timeout: float=5.0, 
                                   seconds_between_checks: float=.5
                                  ) -> int:
    """
    Returns:
        int: Number of threads that are still alive after waiting.
    """
    time_left = timeout
    while (_num_of_alive_threads(threads) > 0) and (time_left > 0):
        # print(f"\ntime_left: {time_left}")
        if time_left < seconds_between_checks:
            time.sleep(time_left)
            break
        else:
            time.sleep(seconds_between_checks)
            time_left -= seconds_between_checks

    return _num_of_alive_threads(threads)


def _num_of_alive_threads(threads: Iterable[Thread]) -> int:
    return sum([thr.is_alive() for thr in threads])