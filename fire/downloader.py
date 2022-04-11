"""
#todo
"""

from __future__ import annotations

# import numpy as np
from datetime import datetime
from urllib.request import urlopen
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import requests #todo: use either requests or urllib if possible
from netrc import netrc
import io
import os
from queue import Queue
from threading import Thread, Event
from tqdm import tqdm
from typing import List, Set, Dict, Tuple, Optional, Union, Any

# Exceptions
from requests.exceptions import ConnectionError
from requests.exceptions import Timeout, ReadTimeout, ConnectTimeout, TooManyRedirects
from http.client import RemoteDisconnected
from urllib3.exceptions import ProtocolError

# own utils
import fire.utils.etc as uetc
import fire.utils.io as uio
import fire.utils.argprocessing as uarg
import fire.utils.parallel as upar


def collect_hyperlinks(page_url: str) -> List[str]:
    """
    Gets all URLs from hyperlinks (a href) on a given webpage.
    
    Args:
        page_url: The url of the page from which to scrape hyperlinks.
    
    Returns:
        List of str; URLs found in <a href=...> fields on the page.
    """
    urls = []
    page = urlopen( page_url ).read()
    soup = BeautifulSoup(page, "lxml")
    soup.prettify()
    
    for anchor in soup.findAll('a', href=True):
        complete_url = urljoin(page_url, anchor['href'])
        if complete_url not in urls:
            urls.append(complete_url)
    
    return urls


def collect_product_root_urls_from_lpdaac(
    lpdaac_root:   str = "https://e4ftl01.cr.usgs.gov/",
    topdir_regex:  str = r"/[A-Z]+/?$",
    product_regex: str = r"(?<=/)[A-Z_0-9]+\.[0-9]+(?=/$)",
    verbose: bool = True
) -> Dict[str,str]:
    """
    #todo
    """
    topdir_urls = [url for url in collect_hyperlinks(lpdaac_root) 
                   if uetc.like(url, topdir_regex)]
    
    product_urls = []
    for tdu in topdir_urls:
        if verbose:
            print("scraping from", tdu)
        product_urls.extend([url for url in collect_hyperlinks(tdu) 
                             if uetc.like(url, product_regex)])
    
    products = {uetc.extract(url, product_regex) : url for url in product_urls}

    return products


def collect_hdf_urls_from_lpdaac(product_root_url: str, 
                                 hdf_regex: str = r"\.hdf$",
                                 date_regex: str = r"/\d{4}\.\d{2}\.\d{2}/?$",
                                 min_date: Optional[Union[str,datetime]] = None,
                                 max_date: Optional[Union[str,datetime]] = None,
                                 verbose: bool = True,
                                 n_threads: int = 10, 
                                 interrupt_timeout: float=20.0
                                ) -> List[str]:
    """
    Gets the download URLs of all hdf-files for a given 
    product directory in the LP DAAC Data Pool [1]. 
    
    [1] https://lpdaac.usgs.gov/tools/data-pool/
    
    Args:
        product_root_url: 
            URL of the page refering to the product for which you want to 
            collect all hdf URLs. This is the page which presents all the dates 
            for which there is data as subdirectories. E.g. for MOD14A1 it would 
            be (2020-02-11):
            https://e4ftl01.cr.usgs.gov/MOLT/MOD14A1.006/
        hdf_regex:
            Regex-pattern used to filter the URLs found on all the pages for 
            hdf-files.
        verbose:
            Whether or not to show a progress bar.
        min_date, max_date (str or datetime): Dates to use for filtering which 
            directories to fetch. Not too accurate, since the dates are usually 
            only the start dates of the 8-day hdf files. If type is `str`, the 
            date has to be in format `yyyy-mm-dd`, e.g. `2020-12-31`.
            
    Returns:
        List (of str) of URLs pointing to hdf-files.
    """
    # get date directories for this product
    date_urls = collect_hyperlinks(product_root_url)

    # filter LPDAAC dirs from which to collect HDF file urls
    date_urls = [u for u in date_urls if uetc.like(u, date_regex)]

    if min_date:
        min_date  = uarg.to_datetime(min_date)
        date_urls = [u for u in date_urls 
                     if _get_dir_date_from_lpdaac_url(u) >= min_date]

    if max_date:
        max_date  = uarg.to_datetime(max_date)
        date_urls = [u for u in date_urls 
                     if _get_dir_date_from_lpdaac_url(u) <= max_date]
    
    # set up queue
    n = len(date_urls)
    q = Queue(maxsize=0)

    # fill queue with tasks, i.e. date dirs in LPDAAC
    for i,d in enumerate(date_urls):
        q.put((i,d)) # tuple of task_id and date_url

    # init results list (will contain lists of scraped urls later)
    results = [None] * n

    # init progress display
    if verbose:
        progr = uetc.ProgressDisplay(n).start_timer().print_status()
    else:
        progr = None
    
    # init event that will allow to interrupt the process via Ctrl-C
    stop_event = Event()

    # init worker threads
    workers = []
    for _ in range(n_threads):
        wrkr = Thread(target=_collect_hdf_urls_from_lpdaac_worker,
                      kwargs={
                          "q": q, "stop_event": stop_event, 
                          "results": results, "progress_display": progr
                      })
        wrkr.setDaemon(True) # allows to exit a hanging thread
        workers.append(wrkr)

    # let util function do the queue processing 
    # (unlike q.join(), this one allows KeyboardInterrupt)
    upar.process_queue(q, workers, interrupt_timeout=interrupt_timeout)

    # end progress display
    if verbose:
        progr.stop()

    # get all urls from the many lists and put into one list
    failed_scrapes = 0
    hdf_urls = []
    for i,l in enumerate(results):
        if l is None:
            failed_scrapes += 1
        else:
            hdf_urls.extend(l)

    # filter out all urls which do not match arg hdf_regex
    hdf_urls = [u for u in hdf_urls if uetc.like(u, hdf_regex)]
    
    if verbose:
        if failed_scrapes == 0:
            print(f"All {n} date directories were scraped successfully.")
        else:
            print(f"{failed_scrapes} of {n} date directories could not be "
                  "scraped.")
        print(f"Found {len(hdf_urls)} HDF file urls.")
    
    return hdf_urls
    

#todo: log
def fetch_many_files(urls: List[str], target_paths: List[str], auth: Any,
                     n_parallel_downloads: int = 10, max_attempts: int = 3,
                     verbose: bool = True, overwrite_existing: bool = False,
                     interrupt_timeout: float=20.0
                    ) -> List[bool]:
    """
    Args:
        n_parallel_downloads:
            Number of downloads to run in parallel (as threads).

            #todo Beware: Content is downloaded in batches of size
            n_threads and all content of one batch is held in 
            memory until this batch is processed. Only then the
            content is written to disk #todo. 

        max_attempts (int): Max attempts to download a file. Defaults to 3. 
        interrupt_timeout (float): Number of seconds to wait for threads
            to finish their work, when Ctrl-C is pressed.
    
    Returns:
        list: List containing bools and possibly Nones. None means that
            the respective URL could not be fetched after `max_attempts`.

    Details:
        With a lot of help of https://www.shanelynn.ie/using-python-threading-for-multiple-results-queue/
    """
    # check for duplicates in target_paths
    #    these would raise problems when files are written 
    #    (because of possible mutliple writes to one file at once)
    if (len(target_paths)) != len(set(target_paths)):
        raise ValueError("duplicates in target_paths") # or sth else?

    # drop urls if files already exist and if they should not be overwritten
    if not overwrite_existing:
        print("Scanning for existing files...")
        already_exist = [os.path.exists(t) for t in tqdm(target_paths)]
        if sum(already_exist) > 0:
            if verbose:
                print(f"{sum(already_exist)} files already exist.")
            urls         = uetc.select(urls,         uetc.invert(already_exist))
            target_paths = uetc.select(target_paths, uetc.invert(already_exist))
        else:
            print("None of the files to be downloaded do already exist.")
    
    # number of dowloads
    n = len(urls)
    if n == 0:
        if verbose:
            print("Nothing to download.")
        return []

    # init successes list
    successes = [False] * n
    
    # set up queue
    q = Queue(maxsize=0) # infinite max queue size

    # tasks may be repeated for some exceptions. Init list to keep track 
    # of left attempts for each task
    left_attempts = [max_attempts] * n
    
    # fill queue with tasks, i.e. urls and corresponding targets
    enumerated_url_target_tuples = zip(range(n), urls, target_paths)
    for iut_tuple in enumerated_url_target_tuples:
        q.put(iut_tuple) # (i, url, target_path)
        
    # progress display
    if verbose:
        print(f"Downloading {n} files.\n")
        progr = uetc.ProgressDisplay(n).start_timer().print_status()
    else:
        progr = None # is passed to worker thread (see _fetch_files_worker())
    
    # init worker threads
    workers = []
    for _ in range(n_parallel_downloads):
        wrkr = Thread(target=_fetch_files_worker,
                      kwargs={
                          "q": q, "auth": auth, "successes": successes, 
                          "left_attempts": left_attempts, 
                          "overwrite_existing": overwrite_existing, 
                          "progress_display": progr
                      })
        wrkr.setDaemon(True) # allows to exit a hanging thread
        workers.append(wrkr)

    # let util function do the queue processing 
    # (unlike q.join(), this one allows KeyboardInterrupt)
    upar.process_queue(q, workers, interrupt_timeout=interrupt_timeout)

    # end progress display
    if verbose:
        progr.stop()
        print("")
        print(f"\n{sum(successes)}/{n} files downloaded successfully "
              f"({round(100*sum(successes)/n, 2)} %)")
    
    return successes


def fetch_file(url: str, target_path: str, auth: object,
               overwrite_existing: bool = True, 
               return_if_exists: bool = True,
               verbose: bool = True, log: bool = False,
               stop_event: Optional[Event]=None, raise_error: bool=False
              ) -> bool:
    """
    Downloads a file from url and writes it to target_path.
    
    Args:
        url:
            URLs to download.
        target_path:
            The downloaded content will be written to this path. 
        auth: 
            As expected by requests.get, e.g. a Tuple[str, str] with values 
            (user, password)
        return_if_exists:
            If there already is a file at the target path and overwriting is 
            not enabled, this value is returned. If you want to keep track of 
            what files you have locally, set to True. If you want to know
            what has been downloaded during this function call, set to False. 
        verbose:
            Overrides log if verbose is set to False.
        log:
            If True and verbose True as well, warnings (file already exists or 
            bad response) are written to logger instead of being printed to the 
            console. stop_event (Event (opt)): Event instance that is checked 
            regularly, in order to abort the download if signaled. Intended for 
            threading.
    
    Returns:
        True or False indicating whether the file was success-
        fully fetched and written to the target path. 
    """
    if log:
        raise NotImplementedError("log=True not implemented yet")

    if (os.path.exists(target_path) 
        and not overwrite_existing):
        if verbose:
            msg = f"File already exists. {target_path}"
            logging.warning(msg) if log else print(msg)
        return return_if_exists
    
    try:
        # (timeout = (connect timeout, read timeout), both in seconds)
        with requests.get(url, stream=True, auth=auth, 
                          timeout=(20,20)) as response:
            # check for bad response
            if response.status_code != 200:
                if verbose:
                    msg = "Bad response while trying to fetch " \
                         f"(status code: {response.status_code})"
                    logging.warning(msg) if log else print(msg)
                return False
            else:
                uio.makedirs(target_path)

                # download and save as temporary file (*.tmp)
                with open(target_path + ".tmp", 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if (stop_event is not None) and stop_event.is_set():
                            os.remove(target_path + ".tmp")
                            return False
                        f.write(chunk)
                
                # rename to actual target name (remove '.tmp')
                os.rename(target_path + ".tmp", target_path)
                return True
    except (Timeout, ReadTimeout, ConnectTimeout, TooManyRedirects):
        if raise_error:
            raise
        return False
    
    
def get_auth_from_netrc(netrc_token:str, netrc_path:str="~/.netrc") -> Tuple[str,str]:
    """
    Get user and password from netrc file.
    
    Args:
        netrc_token: in netrc files called "machine name"
        
    Returns:
        Tuple[str,str] with values (user,password)
    """
    netrc_path = os.path.expanduser(netrc_path)
    usr = netrc(netrc_path).authenticators(netrc_token)[0]
    pwd = netrc(netrc_path).authenticators(netrc_token)[2]
    return (usr, pwd)




# HIDDEN FUNCTIONS
# ----------------

def _collect_hdf_urls_from_lpdaac_worker(q: Queue, stop_event: Event, 
                                         results: List[str], 
                                         progress_display=None
                                        ) -> None:
    """
    Args:
        progress_display (uetc.ProgressDisplay (opt)): ...
    """
    while not q.empty():
        # get unfinished task from tuple
        i, date_dir = q.get()
        
        # process task, thus a single date dir
        returned_val = collect_hyperlinks(date_dir)
        results[i] = returned_val
        
        # notify queue that task has been processed
        q.task_done()
        
        # update progress display
        if progress_display is not None:
            progress_display.update_and_print()
        
        if stop_event.is_set():
            return


def _fetch_files_worker(q: Queue, 
                        stop_event: Event, 
                        auth: Any,
                        successes: List[Union[bool, None]],
                        left_attempts: List[int],
                        overwrite_existing: bool = False,
                        progress_display: uetc.ProgressDisplay = None
                       ) -> None:
    """
    Args:
        q (Queue): Queue with tasks consisting of `i, url, target_path`.
        successes (list): List of length n_tasks, or put 
            differently: One "slot" for each `i` in the tasks. The value 
            returned by fetch_file() for the i-th task will be stored as 
            the i-th element in this list.
    """
    while not q.empty():
        # get unfinished task from tuple
        task = q.get() # task is a tuple (i, url, target_path)
        i, url, target_path = task
        
        # process task, thus a single url
        try:
            left_attempts[i] -= 1
            success = fetch_file(
                url, target_path, auth, 
                overwrite_existing=overwrite_existing,
                verbose=False, stop_event=stop_event)
            successes[i] = success
            if (not success) and (left_attempts[i] > 0):
                # e.g. due to response != 200
                q.put(task)
                url_terminally_processed = False
            else:
                url_terminally_processed = True
        except (ConnectionError, RemoteDisconnected, ProtocolError):
            # put task back into queue if max number of attempts is not 
            # reached
            if left_attempts[i] > 0:
                q.put(task)
                url_terminally_processed = False
            else:
                url_terminally_processed = True # processed but failed
        
        # notify queue that task has been processed
        q.task_done()
        
        # update progress display
        if (progress_display is not None) and (url_terminally_processed):
            progress_display.update_and_print()

        if stop_event.is_set():
            return


def _get_dir_date_from_lpdaac_url(url: str) -> datetime:
    date_str = uetc.extract(url, r"[12][0-9]{3}\.[01][0-9]\.[0-3][0-9]")
    return datetime.strptime(date_str, r"%Y.%m.%d")
