import os
from datetime import datetime, timedelta
import cdsapi
import pickle

import fire.utils.etc as uetc

from typing import Dict, Any, Iterable, List


def download_era5(date: datetime.date, target_dir: str, 
                  options_dict: Dict[str, Any], 
                  data_src_name: str = 'reanalysis-era5-single-levels', 
                  overwrite: bool = False, **kwargs) -> bool:
    """
    Saves options_dict to pickle file with hash as filename.... #todo
    Wrapper for cdsapi.Client.retrieve()

    Args:
        date (datetime.date): A single date for which ERA5 data will be downloa-
            ded. Overwrites `year`, `month`, and `day` keys in `options_dict` 
            (after pickling the dict).
        target_dir (str): 
        options_dict (dict): 
        data_src_name (str (opt)): 
        overwrite (bool): If True, a file that already exists at the target
            path will be overwritten. Defaults to False. 
        **kwargs: Keyword args passed to cdsapi.Client()

    Returns:
        bool: True, if the file was downloaded successfully or (if overwrite=True)
              if the file already existed. Otherwise False.
    """
    # generate hash for options_dict
    options_dict_hash = uetc.dict_to_hash(options_dict, sort_keys=True)
    
    # save options_dict to target_dir as pickle if not exists
    opt_dict_pickle_fname = f"options_dict_{options_dict_hash}.p"
    opt_dict_pickle_fpath = os.path.join(target_dir, opt_dict_pickle_fname)
    if not os.path.exists(opt_dict_pickle_fpath):
        with open(opt_dict_pickle_fpath, mode="wb") as pickle_file:
            pickle.dump(options_dict, pickle_file)
    
    # date as str
    date_str = date.strftime("%Y-%m-%d")
    
    # filename
    fname = date_str + "_" + options_dict_hash + ".grib"
    fpath = os.path.join(target_dir, fname)
    
    if os.path.exists(fpath) and not overwrite:
        return True
    
    c = cdsapi.Client(**kwargs)
    
    c.retrieve(
        data_src_name,
        {
            "year": str(date.year),
            "month": "%02d" % date.month,
            "day": "%02d" % date.day,
            **options_dict
        },
        fpath + ".tmp"
    )
    
    if os.path.exists(fpath + ".tmp"):
        os.rename(fpath + ".tmp", fpath)
        return True
    else:
        return False


def batch_download_era5(dates: Iterable[datetime.date], target_dir: str, 
                        options_dict: Dict[str, Any]) -> List[bool]:
    """
    Wrapper for `download_era5`.
    """
    n = len(dates)
    successes = [False] * n
    
    progr = uetc.ProgressDisplay(n).start_timer().print_status()
    for i, d in enumerate(dates):
        successes[i] = download_era5(d, target_dir, 
                                     options_dict,
                                     # cdsapi.Client kwargs:
                                     progress=False, verify=True, quiet=True)
        progr.update_and_print()
    progr.stop()
    
    n_fails = len([s for s in successes if not s])
    print("Finished")
    print(f"{n_fails} of {n} downloads failed.")