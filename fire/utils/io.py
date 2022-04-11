import os
import warnings
from typing import List, Tuple, Union
import pickle as _pickle
import importlib

try:
    from rasterio.errors import NotGeoreferencedWarning
    import rasterio as rio
    import gdal
except ModuleNotFoundError as e:
    warnings.warn("Exception caught during import: " + str(e), UserWarning)

import fire.utils.etc as uetc


def get_subdataset_path(src_filepath: str, sds_id: Union[int,str]) -> str:
    """
    Args:
        sds_id (int or str): If integer, then the zero-based (!) index of 
            subdatasets. If string, then the name of the subdataset. 
    """
    ds = gdal.Open(src_filepath)
    subdatasets: List[Tuple[str,str]] = ds.GetSubDatasets()

    if type(sds_id) is str:
        # access by name
        for sds_path, _ in subdatasets:
            if uetc.extract(sds_path, r"[^:]+$") == sds_id:
                return sds_path
        raise ValueError(f"No subdataset named {sds_id}")
    else:
        # access by index, zero-based
        return subdatasets[sds_id][0] # 1st elem of tuple is sds path

def split_path(fpath: str) -> List[str]:
    """
    Example: "/home/stuff/some_file.txt" -> ["","home","stuff","some_file.txt"]

    Copied from here https://stackoverflow.com/a/16595356/2337838
    license: CC BY-SA 3.0
    """
    fpath = os.path.normpath(fpath)
    return fpath.split(os.sep)

def count_bands(ds: "gdal.Dataset") -> int:
    i = 1 # 1-based index in gdals GetRasterBand
    while(True): 
        if ds.GetRasterBand(i) is None:
            return i-1
        i += 1

def write_lines(l: List[str], fpath:str, append: bool=False) -> None:
    """
    Args:
        l: list to write to fpath
        fpath: path of file to write to
    """
    dirpath = os.path.dirname(fpath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    with open(fpath, "w" if not append else "a") as f:
        for elem in l:
            f.write("%s\n" % elem)
            
def read_lines(fpath: str) -> List[str]:
    """
    Reads lines from file as text and returns
    these as a list.
    """
    with open(fpath, "rt") as f:
        file_content = f.readlines()
    # remove "\n" at the end of each line
    file_content = [line.rstrip("\n") for line in file_content]
    return file_content

def makedirs(filepath:str) -> None:
    """
    Make dirs from a filepath (may include a filename).
    
    Args:
        filepath:
            Creates directories up to the last element
            ending on a seperator (/ or \).
    """
    target_dir = os.path.split(filepath)[0]
    if target_dir != "" and not os.path.exists(target_dir):
        os.makedirs(target_dir)


class _PickleProtocol:
    """
    Stolen from here https://stackoverflow.com/a/60895965/2337838
    """
    def __init__(self, level):
        self.previous = _pickle.HIGHEST_PROTOCOL
        self.level = level

    def __enter__(self):
        importlib.reload(_pickle)
        _pickle.HIGHEST_PROTOCOL = self.level

    def __exit__(self, *exc):
        importlib.reload(_pickle)
        _pickle.HIGHEST_PROTOCOL = self.previous


def pickle_protocol(level):
    """
    Example:
        with pickle_protocol(4):
            some_pandas_dataframe.to_hdf('fname.h5', 'x')

    Stolen from here https://stackoverflow.com/a/60895965/2337838
    """
    return _PickleProtocol(level)