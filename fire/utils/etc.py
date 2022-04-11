# from __future__ import annotations

from copy import deepcopy
import numpy as np
import pandas as pd
import re
import time
import binascii # for crc32 hashes
import datetime
import io

from typing import List, Iterable, Any, Tuple, Optional, Dict, Union



def max_precision(x) -> float:
    """
    Returns x as numpy.float128 if possible, otherwise as 
    numpy.float64 (depends on OS whether float128 is supported)
    """
    try:
        return np.float128(x)
    except AttributeError:
        return np.float64(x)

def like(x:str, pattern:str) -> bool:
    """
    Checks whether a regex-pattern matches a string.
    """
    return re.search(pattern, x) is not None

def extract(x:str, pattern:str) -> str:
    """
    Extracts a regex-match from a string or returns 
    None if there is no match.
    """
    match_obj = re.search(pattern, x)
    if match_obj is None:
        return None
    else:
        return match_obj.group()
    
def into_chunks(l, n) -> List: #todo return type is generator (?)
    """ 
    Split up a list into chunks of size n (or smaller for the last bit)
    
    tags: batches

    stolen from
    https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def select(x: Iterable[Any], sel: Iterable[bool]) -> List[Any]:
    """
    Returns a list of all elements in `x` for which the respective element in 
    `sel` is True. `sel` is an Iterable of equal length.
    """
    return [elem for elem,s in zip(x,sel) if s]

def remove_keys(d: Dict, keys: Iterable, copy: bool=True) -> Dict:
    """
    Removes keys from a dictionary and returns the altered dictionary. 
    If `copy=False` the original dictionary is modified (*with* side effects).
    """
    if copy:
        d = deepcopy(d)
    for k in keys:
        d.pop(k, None)
    return d

#todo rename negate
def invert(x: Iterable[bool]) -> List[bool]:
    """
    Negates all bools in x and returns them as a list.
    """
    return [not elem for elem in x]

def pool(a: np.ndarray, size: Tuple[int, int], func: str) -> np.ndarray:
    """
    Pools a 2D-array ("matrix"). The stepsize for an axis is the 
    size of the filter along that axis. 
    
    Args:
        a (numpy.ndarray): A numpy array with shape (n,m).
        size (tuple of (int,int)): Shape of the 2d-filter, (p,q), where 
            both p and q are integer mutliples of n and m, i.e. 
            `n % p == 0` and `m % q == 0`.
        func (str): The down-sampling function to use. One of 
            `max, mean, min, sum, median, any`.
    
    Returns:
        numpy.ndarray: #todo
    """
    available_functions = { # if sth is added, add to docstr as well
        "max": np.max, "mean": np.mean, "min": np.min, "sum": np.sum,
        "median": np.median, "any": np.any
    }
    func = available_functions[func]
    
    stack = [] # one array for each elem in one filter position
    for i in range(size[0]):
        for j in range(size[1]):
            stack.append(a[i::size[0],j::size[1]])
    
    stack = np.stack(stack, axis=0)
    return func(stack, axis=0)

def get_2d_neighbors(x: np.ndarray, i: Union[int, Iterable[int]], 
                     j: Union[int, Iterable[int]], dist: int, 
                     return_center: bool=False, 
                     assert_within_edges: bool=True, 
                     outside_val: int=0) -> np.ndarray:
    """
    Extracts the values of all neighbors of the elements at the passed indices.

    Parameters
    ----------
    x : 2d array : shape (n_rows, n_columns)
        [description]
    i : int or array-like of int : length n_idx
        row indices of elements for which neighbors shall be extracted
    j : int or array-like of int : length n_idx
        column indices of elements for which neighbors shall be extracted
    dist : int
        Neighborhood distance. How many rows and columns away an element is 
        still considered a neighbor.
    return_center : bool
        If True, the elements at (i,j) are returned as well. By default False,
        thus neighbors-only.
    assert_within_edges : bool, optional
        Raises an exception in case neighbors would be looked for outside of x, 
        by default True.
    outside_val : (same dtype as x), optional
        Value to return for lookups (uhm... imaginary neighbors?) that are 
        outside x, by default 0. Beware, if x is int, outside_val can't be 
        `np.nan`. Convert to float first.

    Returns
    -------
    2d array : shape (length n_idx, n_neighbors)
        [description]
    """
    i, j = np.atleast_1d(i), np.atleast_1d(j)
    nrows, ncols = x.shape
    if assert_within_edges:
        assert (np.all(i-dist >= 0) and np.all(i+dist < nrows)), \
            "some row indices are too close to the edge for the passed distance"
        assert (np.all(j-dist >= 0) and np.all(j+dist < ncols)), \
            "some col indices are too close to the edge for the passed distance"
    
    # pad x with zeros around it (dist rows and cols in each direction) so that 
    # x can be shifted by dist in each direction
    xpad = np.pad(x, dist, constant_values=outside_val)

    values = []
    for hshift in range(-dist, dist+1):
        for vshift in range(-dist, dist+1):
            if return_center or not ((hshift==0) & (vshift==0)):
                values.append(xpad[dist+hshift:, dist-vshift:][i,j])
    return np.vstack(values).T

def str_to_hash(s: str) -> str:
    s_as_bytes = s.encode("utf-8")
    crc32_hash = hex(binascii.crc32(s_as_bytes)) # "0x..."
    return crc32_hash[2:].upper() # without "0x" and upper case

def bytes_to_hash(b: bytes) -> str:
    crc32_hash = hex(binascii.crc32(b)) # "0x..."
    return crc32_hash[2:].upper() # without "0x" and upper case

def dict_to_hash(d: dict, sort_keys: bool = True, 
                 _return_dict_as_str: bool=False) -> str:
    """
    Generates a hash for a dictionary. The keys are sorted alphabetically first 
    such that the order in which these were entered doesn't matter.
    
    Args:
        sort_keys (bool): Defaults to True.
        _return_dict_as_str (bool): used by `list_of_dicts_to_hash`.
    """
    keys = list(d.keys())
    if sort_keys:
        keys.sort()
        
    dict_as_str = ""
    for k in keys:
        val = d[k]
        if type(val) is np.ndarray:
            val = val.tobytes()
        elif callable(val):
            val = val.__name__ # function name
        dict_as_str += str(k) + ": "
        dict_as_str += str(val) + ",\n"
            
    return str_to_hash(dict_as_str)

def list_of_dicts_to_hash(ld: List[Dict]) -> str:
    dicts_as_str = "\n".join([dict_to_hash(d) for d in ld])
    return str_to_hash(dicts_as_str)

def torch_state_dict_to_hash(sd: "OrderedDict") -> str:
    """
    Args:
        sd (OrderedDict): state_dict as returned by `torch.nn.Module` instances.
    """
    as_bytes = io.BytesIO()
    for k,v in sd.items(): # k: str, v: torch.Tensor
        as_bytes.write(str.encode(k))
        as_bytes.write(v.numpy().tobytes())
    return bytes_to_hash(as_bytes.getvalue())

def isotonic_regression_to_hash(ir: "IsotonicRegression") -> str:
    """
    Creates a hash that identifies the state of an instance of 
    `sklearn.isotonic.IsotonicRegression`. 

    Parameters
    ----------
    ir : sklearn.isotonic.IsotonicRegression
        instance of IsotonicRegression. Must be sklearn version >= 0.24.1

    Returns
    -------
    str
        CRC32 hash of bytes obtained from learnable attributes of 
        IsotonicRegression instance
    """
    as_bytes = io.BytesIO()
    for attr in ["X_min_", "X_max_", "X_thresholds_", "y_thresholds_", "increasing_"]:
        val = getattr(ir, attr)
        as_bytes.write(np.array(val).tobytes())
    return bytes_to_hash(as_bytes.getvalue())

def date_to_datetime(dt: datetime.date) -> datetime.datetime:
    return datetime.datetime.combine(dt, datetime.datetime.min.time())

def first_and_last_date_of_month(year: int, month: int
                                ) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Returns the first and the last day of a given year and month.
    """
    first = pd.Timestamp(year=year, month=month, day=1)
    if month < 12:
        first_of_next_month = pd.Timestamp(year=year, month=month+1, day=1,
                                           freq="D")
    else:
        first_of_next_month = pd.Timestamp(year=year+1, month=1, day=1,
                                           freq="D")
    last = first_of_next_month - (1*first_of_next_month.freq)
    return first, last

def year_day_to_timestamp(year: Union[int,Iterable[int]], 
                          day: Union[int,Iterable[int]],
                          always_as_iterable: bool=False
                         ) -> Union[pd.Timestamp, pd.DatetimeIndex]:
    """
    Args:
        year (int or array-like of int):
            years
        day (int or array-like of int):
            day of year (1-366)
    """
    years, days = np.atleast_1d(year), np.atleast_1d(day)
    assert len(years) == len(days), \
        f"year and day must be of same length, got {len(years)} and {len(days)}"
    datestrings = [np.nan if d<1 else f"{y}{d:03d}" for y, d in zip(years, days)]
    ts = pd.to_datetime(datestrings, format="%Y%j")
    if (len(ts) == 1) and not always_as_iterable:
        return ts[0]
    else:
        return ts

def replace_by_nan(a: np.ndarray, nan_vals: Iterable[Any], 
                   *args, **kwargs) -> np.ndarray:
    """
    Replaces specific values in a numpy array by `numpy.nan`.

    Args:
        a (array-like): 
        nan_vals (list): Values which should be replaced by `numpy.nan`. Values 
            are compared with `numpy.isclose`.
        *args, **kwargs: Arguments passed to `numpy.isclose`.
    Returns:
        numpy.ndarray with dtype float (numpy.nan doesn't work with int)
    """
    nan_mask = np.full(a.shape, False)
    for nv in nan_vals:
        nan_mask |= np.isclose(a, nv, *args, **kwargs)
    a = a.astype(float) # does a copy
    a[nan_mask] = np.nan
    return a

def replace(a: np.ndarray, replacements: Dict[float,float], *args, **kwargs
           ) -> np.ndarray:
    """
    Replaces specific values in a numpy array by values specified in a 
    dictionary. Values to be replaced are selected via `np.is_close`.

    Parameters
    ----------
    a : np.ndarray
        The array in which values are to be replaced.
    replacements : dict
        Dictionary that maps from the values to be replaced (the keys) to the 
        corresponding replacements (values of the dict). 
    *args, **kwargs
        passed to `np.is_close`.

    Returns
    -------
    np.ndarray
        Copy of `a` with replaced values.
    """
    b = a.copy()
    for k,v in replacements.items():
        b[np.isclose(a, k, *args, **kwargs)] = v
    return b

def recycle_list_to_len(x: list, n: int) -> list:
    """
    Repeats a list until it is of length n (must fit exactly).
    """
    n0 = len(x)
    if n % n0 != 0:
        raise ValueError(f"list of length {n0} can't be recycled to length {n} "
                         "(n must be integer multiple of `len(x)`)")
    return x * (n//n0)

def rnd_bool(n: int, n_true: int=None, p_true: float=None, seed=None
            ) -> np.ndarray:
    """
    Generates a bool vector of length n with either a specific number of true 
    values (set n_true) or a given probability of True occuring (set p_true). 
    """
    if (n_true is not None) and (p_true is not None):
        raise ValueError("EITHER n_true OR p_true must be given")
    
    rng = np.random.RandomState(seed)
    if n_true is not None:
        rnd_idx = rng.choice(np.arange(0,n), replace=False, size=n_true)
        bools = np.zeros(n, dtype=bool)
        bools[rnd_idx] = True
        return bools
    elif p_true is not None:
        bools = rng.choice([True,False], size=n, p=[p_true, 1-p_true])
        return bools
    else:
        raise ValueError("either n_true or p must be passed")

def make_seed(rng: Optional[np.random.RandomState]) -> int:
    """
    Returns an int that can be used as seed in `np.random.RandomState`. Useful 
    for functions that do not take RandomState instances or as a workaround 
    when one wants to use the same RNG for several parallel processes.

    Parameters
    ----------
    rng : np.random.RandomState
        RandomState instance

    Returns
    -------
    int
        an int within 0 and 2**32-1 (the maximum value for seeds in numpy)
    """
    return rng.randint(low=0, high=2**32 - 1)

def grid_matrix(*xi) -> np.ndarray:
    """
    Computes the cartesian product of all passed vectors.

    Returns
    -------
    np.ndarray : shape [product of input-lengths, number of inputs]
        ...
    """
    repeats = np.meshgrid(*xi)
    columns = [ri.flatten()[:,None] for ri in repeats]
    return np.hstack(columns)

def grid_like(X, grid_points: int=10000, pad: float=0.0) -> np.ndarray:
    """
    Generates a grid over the space defined by X.

    Parameters
    ----------
    X : array-like : shape (n_samples, n_features)
        Input data from which min and max values of each dimension will be 
        inferred. 
    grid_points : int, optional
        Maximum number of samples that will be returned. The number of grid-
        ticks in each dimension is the floor-value of the d-th root of 
        `grid_points`, where d is the number of dimensions (i.e. columns of X). 
        By default 10000.
    pad : float
        Relative padding around the dimension-wise ranges. The grid over some 
        dimension x goes from min(x)-pad*r to max(x)+pad*r, where r is 
        max(x)-min(x).

    Returns
    -------
    np.ndarray : shape (<=grid_points, X.shape[1])
        Grid data spanned over input data X. 
    """
    nrows, ndims    = X.shape
    dims_with_range = []
    for j in range(ndims):
        xj = X[:,j]
        if len(np.unique(xj)) > 1:
            dims_with_range.append(j)

    points_per_dim = int(grid_points**(1/len(dims_with_range)))
    
    # compute the 1d-grid for each dimension
    dim_lvl_grids  = [] # will store the individual linspaces for all dimensions
    for j in range(ndims):
        xj = X[:,j]
        if j in dims_with_range:
            xj_range   = xj.max() - xj.min()
            grid_start = xj.min() - pad*xj_range
            grid_stop  = xj.max() + pad*xj_range
            grid_j     = np.linspace(grid_start, grid_stop, points_per_dim)
        else:
            grid_j = np.unique(xj)
        dim_lvl_grids.append(grid_j)
    
    # compute the cartesian product of these 1d-grids
    X_grid = grid_matrix(*dim_lvl_grids)

    return X_grid

def drop_nan(*xi):
    """
    Drops all elements from all input arrays, for which any of the arrays has a
    `nan` at hte respective position, i.e. if three arrays x0, x1, and x2 are 
    passed and x2[3] is `nan`, then all 4th elements of x0, x1, and x2 are 
    dropped. Arrays must be of same shape. 

    Returns
    -------
    *xi : tuple of array-likes
        Arrays as passed but processed as described above.
    """
    nan_mask = np.zeros_like(xi[0], dtype=bool)
    for a in xi:
        nan_mask |= np.isnan(a)
    return tuple(a[~nan_mask] for a in xi)

def true_ratio(x: Iterable[bool]) -> float:
    x = (np.atleast_1d(x)).flatten()
    x = x > 0
    return np.sum(x)/len(x)

def clamp(x, low=0, high=1):
    x = np.atleast_1d(x)
    x = x.copy()
    x[x < low]  = low
    x[x > high] = high
    return x

def clip(x, low=0, high=1):
    x = np.atleast_1d(x)
    x = x.copy()
    x = x[x > low]
    x = x[x < high]
    return x

def sigmoid(x):
    return 1/(1+np.exp(-x))

def inverse_sigmoid(x):
    x = np.atleast_1d(x)
    if any((x <= 0) | (x >= 1)):
        raise ValueError("x must be within (0,1).")
    return -np.log(1/x-1)

def non_decreasing(x):
    """
    Returns True if every x[i+1] >= x[i], otherwise False.
    """
    return np.all(np.diff(x) >= 0)

def expand_range(low: float, high: float, abs: Optional[float]=None, 
                 rel: Optional[float]=None) -> Tuple[float,float]:
    if abs is not None and rel is not None:
        raise ValueError("Only one arg of abs and rel must be passed")
    if abs is None:
        span = np.abs(high-low)
        abs  = span*rel
    return low-abs, high+abs

def stretch(x, newmin=None, newmax=None):
    """
    Stretches data onto a new range
    """
    if (newmin is None) and (newmax is None):
        raise ValueError("Pass at least one of newmin or newmax")
    if newmin is None:
        newmin = x.min()
    if newmax is None:
        newmax = x.max()
    L = newmax - newmin
    x = x - x.min()
    x = (x/x.max()) * L + newmin
    return x

def inter_quartile_range(x, return_quartiles: bool=False
                        ) -> Union[float, Tuple[float,float,float]]:
    """
    Compute the InterQuartileRange of an array-like.

    Args:
        x (array-like): Data of shape (n,)
        return_quartiles (bool): See Returns

    Returns:
        If return_quartiles is True:
            iqr (float): inter quartile range
            q1 (float): 25th percentile or 1st quartile
            q3 (float): 75th percentile or 3rd quartile
        If return_quartiles is False:
            iqr (float): inter quartile range
    """
    q1, q3 = np.percentile(x, [25,75])
    iqr    = q3-q1
    if not return_quartiles:
        return iqr
    else:
        return iqr, q1, q3

def boxplot_indicators(x):
    """
    Args:
        x (array-like): input array
    
    Returns:
        lower_whisker, first_quartile, median, third_quartile, upper_whisker
        
    Details:
        lower_whisker: first_quartile - 1.5 * IQR
        upper_whisker: first_quartile + 1.5 * IQR
    """
    first_quartile, median, third_quartile = np.percentile(x, [25,50,75])
    iqr = third_quartile-first_quartile
    lower_whisker = first_quartile - 1.5 * iqr
    upper_whisker = first_quartile + 1.5 * iqr
    return lower_whisker, first_quartile, median, third_quartile, upper_whisker

def all_unique(x) -> bool:
    return np.unique(x).shape == x.shape

def is_constant(x, close: bool=True) -> bool:
    if close:
        return np.isclose(x.min(), x.max())
    else:
        return x.min() == x.max()

def hstack_flat_arrays(*xi) -> np.ndarray:
    columns = [a[:,None] for a in xi]
    return np.hstack(columns)

def columns(A: np.ndarray) -> Tuple:
    cols = [A[:,i] for i in range(A.shape[1])]
    return tuple(cols)

def rolling_mean(x, window, **kwargs) -> np.ndarray:
    return pd.Series(x).rolling(window=window, **kwargs).mean().values

def rolling_std(x, window, **kwargs) -> np.ndarray:
    return pd.Series(x).rolling(window=window, **kwargs).std().values

def proba_to_odds(p: float) -> float:
    """
    Converts a probability (Bernoulli event) to odds.
    """
    return p/(1-p)

def odds_to_proba(o: float) -> float:
    """
    Converts odds (n_event_a / n_event_not_a) to probability (p(a)=1-p(not a)). 
    """
    return o/(o+1)


class ProgressDisplay():
    """
    Prints a progress bar.
    
    How to:
    1) set up progress display 
        progr = ProgressDisplay(ntotal)
    2) start timer
        progr.start_timer()
    3) after each step (with ntotal steps)
        progr.update_and_print()
    4) when finished
        progr.stop()
        
    Developed during ML Lab Course.
    """
    def __init__(self, ntotal, nprocessed = 0, unit="m", eol = "\r",
                 disable=False, prefix: str=""):
        self.ntotal = ntotal
        self.nprocessed = nprocessed
        self.eol = eol
        self.nleft = ntotal
        self.unit = unit
        self.len_of_last_line = 1
        self.disable = disable
        self.prefix = prefix
        self.done = False
        
    def start_timer(self) -> "ProgressDisplay":
        self.tstart = time.time()
        return self
    
    def update(self, nnew) -> "ProgressDisplay":
        """
        nnew: number of tasks processed since last update
        """
        self.nprocessed += nnew
        self.nleft -= nnew
        return self
    
    def compute_estimated_time_left(self):
        time_passed = time.time() - self.tstart
        time_left   = time_passed * self.nleft / self.nprocessed
        
        return self.calc_time_in_unit(time_left)
    
    def calc_time_in_unit(self, t):
        if self.unit == "m":
            t = t / 60
        elif self.unit == "h":
            t = t / 3600
        elif self.unit == "s":
            t = t
        else:
            raise ValueError("t must be one of 'm', 'h', 's'")
        return t
        
    def print_status(self, note: Optional[str]=None) -> "ProgressDisplay":
        if not self.disable:
            perc = 100 * self.nprocessed/self.ntotal
            progress_display = f"{int(perc)}% ({self.nprocessed}/{self.ntotal})"
            if self.nprocessed == 0:
                time_left_display = "?"
                time_passed_display = "?"
                time_per_iter = "?"
            else:
                time_left = self.compute_estimated_time_left()
                time_left_display = f"{time_left:.02f} {self.unit}"
                time_passed = time.time()-self.tstart
                time_per_iter = self.nprocessed / time_passed
                time_per_iter = f"{time_per_iter:.02f}"
                time_passed = self.calc_time_in_unit(time_passed)
                time_passed_display = f"{time_passed:.02f} {self.unit}"
            
            print("\r", " "*self.len_of_last_line, end="\r") # flush line #todo escape
            display = f"\r{self.prefix}{progress_display}, " \
                      f"{time_passed_display} passed, " \
                      f"{time_left_display} left, " \
                      f"{time_per_iter} it/s"
                    # f"est. time left: {time_left_display}"
            if note is not None:
                display += " " + note
            print("\r", display, end="\r")
            self.len_of_last_line = len(display)
        return self
        
    def update_and_print(self, nnew=1, note: Optional[str]=None
                        ) -> "ProgressDisplay":
        self.update(nnew)
        self.print_status(note = note)
        return self
    
    def stop(self) -> "ProgressDisplay":
        total_time = self.calc_time_in_unit( time.time()-self.tstart )
        self.done  = True
        self.print_status(note="  DONE")
        # print(f"\n   total time: {total_time:02f} {self.unit}\n")
        return self