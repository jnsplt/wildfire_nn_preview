import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import List, Tuple, Optional, Union, Any, Dict, Iterable, Callable
import warnings
from contextlib import contextmanager

import scipy.stats as scs
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as mcm

import sklearn.metrics as skm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch
from torch.optim.lr_scheduler import _LRScheduler # base class, for type hinting

import fire.utils.etc as uetc

import pdb


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    oob = (x < 0) | (x > 1) # out of bounds
    if any(oob):
        raise ValueError(f"x must be within [0,1]: {x[oob]}")
    return -torch.log(1/x-1)

def count_by_class(y: Iterable[Any], rel: bool=False
                  ) -> Dict[Any, Union[float,int]]:
    """
    Counts the number of samples by class.

    Parameters
    ----------
    y : Iterable[Any], e.g. array-like of shape (n_samples,)
        Class labels for each sample.
    rel : bool, optional
        If True, counts will be returned as ratios (summing to 1), by default 
        False

    Returns
    -------
    Dict[Any, Union[float,int]]
        Dictionary mapping from class label to count (or ratio, depending on the
        parameter `rel`).
    """
    counts = pd.Series(data=y, name="y").to_frame().groupby("y").size()
    if rel:
        counts /= counts.sum()
    return counts.to_dict()

def inverse_priors(p: np.ndarray, eps=None) -> np.ndarray:
    """
    Args:
        p (array-like): class-wise prior probabilities
        eps (float): Is added to p right at the beginning, e.g.
            to avoid zero-division
    
    Returns:
        w (float): inverse priors normed such that sum(w) = len(p)
    """
    p = np.array(p)
    if eps is not None:
        p = p + eps # to avoid zero div
    p = p/p.sum() # such that sum(p) = 1
    w = len(p) / p / np.sum(1/p)
    return w

def make_class_weights_from_biased_sample(y_biased: np.ndarray, 
                                          r_pos_original: float
                                         ) -> Dict[int, float]:
    """
    weight assigned to pos. class (`pos_weight`) actually is the keep-neg-rate
    """
    n_neg = (~y_biased.astype(bool)).sum()
    n_pos = y_biased.sum()
    
    # calculate the weight to assign to positive samples
    pos_weight = r_pos_original*n_neg / (n_pos*(1-r_pos_original))
    class_weights = {1: pos_weight, 0: 1.0}
    return class_weights

def stratified_idx(y, oversample=True, seed=None) -> np.ndarray:
    """
    Args:
        y (array like): class vector to stratify
        oversample (bool): If True, all examples from the majority class and 
            all exmaples from the minority class are taken. Additionally, 
            examples from the minority class are sampled (with replacement) 
            until the class sizes are equal. If False, all examples of the 
            minority class are taken and the majority class is undersampled 
            (without replacement).
    """
    examples = pd.DataFrame({"y": y}) # also has an index from 0 to n
    class_sizes = examples.groupby("y").size().rename("n").to_frame()
    class_sizes.loc[:, "is_maj"] = class_sizes["n"] == class_sizes["n"].max()
    class_sizes.loc[:, "is_min"] = class_sizes["n"] == class_sizes["n"].min()
    
    maj_size = class_sizes.query("is_maj")["n"].values[0]
    min_size = class_sizes.query("is_min")["n"].values[0]
    
    stratified_selected_example_idx = []
    rng = np.random.RandomState(seed)
    
    for class_name, class_meta in class_sizes.iterrows():
        ii = examples.loc[examples["y"] == class_name].index.to_list()
        if oversample:
            if not class_meta["is_maj"]:
                n_to_sample = maj_size - class_meta["n"]
                ii += list(rng.choice(ii, size=n_to_sample, replace=True))
        else: # undersample
            if not class_meta["is_min"]:
                ii = list(rng.choice(ii, size=min_size, replace=False))
        stratified_selected_example_idx += ii
    
    return stratified_selected_example_idx


#todo del
def compute_performance_scores(y_true, y_pred, y_decf, y_proba=None):
    warnings.warn("Function name is deprecated. Use "
                  "compute_classification_scores instead", DeprecationWarning)
    return compute_classification_scores(y_true, y_pred, y_decf, y_proba)


def compute_classification_scores(y_true, y_pred, y_decf, y_proba=None, 
                                  suppress_warnings: bool=True, 
                                  hard_scores: bool=True, 
                                  soft_scores: bool=True, 
                                  stats: bool=True, knr=None, kwcalib={}):
    """
    Args: 
        knr (float or None): passed to calibration error metrics
    """
    n_classes = len(np.unique(y_true))
    with warnings.catch_warnings():
        if suppress_warnings:
            warnings.simplefilter("ignore")
        score_dict = {}
        if hard_scores:
            score_dict = {**score_dict, **{
                "acc":   skm.accuracy_score(         y_true, y_pred),
                "accba": skm.balanced_accuracy_score(y_true, y_pred, 
                                                     adjusted=True),
                "zol":   skm.zero_one_loss(          y_true, y_pred),
                "mcc":   skm.matthews_corrcoef(      y_true, y_pred),
                "kpa":   skm.cohen_kappa_score(      y_true, y_pred),
                "f1" :   skm.f1_score(               y_true, y_pred),
                "rcl":   skm.recall_score(           y_true, y_pred),
                "prc":   skm.precision_score(        y_true, y_pred),
                "tp" : np.sum((y_true == 1) & (y_pred == 1)), # true positives
                "tn" : np.sum((y_true == 0) & (y_pred == 0)), # true negatives
                "fp" : np.sum((y_true == 0) & (y_pred == 1)), # false positives
                "fn" : np.sum((y_true == 1) & (y_pred == 0)), # false negatives
            }}
        if soft_scores:
            compute_proba_scores = y_proba is not None and n_classes > 1
            if compute_proba_scores:
                p0, p90, p95, p100 = np.percentile(y_proba, [0,90,95,100])
            score_dict = {**score_dict, **{
                "auroc":  skm.roc_auc_score(y_true, y_decf) \
                          if n_classes > 1 else None,
                "avp":    skm.average_precision_score(y_true, y_decf) \
                          if n_classes > 1 else None,
                "logl":   skm.log_loss(y_true, y_proba) \
                          if compute_proba_scores else None,
                "bssl":   skm.brier_score_loss(y_true, y_proba) \
                          if compute_proba_scores else None,

                # calibration error metrics with quantile bins
                "eace_q": expected_calibration_error(y_true, y_proba, knr=knr, 
                                                     relative=False, mode="q",
                                                     weighted=True, **kwcalib) \
                          if compute_proba_scores else None,
                "erce_q": expected_calibration_error(y_true, y_proba, knr=knr, 
                                                     relative=True, mode="q",
                                                     weighted=True, **kwcalib) \
                          if compute_proba_scores else None,
                "aace_q": expected_calibration_error(y_true, y_proba, knr=knr, 
                                                     relative=False, mode="q",
                                                     weighted=False, **kwcalib) \
                          if compute_proba_scores else None,
                "arce_q": expected_calibration_error(y_true, y_proba, knr=knr, 
                                                     relative=True, mode="q",
                                                     weighted=False, **kwcalib) \
                          if compute_proba_scores else None,
                "mace_q": maximum_calibration_error(y_true, y_proba, knr=knr, 
                                                    relative=False, mode="q", 
                                                    **kwcalib) \
                          if compute_proba_scores else None,
                "mrce_q": maximum_calibration_error(y_true, y_proba, knr=knr, 
                                                    relative=True, mode="q", 
                                                    **kwcalib) \
                          if compute_proba_scores else None,

                # calibration error metrics with uniform bins, 
                # clipped at 90th percentile
                "eace_u90": expected_calibration_error(y_true, y_proba, knr=knr, 
                                                       relative=False, mode="u",
                                                       bin_range=(0,p90),
                                                       weighted=True, **kwcalib) \
                            if compute_proba_scores else None,
                "erce_u90": expected_calibration_error(y_true, y_proba, knr=knr, 
                                                       relative=True, mode="u",
                                                       bin_range=(0,p90),
                                                       weighted=True, **kwcalib) \
                            if compute_proba_scores else None,
                "aace_u90": expected_calibration_error(y_true, y_proba, knr=knr, 
                                                       relative=False, mode="u",
                                                       bin_range=(0,p90),
                                                       weighted=False, **kwcalib) \
                            if compute_proba_scores else None,
                "arce_u90": expected_calibration_error(y_true, y_proba, knr=knr, 
                                                       relative=True, mode="u",
                                                       bin_range=(0,p90),
                                                       weighted=False, **kwcalib) \
                            if compute_proba_scores else None,
                "mace_u90": maximum_calibration_error(y_true, y_proba, knr=knr, 
                                                      relative=False, mode="u", 
                                                      bin_range=(0,p90),
                                                      **kwcalib) \
                            if compute_proba_scores else None,
                "mrce_u90": maximum_calibration_error(y_true, y_proba, knr=knr, 
                                                      relative=True, mode="u", 
                                                      bin_range=(0,p90),
                                                      **kwcalib) \
                            if compute_proba_scores else None,

                # calibration error metrics with uniform bins, 
                # clipped at 90th percentile
                "eace_u95": expected_calibration_error(y_true, y_proba, knr=knr, 
                                                       relative=False, mode="u",
                                                       bin_range=(0,p95),
                                                       weighted=True, **kwcalib) \
                            if compute_proba_scores else None,
                "erce_u95": expected_calibration_error(y_true, y_proba, knr=knr, 
                                                       relative=True, mode="u",
                                                       bin_range=(0,p95),
                                                       weighted=True, **kwcalib) \
                            if compute_proba_scores else None,
                "aace_u95": expected_calibration_error(y_true, y_proba, knr=knr, 
                                                       relative=False, mode="u",
                                                       bin_range=(0,p95),
                                                       weighted=False, **kwcalib) \
                            if compute_proba_scores else None,
                "arce_u95": expected_calibration_error(y_true, y_proba, knr=knr, 
                                                       relative=True, mode="u",
                                                       bin_range=(0,p95),
                                                       weighted=False, **kwcalib) \
                            if compute_proba_scores else None,
                "mace_u95": maximum_calibration_error(y_true, y_proba, knr=knr, 
                                                      relative=False, mode="u", 
                                                      bin_range=(0,p95),
                                                      **kwcalib) \
                            if compute_proba_scores else None,
                "mrce_u95": maximum_calibration_error(y_true, y_proba, knr=knr, 
                                                      relative=True, mode="u", 
                                                      bin_range=(0,p95),
                                                      **kwcalib) \
                            if compute_proba_scores else None,

                # calibration error metrics with uniform bins, 
                # clipped at two-sided at min and max
                "eace_umm": expected_calibration_error(y_true, y_proba, knr=knr, 
                                                       relative=False, mode="u",
                                                       bin_range=(p0,p100),
                                                       weighted=True, **kwcalib) \
                            if compute_proba_scores else None,
                "erce_umm": expected_calibration_error(y_true, y_proba, knr=knr, 
                                                       relative=True, mode="u",
                                                       bin_range=(p0,p100),
                                                       weighted=True, **kwcalib) \
                            if compute_proba_scores else None,
                "aace_umm": expected_calibration_error(y_true, y_proba, knr=knr, 
                                                       relative=False, mode="u",
                                                       bin_range=(p0,p100),
                                                       weighted=False, **kwcalib) \
                            if compute_proba_scores else None,
                "arce_umm": expected_calibration_error(y_true, y_proba, knr=knr, 
                                                       relative=True, mode="u",
                                                       bin_range=(p0,p100),
                                                       weighted=False, **kwcalib) \
                            if compute_proba_scores else None,
                "mace_umm": maximum_calibration_error(y_true, y_proba, knr=knr, 
                                                      relative=False, mode="u", 
                                                      bin_range=(p0,p100),
                                                      **kwcalib) \
                            if compute_proba_scores else None,
                "mrce_umm": maximum_calibration_error(y_true, y_proba, knr=knr, 
                                                      relative=True, mode="u", 
                                                      bin_range=(p0,p100),
                                                      **kwcalib) \
                            if compute_proba_scores else None
            }}
        if stats:
            score_dict = {**score_dict, **{
                "n_total": len(y_true),
                "n_true" : np.sum(y_true),
                "r_true" : np.mean(y_true),
                "y_pred_mean":  np.mean(y_pred),
                "y_proba_min": None if y_proba is None else np.min(y_proba),
                "y_proba_max": None if y_proba is None else np.max(y_proba),
                "y_proba_mean": None if y_proba is None else np.mean(y_proba),
                "y_proba_std":  None if y_proba is None else np.std(y_proba)
        }}
    return score_dict

def compute_regression_scores(y_true, y_pred) -> Dict[str, float]:
    score_dict = {
        "mae":   skm.mean_absolute_error(y_true, y_pred),
        "mse":   skm.mean_squared_error(y_true, y_pred),
        "rmse":  skm.mean_squared_error(y_true, y_pred, squared=False),
        "msre":  mean_squared_relative_error(y_true, y_pred, squared=True),
        "rmsre": mean_squared_relative_error(y_true, y_pred, squared=False),
        "scor":  scs.spearmanr(y_true, y_pred)[0],
        "pcor":  scs.pearsonr(y_true, y_pred)[0],
        "maxae": maximum_absolute_error(y_true, y_pred),
        "bias":  bias(y_true, y_pred)
    }
    return score_dict

def bias(y_true, y_pred) -> float:
    """
    Returns the mean error, i.e. mean(y_pred - y_true).

    Details:
        https://en.wikipedia.org/wiki/Bias_of_an_estimator
    """
    return np.mean(y_pred - y_true)

def maximum_absolute_error(y_true, y_pred) -> float:
    abs_errors = np.abs(y_true - y_pred)
    return np.max(abs_errors)

def mean_squared_relative_error(y_true, y_pred, squared=True) -> float:
    error     = y_pred - y_true
    rel_error = error / y_true
    msre = np.mean(rel_error**2)
    return msre if squared else np.sqrt(msre)

def expected_calibration_error(y_true, y_proba, weighted=True, n_bins: int=10, 
                               bin_range=(0,1), mode="uniform", 
                               relative: bool=False, knr: Optional[float]=None
                              ) -> float:
    """
    Args:
        knr: If not None, passed as `keep_neg_rate` to `adjust_bin_means`.
        mode (str, optional): Passed to `reliability_bins`.
    """
    try:
        abs_errors, rel_errors, w = _calibration_error_basics(
            y_true=y_true, y_proba=y_proba, n_bins=n_bins, bin_range=bin_range,
            mode=mode, knr=knr
        )
        if weighted:
            weighted_errors = (w * rel_errors) if relative else (w * abs_errors)
            return weighted_errors.sum()
        else:
            return rel_errors.mean() if relative else abs_errors.mean()
    except Exception as e:
        warnings.warn(f"exception in expected_calibration_error: {e}, returning nan")
        return np.nan
    
def maximum_calibration_error(y_true, y_proba, n_bins: int=10, 
                              bin_range=(0,1), mode="uniform", 
                              relative: bool=False, knr: Optional[float]=None
                             ) -> float:
    try:
        abs_errors, rel_errors, _ = _calibration_error_basics(
            y_true=y_true, y_proba=y_proba, n_bins=n_bins, bin_range=bin_range,
            mode=mode, knr=knr
        )
        try:
            return rel_errors.max() if relative else abs_errors.max()
        except ValueError:
            # .mean() works on (e.g.) empty arrays (returns np.nan), .max() not
            return np.nan
    except Exception as e:
        warnings.warn(f"exception in maximum_calibration_error: {e}, returning nan")
        return np.nan

def _calibration_error_basics(y_true, y_proba, n_bins: int=10, 
                              bin_range=(0,1), mode="uniform", 
                              knr: Optional[float]=None
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        abs_errors, rel_errors, weights
    """
    rdf: pd.DataFrame = reliability_bins(
        y_true, y_proba, n_bins=n_bins, bin_range=bin_range, 
        mode=mode, keep_neg_rate=knr, as_frame=True)
    rdf.dropna(inplace=False)

    true_ratios = rdf["y_true_mean" if knr is None else "y_true_mean_adj"]
    abs_errors  = np.abs(true_ratios - rdf["y_proba_mean"])
    rel_errors  = abs_errors / true_ratios
    supports = rdf["support" if knr is None else "support_adj"]
    weights  = supports / supports.sum()
    return np.r_[abs_errors], np.r_[rel_errors], np.r_[weights]

def reliability_bins(y_true, y_proba, n_bins: int=10, bin_range=(0,1), 
                     mids_as_labels: bool=True, mode: str="uniform", 
                     precision: int=30, bin_edges: 
                     Optional[Union[Iterable[float], pd.IntervalIndex]]=None,
                     as_frame: bool=False, keep_neg_rate: Optional[float]=None
                    ) -> Union[Tuple, pd.DataFrame]:
    """
    [summary]

    Parameters
    ----------
    y_true : [type]
        [description]
    y_proba : [type]
        [description]
    n_bins : int, optional
        Ignored if bin_edges is not None, by default 10
    bin_range : tuple, optional
        If mode is 'uniform', absolute limits. If mode is 'quantile', limits
        in quantiles, e.g. `0.9` for 90th percentile. By default (0,1).
    mids_as_labels : bool, optional
        pd.Interval.mid of each bin (not the mean of any data!), by default True
    mode : str, optional : {"uniform","quantile"} or first characters of these
        Ignored if bin_edges is not None, by default "uniform"
    bin_edges : sequence of scalars or pandas.IntervalIndex, optional
        Passed to pandas.cut, if not None, by default None
    keep_neg_rate : float or None, optional
        ...
    as_frame : bool
        If True, return results as dataframe (contains more results)

    Returns
    -------
    tuple or dataframe
        if `as_frame = False`: 
            y_proba_bins, y_proba_means, y_true_means, supports
        if `as_frame = True`:
            dataframe with columns:
                y_proba_bin: pd.Interval or mid point (not mean!) of bin
                y_proba_mean: mean of estimates
                y_true_mean: mean of y_true
                y_true_mean_adj: y_true_mean adjusted using `keep_neg_rate`. If
                    this param is None, this column doesn't exist.
                support: number of samples in the corresponding bin.
                n_pos: number of positive samples in the bin.
                ratio_support: ratio of samples in this bin of all samples.
                ratio_support_adj: ratio of samples that would be in this bin
                    if dataset would not have been undersampled. Estimated 
                    using `keep_neg_rate`. If this param is None, this column
                    doesn't exist.
    """
    if bin_edges is None:
        lo, hi = bin_range
        if "uniform".startswith(mode):
            bin_edges = np.linspace(lo, hi, num=n_bins+1)
        elif "quantile".startswith(mode):
            bin_edges = np.percentile(y_proba, np.linspace(lo, hi*100, n_bins+1))
        else:
            raise ValueError("mode must be 'uniform' or 'quantile'")

    bin_df = (
        pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
        .assign(y_proba_binned = lambda df: 
                pd.cut(df["y_proba"], bins=bin_edges, precision=precision,
                       include_lowest=True, duplicates='drop')) # or 'raise'
        .groupby("y_proba_binned")
        .agg(["mean","count","sum"])
    )

    y_proba_bins: List[pd.Interval] = bin_df.index.values
    if mids_as_labels:
        y_proba_bins = np.array([itv.mid for itv in y_proba_bins])
    y_proba_means = bin_df[("y_proba","mean")].to_numpy()
    y_true_means  = bin_df[("y_true","mean")].to_numpy()
    n_samples      = bin_df[("y_true","count")].to_numpy()
    ratio_support = n_samples / n_samples.sum()

    if any(np.isnan(y_true_means)):
        warnings.warn("NaNs in reliability_bins y_true_means")
    
    if as_frame:
        n_pos = bin_df[("y_true","sum")].to_numpy()
        df = pd.DataFrame({
            "y_proba_bin": y_proba_bins, "y_proba_mean": y_proba_means, 
            "y_true_mean": y_true_means, "n_samples": n_samples, "n_pos": n_pos, 
            "support": ratio_support
        })
        if keep_neg_rate is not None:
            df.insert(3, "y_true_mean_adj", adjust_bin_means(
                y_true_means, keep_neg_rate=keep_neg_rate))
            n_samples_adj = n_pos + ((n_samples-n_pos) / keep_neg_rate)
            df.insert(7, "support_adj", n_samples_adj / n_samples_adj.sum())
        if df.isna().any(axis=None):
            warnings.warn("NaNs in reliability_bins dataframe")
        return df
    else:
        return y_proba_bins, y_proba_means, y_true_means, n_samples

def adjust_bin_means(bin_means: np.ndarray, 
                     r_pos_sampled: Optional[float]=None, 
                     r_pos_original: Optional[float]=None, 
                     keep_neg_rate: Optional[float]=None) -> np.ndarray:
    """
    Corrects the accuracies on reliability bins for an undersampled dataset. 

    Either `r_pos_sampled` and `r_pos_original`, OR `keep_neg_rate` must be 
    passed.

    Parameters
    ----------
    bin_means : np.ndarray
        As `y_true_means` returned by `reliability_bins`. Ratio of ground truth 
        positives in several bins
    r_pos_sampled : Optional[float], optional
        Ratio of ground truth positives sampled during under-/oversampling.
    r_pos_original : Optional[float], optional
        Ratio of ground truth positives that are actually in the population, 
        thus sampled when NOT under-/oversampling.
    keep_neg_rate : Optional[float], optional
        Ratio of negative samples that were kept in the course of undersampling. 

    Returns
    -------
    np.ndarray
        Corrected/adjusted bin means.
    """
    if (r_pos_original is not None) and (r_pos_sampled is not None):
        if keep_neg_rate is not None:
            raise ValueError("Either `r_pos_sampled` and `r_pos_original`, "
                             "OR `keep_neg_rate` must be passed.")
        keep_neg_rate = calculate_keep_neg_rate(r_pos_sampled, r_pos_original)
    
    if keep_neg_rate is None:
        raise ValueError("Either `r_pos_sampled` and `r_pos_original`, "
                         "OR `keep_neg_rate` must be passed.")
    bin_odds_sampled   = uetc.proba_to_odds(bin_means)
    bin_odds_adjusted  = bin_odds_sampled * keep_neg_rate
    bin_means_adjusted = uetc.odds_to_proba(bin_odds_adjusted)
    return bin_means_adjusted

def knr(r_pos_sampled: Optional[float]=None, 
        r_pos_original: Optional[float]=None) -> np.ndarray:
    """
    Shorthand for `calculate_keep_neg_rate`.
    """
    return calculate_keep_neg_rate(r_pos_sampled=r_pos_sampled, 
                                   r_pos_original=r_pos_original)

def calculate_keep_neg_rate(r_pos_sampled: Optional[float]=None, 
                            r_pos_original: Optional[float]=None) -> np.ndarray:
    odds_original = uetc.proba_to_odds(r_pos_original) # odds = n_pos/n_neg
    odds_sampled  = uetc.proba_to_odds(r_pos_sampled)
    return odds_original / odds_sampled

def reliability_plot_old(bins, proba_means, accs, supports=None, figsize=None, 
                         ax=None, zoom_x: bool=False, zoom_y: bool=False, 
                         bars: bool=True):
    """
    Creates a reliability plot from data generated (e.g.) by `reliability_bins()`

    Parameters
    ----------
    bins : array-like, shape (n,)
        Mids of the n bins. Expected to be spaced evenly.
    proba_means : array-like, schape (n,)
        Means of probas per bin.
    accs : array-like, shape (n,)
        Ratio of ground truth positives for each bin. 
    figsize : optional
        Passed to `pyplot.subplots`, by default None
    ax : optional
        Matplotlib axis to plot on, by default None
    zoom_x : bool, optional
        If True, x-axis limits are set so that the first and last bins shown
        are those for which accuracies are available, by default False
    zoom_y : bool, optional
        As in `zoom_x`.

    Returns
    -------
    [type]
        fig, ax
    """
    bin_width = np.diff(bins)[0]
    ax_passed = ax is not None
    if not ax_passed:
        fig, ax = plt.subplots(figsize=figsize)
    if bars:
        if supports is None:
            ax.bar(bins, accs, width=bin_width, align="center", edgecolor="k", 
                color="0.8")
        else:
            # color bars by support
            # with code from https://stackoverflow.com/a/51205723/2337838
            # fig, ax = plt.subplots(figsize=(15, 4))
            cmap   = plt.cm.get_cmap("BuGn")
            colors = cmap(np.array(supports)/max(supports))
            bars   = ax.bar(bins, accs, width=bin_width, align="center", 
                            edgecolor="k", color=colors)
            sm = mcm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(min(supports), max(supports)))
            # sm.set_array([])
            cbar = plt.colorbar(sm) # , pad=.05
            cbar.set_label("number of samples per bin", rotation=270, labelpad=20)

    ax.plot(bins, proba_means, c="r", marker="x")

    if zoom_x:
        non_nan_accs = np.where(~np.isnan(accs))
        xlim_low  = bins[non_nan_accs[0][ 0]] - bin_width/2
        xlim_high = bins[non_nan_accs[0][-1]] + bin_width/2
    else:
        xlim_low, xlim_high = 0, 1
        ax.set_aspect("equal")
    if zoom_y:
        non_nan_accs = np.where(~np.isnan(accs))
        ylim_low  = accs[non_nan_accs[0][ 0]] - bin_width/2
        ylim_high = accs[non_nan_accs[0][-1]] + bin_width/2
    else:
        ylim_low, ylim_high = 0, 1
        ax.set_aspect("equal")
    xlim_low, xlim_high = uetc.expand_range(xlim_low, xlim_high, rel=.05)
    ylim_low, ylim_high = uetc.expand_range(ylim_low, ylim_high, rel=.05)
    ax.set_xlim(xlim_low, xlim_high)
    ax.set_ylim(ylim_low, ylim_high)

    ax.plot([0,1], [0,1.], "k--", linewidth=1, zorder=10 if bars else 0)
    ax.set_xlabel("estimated probability")
    ax.set_ylabel("mean of y")
    return fig if not ax_passed else None, ax

def reliability_plot(reliability_df: pd.DataFrame, plot_ratio_samples=True, 
                     log_reliability=False, log_support=False, 
                     yleft_adj=False, yright_adj=False, figsize=(5,5.11)):
    """
    Args:
        reliability_df (dataframe):
            As returned by `uml.reliability_bins(..., as_frame=True)`.
        plot_ratio_samples (bool, opt.):
            If True, a second, log-scaled y-axis on the righ will be added which
            shows the ratio of samples going to each bin.
        yleft_adj (bool, optional):
            If True, `y_true_mean_adj` from `reliability_df` is used as 
            y(left)-data; if False `y_true_mean`.
        yright_adj (bool, optional):
            If True, `ratio_support_adj` from `reliability_df` is used as 
            y(left)-data; if False `ratio_support`.
    Hints:
        figsize: 
            If ratio_samples is plotted (right y-axis), the diagonal might not 
            end exactly in the upper right corner, but slightly off. Play 
            around with figsize (e.g. slightly wider) to get this right.
            This is how the odd default `(5,5.11)` arose.

    Returns:
        ax1, ax2
        If `plot_ratio_samples` is None, ax2 will be None as well.
    """
    y_true_means_col  = "y_true_mean_adj" if yleft_adj else "y_true_mean"
    ratio_samples_col = "support_adj" if yright_adj else "support"
    
    _, ax1 = plt.subplots(figsize=figsize)
    p1 = ax1.plot(
        reliability_df["y_proba_mean"], reliability_df[y_true_means_col], 
        linestyle="--", marker="P", color="tab:red", markersize=8, zorder=10)
    
    if log_reliability: 
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    ax1.set_aspect("equal")
    ax1.set_xlabel("estimated fire probability")
    ax1.set_ylabel("observed fire ratio", 
                   color=p1[0].get_color() if plot_ratio_samples else "k")

    ax2 = None
    if plot_ratio_samples:
        ax2 = ax1.twinx()
        p2  = ax2.plot(
            reliability_df["y_proba_mean"], reliability_df[ratio_samples_col], 
            linestyle="--", marker="P", color="tab:blue", markersize=8)
        if log_support: 
            ax2.set_yscale("log")
        ylabel = "ratio of samples" 
        if yright_adj:
            ylabel += "\n(corrected for undersampling)"
        ax2.set_ylabel(ylabel, color=p2[0].get_color())

    # plot diagonal (where estimates should lie on) and set axis limits
    xmin, xmax, ymin, ymax = ax1.axis()
    xymin = min(xmin, ymin) if log_reliability else 0
    xymax = max(xmax, ymax)
    ax1.plot([xymin, xymax], [xymin, xymax], "k--", linewidth=1, zorder=0)
    ax1.set_xlim(xymin, xymax)
    ax1.set_ylim(xymin, xymax)

    return ax1, ax2


class KDEClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian generative classification based on KDE.
    
    Args:
        bandwidth (float or list of floats):
            The kernel bandwidth within each class. If a list of floats is 
            passed, it has to be of length n_classes (one bandwidth for each
            class).
        kernel (str):
            the kernel name, passed to KernelDensity
        max_samples_per_class (int (opt)):
            Maximum number of samples used for training for one class. This can
            speed up prediction a lot. Samples are drawn from X[y==y_i] randomly.
        rnd_seed (opt): 
            Passed to numpy RandomState if max_samples_per_class is not None, 
            otherwise ignored.
    
    References:
        copy-pasted from (but edited: added random sampling to reduce set sizes) 
        https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian', 
                 max_samples_per_class=None, rnd_seed=None):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.max_samples_per_class = max_samples_per_class
        self.rnd_seed = rnd_seed

        if self.max_samples_per_class is not None:
            self._rng      = np.random.RandomState(rnd_seed).randint
            self._max_seed = 2**32 - 1
        
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [self._sample_if_too_large(X[y == yi])
                         for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=bw,
                                      kernel=self.kernel).fit(Xi)
                        for Xi, bw 
                        in zip(training_sets, self._get_list_of_bandwidths())]

        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self
        
    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

    def _sample_if_too_large(self, X):
        n = X.shape[0]
        if self.max_samples_per_class is None or n <= self.max_samples_per_class:
            return X
        else:
            bool_sel = uetc.rnd_bool(
                n = n, n_true = self.max_samples_per_class, 
                seed = self._rng(self._max_seed))
            return X[bool_sel]

    def _get_list_of_bandwidths(self) -> List[float]:
        n_classes  = len(self.classes_)
        bandwidths = np.atleast_1d(self.bandwidth)
        if len(bandwidths) == 1:
            bandwidths = np.repeat(bandwidths, n_classes)
        return bandwidths


class FreqTableClassifier(BaseEstimator, ClassifierMixin):
    """
    ...
    
    Args:
        max_proba (float (opt)): If passed (0 <= max_proba <= 1), all 
            probabilities will be scaled such that the largest probability for 
            class "1" computed in the training will equal `max_proba`. 
        unk_proba (float): The probability p(y=1 | x) for any x that has not
            been seen during training.
    """
    def __init__(self, unk_proba: float=0.0, threshold: float=0.5, 
                 max_proba: Optional[float]=None):
        self.unk_proba = unk_proba
        self.threshold = threshold
        self.max_proba = max_proba
        
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        Xy_df, x_cols = self._make_df_from_X_and_y(X, y)
        freq_table = Xy_df.groupby(x_cols).mean().rename(columns={"y":"p"})
        if self.max_proba is not None:
            freq_table = freq_table / freq_table.max() * self.max_proba

        self.x_cols_ = x_cols
        self.freq_table_ = freq_table
        return self
        
    def predict_proba(self, X):
        X = self._make_df_from_X(X)
        X = X.merge(self.freq_table_, how="left", on=self.x_cols_) # with p col
        X.loc[X["p"].isna(), "p"] = self.unk_proba
        p1 = X["p"].to_numpy() # proba for class 1
        p0 = 1-p1
        return np.hstack([p0[:,None], p1[:,None]])
        
    def predict(self, X):
        p1 = self.predict_proba(X)[:,1]
        return (p1 >= self.threshold) * 1

    def _make_df_from_X_and_y(self, X, y) -> Tuple[pd.DataFrame, List[str]]:
        """
        Returns:
            Xy (pd.DataFrame): DataFrame with all but last columns X, last 
                column y.
            x_cols (list of str): List of x-column names, i.e. 
            ["x0", ..., "xn"].
        """
        Xy = np.hstack([X, y[:,None]])
        n_cols   = X.shape[1]
        x_cols   = [f"x{i}" for i in range(n_cols)]
        all_cols = x_cols + ["y"]
        Xy = pd.DataFrame(Xy, columns=all_cols)
        return Xy, x_cols

    def _make_df_from_X(self, X) -> pd.DataFrame:
        return pd.DataFrame(X, columns=self.x_cols_)


class RidgeRegressionClassifier(BaseEstimator, ClassifierMixin):
    """
    ...
    
    Args:
        ...
    """
    def __init__(self, alpha=1.0, threshold=0.5, max_proba=None):
        self.alpha = alpha
        self.threshold = threshold
        self.max_proba = max_proba
        
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        assert self.classes_[0] == 0 and self.classes_[1] == 1, \
            "classes must be 0 and 1"
        self.ridge_ = Ridge(alpha = self.alpha).fit(X, y)
        self.max_p_ = self.ridge_.predict(X).max()
        return self
        
    def predict_proba(self, X):
        p1 = self.ridge_.predict(X)
        if self.max_proba is not None:
            p1 = p1 / self.max_p_ * self.max_proba
        p1[p1 > 1] = 1
        p1[p1 < 0] = 0
        p0 = 1-p1
        return np.hstack([p0[:,None], p1[:,None]])
        
    def predict(self, X):
        p1 = self.predict_proba(X)[:,1]
        return (p1 >= self.threshold) * 1


#todo if with val set, take params of best val loss
class SklearnTorchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, net: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 criterion=nn.BCELoss, 
                 epochs: int=100, batch_size: int=50, 
                 scheduler: Optional[_LRScheduler]=None, 
                 stopping_criteria: List["StoppingCriterion"]=[], 
                 postprocessor: Optional["PostProcessor"]=None,
                 train_size: Optional[Union[float,int]]=None, 
                 val_size: Optional[Union[float,int]]=None,
                 is_clf: bool=True, rnd_state_split=None, verbose: bool=True,
                 print_freq=10, show_batch_progr=False, 
                 score_every_n_batches: Optional[int]=None,
                 score_before: bool=True, double_precision: bool=True, 
                 keep_dataframe: bool=True,
                 output_getter: Optional[Callable]=None,
                 criterion_kwargs={}):
        """
        Args:
            keep_dataframe (bool):
                If True, X of type pandas.DataFrame is allowed and batches of X
                will be passed to the net as DataFrame as well.
            output_getter (callable, optional):
                function with signature `f(net: nn.Module, X: Tensor) -> Tensor`
                from which the output will be passed to an instance made from 
                `criterion`. If None, output will be obtained by 
                `self.net(X).squeeze()`.

            ... #todo
        """
        self.net        = net
        self.epochs     = epochs
        self.verbose    = verbose
        self.print_freq = print_freq
        self.batch_size = batch_size
        self.show_batch_progr      = show_batch_progr
        self.score_every_n_batches = score_every_n_batches
        self.score_before = score_before
        self.double_precision = double_precision
        self.keep_dataframe = keep_dataframe
        self.output_getter  = output_getter
        self.criterion_kwargs = criterion_kwargs

        if double_precision:
            self.net.double()
        else:
            self.net.float()

        self.criterion = criterion
        self.optimizer = optimizer
        self.stopping_criteria = stopping_criteria
        self.scheduler     = scheduler
        self.postprocessor = postprocessor

        self._do_split  = train_size is not None or val_size is not None
        self.train_size = train_size
        self.val_size   = val_size
        self.rnd_state_split = rnd_state_split

        self.is_clf = is_clf

        self.net.eval()
        
    def fit(self, X, y, class_weights: Optional[Dict[int,float]]=None):
        """
        Fits the ANN.

        Parameters
        ----------
        X : numpy.ndarray
            [description]
        y : [type]
            [description]
        class_weights : Optional[Dict[int,float]], optional
            Dictionary that maps values of y to their respective sample weights, 
            by default None. If not None, all samples for which y is not 
            in this dict will be weighted 1.

        Returns
        -------
        self
        """
        # split train/val if wanted by user
        if self._do_split:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, train_size=self.train_size, test_size=self.val_size, 
                shuffle=True, random_state=self.rnd_state_split, 
                stratify = y if self.is_clf else None)

            # prepare data
            val_dl, X_df_val = self._make_dataloader(X_val, y_val)
        else:
            X_train, y_train = X, y
            val_dl, X_df_val = None, None
        
        # prepare data
        train_dl, X_df_train = self._make_dataloader(X_train, y_train)

        self.fit_from_dataloader(train_dl, val_dl, X_df_train=X_df_train, 
                                 X_df_val=X_df_val, class_weights=class_weights)
        return self

    def _make_dataloader(self, X, y):
        X, y = self._to_tensors(X, y) # might return X as dataframe, see 
                                      # self.keep_dataframe
        if isinstance(X, pd.DataFrame):
            X_df = X
            idx  = np.arange(len(X_df))
            ds   = Dataset(idx, y)
        else:
            ds   = Dataset(X, y)
            X_df = None

        dl = torch.utils.data.DataLoader(ds, batch_size = self.batch_size)
        return dl, X_df
    
    def fit_from_dataloader(self, train_dl, val_dl=None, 
                            X_df_train: Optional[pd.DataFrame]=None, 
                            X_df_val: Optional[pd.DataFrame]=None, 
                            class_weights: Optional[Dict[int,float]]=None):
        """
        [summary]

        Parameters
        ----------
        train_dl : [type]
            [description]
        val_dl : [type], optional
            [description], by default None
        X_df_train : Optional[pd.DataFrame], optional
            If not None, it's assumed that the dataloaders merely output indices 
            for X and not batches of X, by default None
        X_df_val : Optional[pd.DataFrame], optional
            See X_df_train. Required if val_dl is not None, by default None
        class_weights : Optional[Dict[int,float]], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        self.net.train()

        if not hasattr(self, "train_loss_history_"):
            self.train_loss_history_ = []
        if not hasattr(self, "val_loss_history_"):
            self.val_loss_history_ = []
        if not hasattr(self, "score_steps_"):
            self.score_steps_ = []

        if self.postprocessor is not None:
            self.postprocessor.register(self, train_dl, val_dl)

        if X_df_train is None: # dataloaders contain X itself and not indices
            X_train_complete = train_dl.dataset.X
            X_val_complete = val_dl.dataset.X if val_dl is not None else None
        else: # dataloaders contain X itself and not indices
            X_train_complete = X_df_train
            X_val_complete = X_df_val if val_dl is not None else None
        y_train_complete = train_dl.dataset.y
        y_val_complete = val_dl.dataset.y if val_dl is not None else None

        if self.score_before:
            self._compute_and_store_losses(X_train_complete, y_train_complete, 
                                           X_val_complete, y_val_complete, 
                                           -1, class_weights=class_weights)
            if self.verbose:
                self._print_training_progress(-1)
        
        for epoch in range(self.epochs):
            for i, batch in enumerate(tqdm(train_dl, desc=f"ep {epoch} training", 
                                           disable=not self.show_batch_progr)):
                X, y = batch[0], batch[1]
                if X_df_train is not None: # then X is an array of indices
                    X = X_df_train.iloc[X]
                sample_weights = self._make_sample_weights(y, class_weights)
                self.optimizer.zero_grad()

                # Forward pass
                if self.output_getter is None:
                    output = self.net(X).squeeze()
                else:
                    output = self.output_getter(self.net, X)

                # Compute loss (only for current batch and step!)
                if sample_weights is not None:
                    criterion_instance = self.criterion(weight=sample_weights, 
                                                        **self.criterion_kwargs)
                else:
                    criterion_instance = self.criterion(**self.criterion_kwargs)
                loss = criterion_instance(output, y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()

                # score in the middle of epoch
                if self.score_every_n_batches is not None:
                    if i % self.score_every_n_batches == 0:
                        # compute where we are within epochs.
                        # Must be smaller than `epoch` to make sense later
                        # in the score history
                        epoch_ratio = self.score_every_n_batches / len(train_dl)
                        score_step  = epoch - 1 + epoch_ratio 

                        self._compute_and_store_losses(
                            X_train_complete, y_train_complete, 
                            X_val_complete, y_val_complete, 
                            score_step, class_weights=class_weights)

                if self.postprocessor is not None:
                    self.postprocessor.step(batch, output, loss)
                
            # Compute losses
            train_loss, val_loss = self._compute_and_store_losses(
                X_train_complete, y_train_complete, 
                X_val_complete, y_val_complete, 
                epoch, class_weights=class_weights)

            # Decay learning rate
            if self.scheduler is not None:
                self.scheduler.step(train_loss if val_loss is None else val_loss)
            self._log_lr()

            # check if training can be stopped early
            if self._any_stopping_criterion_fulfilled(train_loss, val_loss):
                break

            # 
            if self.postprocessor is not None:
                self.postprocessor.epoch()
            
            # 
            if self.verbose:
                self._print_training_progress(epoch)
        
        # end of training
        if self.verbose:
            print("-"*20)
            print(f"Stopped training after {epoch+1} epochs with final...")
            _val_loss = "None" if val_loss is None else f"{val_loss:.5f}"
            print(f"train loss: {train_loss:.5f},   val loss: {_val_loss}")

        self.net.eval()

        if self.postprocessor is not None:
            self.postprocessor.final()

        if self.verbose:
            print()        
        self.classes_ = np.sort(np.unique(y))
        return self

    def _make_sample_weights(self, y, class_weights=None):
        if class_weights is None:
            return None
        
        sample_weights = torch.ones(
            size=(len(y),), dtype=torch.double if self.double_precision else torch.float)
        for label, weight in class_weights.items():
            sample_weights[y == label] = weight

        return (sample_weights) # decides whether double or float

    def _any_stopping_criterion_fulfilled(self, train_loss, val_loss):
        for i, sc in enumerate(self.stopping_criteria):
            sc.step(train_loss, val_loss)
            if sc.fulfilled:
                if self.verbose:
                    print(f"Stopping criterion [{i}] fulfilled: {repr(sc)}")
                return True
        return False

    def _log_lr(self) -> None:
        """
        Appends the current learning rate(s) to the list `self.lr_history_`. If 
        list doesn't exist, it is created.
        """
        if not hasattr(self, "lr_history_"):
            self.lr_history_ = []
        current_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        self.lr_history_.append(tuple(current_lrs))

    def _raw_numpy_output_from_net(self, X) -> np.ndarray:
        check_is_fitted(self)
        X = self._to_tensors(X)
        return self.net(X).detach().squeeze().numpy()
    
    def decision_function(self, X):
        if not self.is_clf:
            warnings.warn("decision_function called on regression model")
        return self._raw_numpy_output_from_net(X)
    
    def predict_proba(self, X):
        if not self.is_clf:
            warnings.warn("predict_proba called on regression model")
        p1 = self.decision_function(X)[:,None]
        p1 = uetc.clamp(p1, low=0, high=1)
        return np.hstack([1-p1, p1])
    
    def predict(self, X):
        if self.is_clf:
            p1 = self.decision_function(X)
            return p1 > .5
        else:
            return self._raw_numpy_output_from_net(X)
    
    def _to_tensors(self, X, y=None):
        if not (isinstance(X, pd.DataFrame) and self.keep_dataframe):
            X = torch.DoubleTensor(X) if self.double_precision else torch.Tensor(X)
        if y is not None:
            y = torch.DoubleTensor(y) if self.double_precision else torch.Tensor(y)
            return X, y
        return X
    
    def _print_training_progress(self, epoch: int):
        if self.epochs < self.print_freq:
            long_time_since_last_print = True # for very small n of epochs
        else:
            long_time_since_last_print = epoch % self.print_freq == 0
            
        if (epoch == 0 or long_time_since_last_print):
            msg = (f"Epoch {epoch+1:5d}:   "
                   f"train loss: {self.train_loss_history_[-1]:.5f}")
            if self._do_split:
                msg += f",   val loss: {self.val_loss_history_[-1]:.5f}"
            print(msg, flush=True)

    def _compute_and_store_losses(self, X_train, y_train, X_val, y_val, epoch, 
                                  class_weights=None) -> Tuple[float,float]:
        # Compute loss on entire training set
        train_loss = self.score(X_train, y_train, class_weights=class_weights)
        self.train_loss_history_.append(train_loss)

        # Compute loss on entire validation set
        val_loss = self.score(X_val, y_val, class_weights=class_weights) \
                   if X_val is not None else None
        self.val_loss_history_.append(val_loss)

        self.score_steps_.append(epoch)
        return train_loss, val_loss
    
    def score(self, X, y, class_weights=None) -> float:
        with evaluating(self.net):
            if self.output_getter is None:
                output = self.net(X).squeeze()
            else:
                output = self.output_getter(self.net, X)
        if class_weights is None:
            criterion_instance = self.criterion(**self.criterion_kwargs)
        else:
            sample_weights = self._make_sample_weights(
                y, class_weights=class_weights)
            criterion_instance = self.criterion(weight=sample_weights, 
                                                **self.criterion_kwargs)
            
        return criterion_instance(output, y).item()

    def plot_learning_curves(self, ax=None, plot_lr: bool=False, **kwargs):
        if plot_lr:
            raise NotImplementedError("plt_lr not implemented yet")
        if ax is None:
            fig, ax = plt.subplots(1, **kwargs)
        else:
            fig = None
        ax.plot(self.score_steps_, self.train_loss_history_, label="train")
        if self.val_loss_history_[0] is not None:
            ax.plot(self.score_steps_, self.val_loss_history_, label="val")
            ax.legend()
        return fig, ax


class Dataset(torch.utils.data.Dataset):
    """
    Represents a dataset for PyTorch
    
    Args:
        X (tensor, array-like, or dataframe): shape (n_examples, n_features)
        y (tensor, array-like): shape (n_examples,)
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        """
        Denotes the total number of samples
        """
        return self.X.shape[0]

    def __getitem__(self, index):
        """
        Returns one sample of data
        """
        if isinstance(self.X, pd.DataFrame):
            return self.X.iloc[index], self.y[index]
        else:
            return self.X[index], self.y[index]


class LeakyHardtanh(nn.Module):
    def __init__(self, id_start: float=-1, id_stop: float=1, slope: float=.01, 
                 inplace: bool=False):
        super(LeakyHardtanh, self).__init__()
        self.id_start, self.id_stop = id_start, id_stop
        self.leaky_relu = nn.LeakyReLU(negative_slope=slope, inplace=inplace)
    
    def forward(self, x) -> torch.Tensor:
        x = self.leaky_relu( x-self.id_start)+self.id_start
        x = self.leaky_relu(-x+self.id_stop )*-1 + self.id_stop
        return x
    
    def __repr__(self):
        extra_repr  = f"id_start={self.id_start}, "
        extra_repr += f"id_stop={self.id_stop}, "
        extra_repr += f"slope={self.leaky_relu.negative_slope}, "
        extra_repr += f"inplace={self.leaky_relu.inplace}"
        return f"LeakyHardtanh({extra_repr})"


class ElementWiseMultiplication(nn.Module):
    """
    n_out = n_in
    linear layers have bias to allow for some x_i to be just passed through, 
    i.e. z_i = x_i * bias, where bias=1

    somewhat inspired by https://keras.io/api/layers/merging_layers/multiply/
    and
    https://medium.com/octavian-ai/incorporating-element-wise-multiplication-can-out-perform-dense-layers-in-neural-networks-c2d807f9fdc2
    """
    def __init__(self, n_in: int):
        super(ElementWiseMultiplication, self).__init__()
        self.l0 = nn.Linear(n_in, n_in)
        self.l1 = nn.Linear(n_in, n_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a0 = self.l0(x)
        a1 = self.l1(x)
        prod = a0 * a1
        return prod


class MSRELoss:
    """
    Mean Squared Relative Error

    For use as criterion in regression with pytorch models.

    Args:
        min_denominator (float, optional):
            Relative error, i.e. `error / target`, might blow up for small 
            targets. `min_denominator` is the minimum of `target` used in the 
            transform of the abs. error to the relative error.
    """
    def __init__(self, min_denominator: float=.01):
        self.min_denominator = min_denominator
    
    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        target_ = target.clone()
        target_[target_ < self.min_denominator] = self.min_denominator
        abs_error = target - output
        rel_error = abs_error / target_
        return torch.mean(rel_error ** 2)


class StoppingCriterion:
    """
    Base class for all stopping criteria (e.g. early stopping rule 
    implementations). To use as parent class only. Not to be instanciated.
    """
    def __init__(self):
        self.train_losses   = []
        self.val_losses     = []
    
    def step(self, train_loss: Optional[float]=None, 
             val_loss: Optional[float]=None):
        """
        Supply the rule with the most recent scores. 

        Parameters
        ----------
        train_loss : Optional[float], optional
            [description], by default None
        val_loss : Optional[float], optional
            [description], by default None

        Raises
        ------
        TypeError
            [description]
        TypeError
            [description]
        """
        if train_loss is not None and type(train_loss) is not float:
            raise TypeError("train_loss must be None or of type float, "
                            f"got {type(train_loss)}.")
        if val_loss is not None and type(val_loss) is not float:
            raise TypeError("val_loss must be None or of type float, "
                            f"got {type(val_loss)}.")
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    @property
    def fulfilled(self) -> bool:
        """
        Returns `True` iff training can be stopped following this rule.
        """
        raise NotImplementedError()

    @property
    def n_steps(self) -> int:
        return len(self.train_losses)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__extra_repr__()})"
    
    def __extra_repr__(self) -> str:
        """
        Meant to be overridden with init arguments, e.g. 
        'min_steps=5, tol=1e-3'.
        """
        return "..."


class PlateauStoppingCriterion(StoppingCriterion): #todo min_steps
    def __init__(self, tol: float=1e-5, patience: int=5):
        super().__init__()
        self.tol = tol
        self.patience = patience

        self.best_loss = np.inf
        self.steps_without_progress = 0

    @property
    def fulfilled(self) -> bool:
        if self.n_steps < self.patience:
            return False
        elif self.steps_without_progress > self.patience:
            return True
        else:
            return False
        
    def step(self, train_loss, val_loss) -> None:
        super(PlateauStoppingCriterion, self).step(train_loss, val_loss)
        loss = train_loss if val_loss is None else val_loss

        progress_was_made = loss < self.best_loss - self.tol
        if progress_was_made:
            self.best_loss = min(self.best_loss, loss) # min() due to tol
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1

    def __extra_repr__(self) -> str:
        return f"tol={self.tol}, patience={self.patience}"



class EarlyStoppingUPsRule(StoppingCriterion):
    """
    Early Stopping UP_s rule. Uses `val_loss` values if they are passed each 
    step, otherwise uses `train_loss` values.
    
    Args:
        s (int): 1 or greater. Number of consecutive strips for which the UP_1
            rule has to yield "stop". (correct?) #todo
        k (int): strip length.
        
    References:
        Montavon et al 2012: Neural Networks: Tricks of the Trade, 2nd edition,
            Springer, ISBN 978-3-642-35288-1, p.57
    """
    def __init__(self, s: int=1, k: int=5, verbose: bool=True):
        super().__init__()
        self.s = s
        self.k = k
        self.verbose = verbose

    @property
    def fulfilled(self) -> bool:
        """
        Returns `True` iff training can be stopped following this rule.
        """
        if len(self.val_losses) == 0:
            return False
        
        val_losses_available = self.val_losses[-1] is not None
        return self._decide_for_step(
            self.val_losses if val_losses_available else self.train_losses, 
            self.s)

    # contains recursive calls
    def _decide_for_step(self, scores: List[float], s: int) -> bool:
        n_steps  = len(scores)
        n_strips = n_steps // self.k # complete strips
        
        # only check on end-of-strip steps, thus never stop "during" a strip
        step_is_end_of_strip = n_steps % self.k == 0
        too_few_strips       = n_strips < s+1
        if too_few_strips or not step_is_end_of_strip:
            return False
        
        if s == 0:
            if self.verbose:
                self._print_info(s, n_steps, n_strips, decision=True)
            return True
        else:
            # decision for the strip at hand
            loss_at_t         = scores[-1]
            loss_at_t_minus_k = scores[-1-self.k] # loss at end of previous strip
            score_worsened    = loss_at_t > loss_at_t_minus_k
            
            # decision for s-1
            s_minus_1_decision = self._decide_for_step(
                scores[:(n_strips-1)*self.k], s=s-1)
            
            # combine decisions
            decision = score_worsened and s_minus_1_decision
            if self.verbose:
                self._print_info(s, n_steps, n_strips, decision, score_worsened)
            return decision

    def _print_info(self, s, n_steps, n_strips, decision, 
                    score_worsened: Optional[bool]=None) -> None:
        if s==0:
            print(f"Early Stopping UP-rule with k = {self.k}")
        else:
            print(f"\ts:{s:2d}, n_steps:{n_steps:3d}, n_strips:{n_strips:3d}, "
            f"score_up: {str(score_worsened):5s} -> {decision}")

    def __extra_repr__(self) -> str:
        return f"s={self.s}, k={self.k}"


@contextmanager
def evaluating(net):
    """
    Temporarily switch to evaluation mode.

    License:
        MIT License, Christoph Heindl, Jul 2018, 
        https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
    """
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


class PostProcessor:
    """
    Base class for postprocessing classes, which allow to perform additional 
    tasks after a step, an epoch, or after completion of training within an
    SklearnTorchWrapper instance. 
    """
    def __init__(self):
        self.sktw = None
        self.train_dl = None
        self.val_dl   = None

        self.n_epochs = 0
        self.n_steps  = 0
        self.finalized = False

    def register(self, sktw: SklearnTorchWrapper, train_dl, val_dl) -> None:
        self.sktw = sktw
        self.train_dl = train_dl
        self.val_dl   = val_dl

    def step(self, batch, output, loss) -> None:
        self.n_steps += 1

    def epoch(self) -> None:
        self.n_epochs += 1

    def final(self) -> None:
        self.finalized = True


class Ensemble:
    """
    Ensemble estimators that are already fitted.
    
    Args:
        estimators:
            List of (name, estimator) tuples.
    """
    def __init__(self, estimators):
        self.estimators = estimators
    
    def __iter__(self):
        return self.estimators.__iter__
    
    def __len__(self):
        return self.estimators.__len__()
    
    def predict(self, X) -> np.ndarray:
        raise NotImplementedError()
    
    def predict_proba(self, X) -> np.ndarray:
        return self._predict_proba_outputs(X).mean(axis=2)
        
    def _predict_proba_outputs(self, X) -> np.ndarray:
        all_outputs = [e.predict_proba(X) for _, e in self.estimators]
        return np.stack(all_outputs, axis=2)

    def _predict_all_proba_outputs(self, X) -> np.ndarray:
        all_outputs = [e.predict_all_probas(X) for _, e in self.estimators]
        return np.stack(all_outputs, axis=2)
    
    def predict_proba_and_std(self, X) -> Tuple[np.ndarray, np.ndarray]:
        proba_pos = self._predict_proba_outputs(X)[:,1,:] # shape (n_samples, n_estimators)
        p_hat, std = proba_pos.mean(axis=1), proba_pos.std(axis=1)
        return p_hat, std

    def predict_all_probas(self, X) -> Tuple[np.ndarray, np.ndarray]:
        assert hasattr(self.estimators[0][1], "predict_all_probas"), \
            "at least the first estimator has no predict_all_probas method"
        P = self._predict_all_proba_outputs(X)
        P_mean, P_std = P.mean(axis=2), P.std(axis=2)
        return P_mean, P_std
        
    def __repr__(self):
        return f"Ensemble(estimators =\n{self.estimators})"


class HistogramCalibrator:
    def __init__(self, n_bins: int=10, strategy="quantile", interp_method="linear"):
        self.n_bins   = n_bins
        self.strategy = strategy
        self.interp_method = interp_method
        
    def fit(self, y_proba, y_true, r_pos_original=None):
        keep_neg_rate = knr(y_true.mean(), r_pos_original) \
                        if r_pos_original is not None else None
        rdf = reliability_bins(
            y_true, y_proba, bin_range=(0, 1), mode=self.strategy, 
            as_frame=True, n_bins=self.n_bins, keep_neg_rate=keep_neg_rate)
        
        true_ratios = rdf["y_true_mean" if r_pos_original is None 
                          else "y_true_mean_adj"]
        proba_means = rdf["y_proba_mean"]
        
        # prepend 0 and append 1
        true_ratios = np.r_[0, true_ratios, 1]
        proba_means = np.r_[0, proba_means, 1]
        
        # make interpolator settings (unlike the interpolator function, these 
        # are easily pickleable)
        self.interp_kwargs_ = {
            "x": proba_means,
            "y": true_ratios,
            "kind": self.interp_method
        }
        
        # try if it works
        self.predict(y_proba)
        return self
        
    @property
    def is_fit(self):
        if hasattr(self, "interp_kwargs_"):
            return True
        else:
            return False
    
    def predict(self, y_proba) -> np.ndarray:
        if self.is_fit:
            f = interpolate.interp1d(**self.interp_kwargs_, bounds_error=True)
            return f(y_proba)
        else:
            raise RuntimeError("HistogramCalibrator is not fit yet")

