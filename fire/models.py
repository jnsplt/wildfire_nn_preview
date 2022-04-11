from typing import Callable, Any, List, Iterable, Tuple, Optional, Union, Dict
import warnings
import dill # required by pickle module to pickle lambda functions
import pickle
import datetime
from copy import deepcopy

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
import sklearn.metrics as skm
import sklearn.calibration as skc

import torch.nn as nn
import torch.functional as F
import torch

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.interpolate import LinearNDInterpolator

import fire.utils.etc as uetc
import fire.utils.ml as uml
import fire.utils.pandas as upd


class Perceptron(torch.nn.Module):
    def __init__(self, n_in, activation = nn.Sigmoid()):
        super(Perceptron, self).__init__()
        self.fc  = nn.Linear(n_in, 1)
        self.act = activation
        
    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        return x


class UpperBoundLogisticRegression(torch.nn.Module):
    def __init__(self, n_in, activation = nn.Sigmoid(), p_max_init=1, 
                 n_est: int=100, verbose: bool=True):
        super(UpperBoundLogisticRegression, self).__init__()
        self.verbose = verbose
        self.p_max_init = p_max_init
        self.p_max = p_max_init
        self.n_est = n_est
        self.fc  = nn.Linear(n_in, 1)
        self.act = activation
        # self.upb = nn.Linear(1, 1, bias=False)
        
        # custom_weight   = torch.ones_like(self.upb.weight)*p_max
        # self.upb.weight = nn.Parameter(custom_weight, requires_grad=False)
        
    def forward(self, x):
        print(x.shape)
        x = self.fc(x)
        x = self.act(x)
        # x = self.upb(x)
        x = x * self.p_max
        return x
    
    def two_pass_fit(self, sklearn_model, X: np.ndarray, y: np.ndarray):
        p_max_est = estimate_p_max(X, y, sklearn_model, N=self.n_est)
        if self.verbose:
            print("2-pass fit")
            print(f"initial p_max:  {sklearn_model.net.p_max:.4f}")
        sklearn_model.net.p_max = p_max_est
        if self.verbose:
            print(f"estimated pmax: {sklearn_model.net.p_max:.4f}\n")


class FeatureFusionNetSigmoid(nn.Module):
    def __init__(self, n_f: int=1, n_i: int=0):
        super(FeatureFusionNetSigmoid, self).__init__()
        # flammability feature branch
        self.n_f   = n_f
        self.f_fc  = nn.Linear(n_f, 1, bias=True)
        self.f_act = nn.Sigmoid()
        
        # ignition src feature branch
        self.n_i   = n_i
        self.i_fc  = nn.Linear(max(1,n_i), 1, bias=True)
        self.i_act = nn.Sigmoid()
        
    def forward(self, x):
        x_f, x_i = self._split_features(x)
        x_f = self.forward_xf(x_f) # forward flammability features
        x_i = self.forward_xi(x_i) # forward ign src features
        x   = x_f * x_i            # fusion
        return x
    
    def forward_xf(self, x_f):
        x_f = self.f_fc(x_f)
        x_f = self.f_act(x_f)
        return x_f
    
    def forward_xi(self, x_i=None):
        """
        If x_i is None, a single 1 is fed to the ignition src branch
        """
        if self.n_i == 0:
            if x_i is not None:
                raise ValueError("x_i must be None, since n_i is 0")
            # generate 1s
            x_i = torch.ones(1, 1)
        x_i = self.i_fc(x_i)
        x_i = self.i_act(x_i)
        return x_i
    
    def predict_all_probas(self, x, clamp: bool=False):
        """
        Predicts p_f, p_i, and p (shape (n,3))
        """
        n = x.shape[0]
        x_f, x_i = self._split_features(x)
        x_f = self.forward_xf(x_f) # forward flammability features
        x_i = self.forward_xi(x_i) # forward ign src features
        if x_i.shape[0] == 1:
            x_i = x_i.repeat((n,1))
        x = x_f*x_i
        probas = torch.stack([x_f.flatten().detach(), 
                              x_i.flatten().detach(), 
                              x.flatten().detach()], dim=1)
        return probas
    
    def _split_features(self, x):
        x_f = x[:,:self.n_f]
        x_i = x[:,self.n_f:] if self.n_i > 0 else None
        return x_f, x_i


class MultiLayerPerceptron(nn.Module):
    def __init__(self, n_in: int, hidden: List[int], n_out: int, 
                 act: nn.Module=nn.Sigmoid()):
        super().__init__()
        self.n_in   = n_in
        self.hidden = hidden
        self.n_out  = n_out
        
        hidden_layers  = [nn.Linear(hidden[i], hidden[i+1]) 
                          for i in range(len(hidden)-1)]
        layers = [nn.Linear(n_in, hidden[0]), act]
        for h in hidden_layers:
            layers.append(h)
            layers.append(act)
        layers += [nn.Linear(hidden[-1], n_out), act]
        self._model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self._model(x)


class ModuleProduct(nn.Module):
    def __init__(self, nets: nn.ModuleList, slices: Iterable[slice]):
        super(ModuleProduct, self).__init__()
        assert len(nets) == len(slices), \
            "nets and slices must be of same length"
        self.nets   = nets
        self.slices = slices
        
    def forward(self, x):
        _, product = self.factors_and_product(x)
        return product
    
    def factors_and_product(self, x: torch.Tensor
                           ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forwards x but returns all intermediate outputs as well, i.e. the 
        outputs of the individual nets.

        Parameters
        ----------
        x : torch.Tensor
            As it would be passed to `self.forward`.

        Returns
        -------
        outputs : List[torch.Tensor] : length len(self.nets)
        product : torch.Tensor
        """
        outputs = []
        for n, s in zip(self.nets, self.slices):
            outputs.append(n(x[:, s]))

        product = torch.ones_like(outputs[0])
        for i in range(len(self.nets)):
            product *= outputs[i]
        return outputs, product
    
    def factors_and_product_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forwards x but returns all intermediate outputs as well, i.e. the 
        outputs of the individual nets.

        Might fail, if the individual nets do not output a tensor with only one
        element for each sample.

        Parameters
        ----------
        x : torch.Tensor
            As it would be passed to `self.forward`.

        Returns
        -------
        torch.Tensor : shape (n_samples, len(self.nets)+1)
            Stacked tensor of all net-outputs and the final product. Thus, the
            product of all elements of a row but the last one, is the last 
            element. For all but the last elements: The i-th element is the 
            output of the i-th net in `self.nets`. 
        """
        outputs, product = self.factors_and_product(x)
        outputs = [o.flatten().detach() for o in outputs]
        outputs.append(product.flatten().detach())
        return torch.stack(outputs, dim=1)


class ProbaProductNet(nn.Module):
    """
    [summary]

    Parameters
    ----------
    net_0 : nn.Module
        [description]
    slice_0 : Union[slice, List[int], List[str]]
        Column selector for `x` in `forward`. If `x` will be of type 
        `pd.DataFrame`, slices must be passed as list of str. 
    net_1 : nn.Module
        [description]
    slice_1 : Union[slice, List[int], List[str]]
        [description]
    double_precision : bool, optional
        [description], by default False
    """
    def __init__(self, 
                 net_0: nn.Module, slice_0: Union[slice, List[int], List[str]], 
                 net_1: nn.Module, slice_1: Union[slice, List[int], List[str]],
                 double_precision: bool=True):
        super(ProbaProductNet, self).__init__()
        self.net_0   = net_0
        self.net_1   = net_1
        self.slice_0 = slice_0
        self.slice_1 = slice_1
        self.double_precision = double_precision

        if self.double_precision:
            self.net_0.double()
            self.net_1.double()
        else:
            self.net_0.float()
            self.net_1.float()

    def forward(self, x):
        _, _, product = self._intermediates_and_product(x)
        return product
    
    def _intermediates_and_product(self, x: torch.Tensor) -> Tuple[torch.Tensor, 
                                                                   torch.Tensor, 
                                                                   torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor or pandas.DataFrame or numpy.ndarray
            If `torch.Tensor`, as it would be passed to `self.forward`. 

        Returns
        -------
        p_0 : torch.Tensor
        p_1 : torch.Tensor
        product : torch.Tensor
        """
        x_0, x_1 = self._split_and_convert_x(x)

        p_0 = self.net_0(x_0)
        p_1 = self.net_1(x_1)

        product = p_0 * p_1
        return p_0, p_1, product
    
    def predict_all_probas(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forwards x but returns all intermediate outputs as well, i.e. the 
        outputs of the individual nets.

        Might fail, if the individual nets do not output a tensor with only one
        element for each sample.

        Parameters
        ----------
        x : torch.Tensor
            As it would be passed to `self.forward`.

        Returns
        -------
        torch.Tensor : shape (n_samples, 3)
            Stacked tensor of p_0, p_1, and their product.
        """
        p_0, p_1, product = self._intermediates_and_product(x)

        p_0     = p_0.flatten().detach()
        p_1     = p_1.flatten().detach()
        product = product.flatten().detach()

        return torch.stack([p_0, p_1, product], dim=1)

    def _split_and_convert_x(self, x: Union[torch.Tensor, pd.DataFrame, np.ndarray]
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, pd.DataFrame):
            x_0 = x.loc[:, self.slice_0].to_numpy()
            x_1 = x.loc[:, self.slice_1].to_numpy()
        else: # assume Tensor, DoubleTensor, or np.ndarray
            x_0 = x[:, self.slice_0]
            x_1 = x[:, self.slice_1]
        
        # convert to Tensor or DoubleTensor
        T   = torch.DoubleTensor if self.double_precision else torch.Tensor
        x_0 = T(x_0)
        x_1 = T(x_1)
        return x_0, x_1


class ScalarWeightedNet(nn.Module):
    def __init__(self, basenet: nn.Module):
        super(ScalarWeightedNet, self).__init__()
        self.basenet = basenet
        self.weight_and_sigmoid = nn.Sequential(
            nn.Linear(1, 1, bias=False), # merely a weight
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.basenet(x)
        x = self.weight_and_sigmoid(x)
        return x


class PfPiCalibrator:
    """
    Args:
        pad (float):
            If > 0, qf bins of this size (in percentiles) will be added on the 
            left and right. Use this to add very small bins to prevent high 
            min(pf_hat) and early start of plateauing of pf_hat to 1.
        manual_bins_pf (list or None):
            List of percentiles (0-100). Overrides pad and n_bins_pf. 0 and 100
            are not added automatically.
        manual_bins_pi (list or None):
            List of percentiles (0-100). Overrides n_bins_pi. 0 and 100 are not 
            added automatically.
        means_as_pos (bool, optional):
            If True, means (one mean for each qf-bin) of qf values will be used 
            to position the qf reference points for scaling. If False, bin-mids
            will be used.
        min_n_samples_last_bin (int, optional): 
            Minimum number of samples in each of the rightmost qf-bins (those 
            with the highest qf values). If there are fewer samples in any of 
            these, a RuntimeError is raised. As these bins are taken as 
            reference for all other scalings in the corresponding qi-bin, this 
            number should be high enough to allow for kinda significant y-means.
            By default 100.
        method (str, optional):
            one of {"isotonic", "linear"} (or all other options in 
            `scipy.interpolate.interp1d`, arg `kind`, however not recommended)
        pi_binning (str, optional):
            one of {"uniform","quantile"}. For quantile, bin edges for pi are
            computed from quantiles. For uniform, bin edges are spaced linearly
            from 0 to 1. 
        behavior_last_bin_empty (str, optional):
            one of {"raise","drop"}. Behavior for the case that a pf bin 
            on the rightmost end (max pf within one pi bin) is emtpy or has 
            too few samples (see `min_n_samples_last_bin`).
        limit_pf_hat_scaled (str or None, optional): 
            one of {"micro", "macro", None}. To calibrate pf_hat for each
            bin, each bin gets a `pf_hat_scaled` assigned which equals
            `y_mean_m / y_mean_M`, where `y_mean_m` is the mean of y in the 
            corresponding bin (m) and `y_mean_M` is that of the last bin, i.e.
            with greatest qf values. It can happen (e.g. just due to randomness 
            in sampling) that `y_mean_m > y_mean_M` which would lead to 
            `pf_hat_scaled > 1`. `pf_hat_scaled` is calculated for each qf-bin 
            in each qi-bin.

            "micro": (upper-)limit `pf_hat_scaled` to 1 in each qi-bin 
                individually
            "macro": (upper-)limit `pf_hat_scaled` to 1 after averaging the 
                `pf_hat_scaled` values over all qi-bins
            None: do not limit at all

            Either way, if any `pf_hat_scaled > 1` a warning is shown. 
            Furthermore, the number of `pf_hat_scaled` values greater 1 on micro 
            level is stored for in `self.pf_hat_scaled_values_greater_one_`. 
        ref_bin (str, optional):
            one of {"last","max"}. The qf-bin to take as reference for scaling.
            "last": the last bin is considered to have pf_hat=1 and all others
                are adjusted accordingly.
            "max": the bin with highest y_mean is considered to have pf_hat=1 
                and all others are adjusted accordingly.

    Properties:
        calibrator_ (sklearn.isotonic.IsotonicRegression):
            Exists iff fitted.
        bin_results_ (Dict):
            For inspection only. Contains for each pi bin what was returned by 
            `uml.reliability_bins` in `_scale_pf_isolated`.
        n_bins_pf_, n_bins_pi_ (both int):
            Actual number of bins (differs from n_bins_pf and n_bins_pi if
            pf_hat or pi_hat were constant during fit or if they had too few
            unique values for the targeted number of bins)
    """
    def __init__(self, n_bins_pf: int, n_bins_pi: int, precision: int=10,
                 use_py_hat=False, pad: float=1.0, method="linear", 
                 pi_binning: str="quantile", behavior_last_bin_empty: str="drop",
                 manual_bins_pf=None, manual_bins_pi=None, means_as_pos=True,
                 min_n_samples_last_bin: int=100, min_pf_hat: float=.001,
                 weighted_isotonic: bool=False, 
                 limit_pf_hat_scaled: Optional[str]=None, ref_bin: str="last"):
        self.n_bins_pf  = n_bins_pf
        self.n_bins_pi  = n_bins_pi
        self.precision  = precision
        self.min_pf_hat = min_pf_hat
        self.use_py_hat = use_py_hat
        self.pad = pad
        self.method = method
        self.pi_binning = pi_binning
        self.manual_bins_pf = manual_bins_pf
        self.manual_bins_pi = manual_bins_pi
        self.means_as_pos = means_as_pos
        self.min_n_samples_last_bin = min_n_samples_last_bin
        self.weighted_isotonic = weighted_isotonic
        self.behavior_last_bin_empty = behavior_last_bin_empty
        self.limit_pf_hat_scaled = limit_pf_hat_scaled
        self.ref_bin = ref_bin

        assert self.limit_pf_hat_scaled in {"micro","macro",None}, \
            (f"limit_pf_hat_scaled must be 'micro', 'macro', or None. "
             f"Got {self.limit_pf_hat_scaled}.")
        assert self.ref_bin in {"last","max"}, \
            (f"ref_bin must be 'last' or 'max'. "
             f"Got {self.ref_bin}.")

    def scale_pf_with_pi_binning(self, pf_hat, pi_hat, y_true,
                                 r_pos_original=None) -> pd.DataFrame:
        self.pf_hat_scaled_values_greater_one_ = 0 # may get incremented in 
                                                   # _scale_pf_isolated

        if uetc.is_constant(pf_hat):
            warnings.warn("pf_hat is constant")
            pf_bins = np.array([0,1])
        else:
            if self.manual_bins_pf is None:
                edges = np.linspace(self.pad, 100-self.pad, self.n_bins_pf+1)
                if self.pad > 0:
                    edges = np.r_[0, edges, 100]
            else:
                edges = self.manual_bins_pf
            self.pf_bin_percentiles_ = edges
            pf_bins = np.percentile(pf_hat, edges)
            pf_bins = np.unique(pf_bins)
        self.n_bins_pf_ = len(pf_bins)-1
        self.qf_bins_ = pf_bins

        if uetc.is_constant(pi_hat):
            warnings.warn("pi_hat is constant")
            pi_bins = np.array([0,1])
        else:
            if self.pi_binning == "quantile":
                self.pi_bin_percentiles_ = np.linspace(0, 100, self.n_bins_pi+1) \
                                           if self.manual_bins_pi is None \
                                           else self.manual_bins_pi
                pi_bins = np.percentile(pi_hat, self.pi_bin_percentiles_)
                pi_bins = np.unique(pi_bins)
            elif self.pi_binning == "uniform":
                self.pi_bin_percentiles_ = None
                pi_bins = np.linspace(0,1,num=self.n_bins_pi)
            else:
                raise ValueError("pi_binning must be one of {uniform, quantile}")
        self.n_bins_pi_ = len(pi_bins)-1
        self.qi_bins_ = pi_bins

        r_pos_sampled = y_true.mean() if r_pos_original else None

        self.pf_scalings_all_ = ( # allow inspection afterwards
            # assign each pf and pi to their respective bins
            pd.DataFrame({"pf_hat": pf_hat, "pi_hat": pi_hat, "y_true": y_true,
                          "py_hat": pf_hat*pi_hat})
            .assign(pf_bin = lambda df: pd.cut(df["pf_hat"], bins=pf_bins, 
                                               precision=self.precision),
                    pi_bin = lambda df: pd.cut(df["pi_hat"], bins=pi_bins, 
                                               precision=self.precision)
                   )
            # consider pi in each pi_bin constant and compute scaled pf_hat values
            .groupby("pi_bin")
            .apply(lambda grp: self._scale_pf_isolated(
                grp["pf_hat"], grp["y_true"], 
                pi_bin=self._try_to_get_pi_bin_from_grp(grp), 
                r_pos_sampled=r_pos_sampled, r_pos_original=r_pos_original, 
                bin_edges=pf_bins, as_dataframe=True, 
                pi_hat=grp["pi_hat"], py_hat=grp["py_hat"]))
        )
        self.pf_scalings_all_.index.set_names(["pi_bin", "pf_bin_id"], 
                                              inplace=True)

        pf_scalings: pd.DataFrame = (
            self.pf_scalings_all_
            # for each pf_bin: average the scalings, weighted by support
            .groupby("pf_bin")
            .apply(lambda grp: pd.Series({
                "pf_hat_scaled": self.weighted_average(grp["pf_hat_scaled"], 
                                                       grp["supports"]),
                "support": grp["supports"].sum()})
            )
            # .apply(lambda grp: self.weighted_average(grp["pf_hat_scaled"], 
            #                                          grp["supports"])
            #       ) #todo mult support by pfmaxbin support? or by some p value
            # .rename("pf_hat_scaled")
            # .to_frame()
            .reset_index()
            .assign(pf_bin_pos = lambda df: np.r_[
                df["pf_bin"].values[0].left, 
                upd.interval_mids(df["pf_bin"].values[1:])
            ])
        )

        if self.limit_pf_hat_scaled == "macro":
            pf_scalings.loc[:,"pf_hat_scaled"] = np.minimum(
                pf_scalings.loc[:,"pf_hat_scaled"], 1)
        

        if self.means_as_pos:
            pf_means = (
                self.pf_scalings_all_
                .groupby("pf_bin")
                .apply(lambda grp: self.weighted_average(grp["pf_hat_means"], 
                                                         grp["supports"]))
                .rename("pf_bin_pos")
            )
            pf_scalings = (
                pf_scalings
                .drop(columns="pf_bin_pos")
                .merge(pf_means, on="pf_bin")
            )

        self.pf_always_scaled_up_ = all(
            pf_scalings["pf_bin_pos"] <= pf_scalings["pf_hat_scaled"])
        if not self.pf_always_scaled_up_:
            warnings.warn("pf will not always be scaled up. Risk of clipped pi.")

        return pf_scalings

    @staticmethod
    def _try_to_get_pi_bin_from_grp(grp: pd.DataFrame) -> Union[None, pd.Interval]:
        try:
            return grp["pi_bin"].iloc[0]
        except Exception:
            # grp is probably empty
            return None

    def _scale_pf_isolated(self, pf_hat, y_true, pi_bin, bin_edges=None, 
                           r_pos_sampled: Optional[float]=None,
                           r_pos_original: Optional[float]=None,
                           as_dataframe=False,
                           pi_hat=None, py_hat=None):
        """
        Args:
            r_pos_sampled:
                ratio of positive samples in training data. If r_pos_original 
                is not None, this one must not be None as well. Ignored otherwise.
            r_pos_original:
                ratio of positive samples when not over-/undersampling, thus 
                prior p(y)
            pi_bin: 
                Only needed as key in `self.bin_results_`. 
            pi_hat:
                Merely for logging (will be added as column to output df)
        """
        if self.use_py_hat:
            y_true = py_hat
        
        pf_bins, pf_hat_means, y_true_means, supports = uml.reliability_bins(
            y_true, pf_hat, bin_edges=bin_edges, mids_as_labels=False)

        if pi_hat is not None:
            _, _, pi_hat_means, _ = uml.reliability_bins(
                pi_hat, pf_hat, bin_edges=bin_edges, mids_as_labels=False)
        else:
            pi_hat_means = np.full_like(pf_hat_means, np.nan)
        
        if supports[-1] < self.min_n_samples_last_bin:
            if self.behavior_last_bin_empty == "raise" and bin_edges is not None:
                raise RuntimeError("A last bin (reference for scaling) has too "
                                   f"few samples (pi_bin: {pi_bin}, n. samples: "
                                   f"{supports[-1]}) // try less bins or less "
                                   "min_n_samples_last_bin")
            return None

        # adjust for oversampling
        y_true_means_raw = y_true_means.copy() # for inspection in output
        if r_pos_original is not None: #  and (not self.use_py_hat) #todo analyze
            y_true_means = uml.adjust_bin_means(y_true_means, 
                                                r_pos_sampled=r_pos_sampled, 
                                                r_pos_original=r_pos_original)

        # log results of lines above
        if not hasattr(self, "bin_results_"):
            self.bin_results_ = {}
        self.bin_results_[pi_bin] = {
            "pf_bins": pf_bins, "y_true_means": y_true_means, "supports": supports
        }

        # scaling
        y_mean_of_max_pf_hat = y_true_means[-1] if self.ref_bin=="last" \
                               else np.max(y_true_means)
        
        #todo anything with correlation pf pi? assume locally linear

        pf_hat_scaled_raw = y_true_means / y_mean_of_max_pf_hat
        if np.any(pf_hat_scaled_raw > 1):
            warnings.warn(f"{np.sum(pf_hat_scaled_raw > 1)} pf_hat_scaled "
                          "values for one pi_bin were >1. "
                          "More n_bins_pi might help. "
                          "Values are clamped at 1 for further processing. ")
            self.pf_hat_scaled_values_greater_one_ += np.sum(pf_hat_scaled_raw > 1)
        
        # ensure near-zero at left end as ramp-up in isotonic later
        current_leftmost_interval = pf_bins[0] 
        if current_leftmost_interval.left <= 0:
            # currently: should not occur anymore; was due to rounding error in
            # pd.cut in uml.reliability_bins. But better safe than sorry.
            warnings.warn("leftmost interval had edge <= 0: "
                          f"{current_leftmost_interval.left}")
            pf_bins    = np.r_[pf_bins] # can't set new category as long as 
                                        # the type of pf_bins is Categorical
            pf_bins[0] = pd.Interval(0, current_leftmost_interval.right)

        # prepend interval which will later be used as lower limit 
        new_leftmost_interval = pd.Interval(0, pf_bins[0].left)
        pf_bins       = np.r_[new_leftmost_interval, pf_bins]
        pf_hat_scaled_raw = np.r_[self.min_pf_hat, pf_hat_scaled_raw]
        pf_hat_scaled = np.minimum(pf_hat_scaled_raw, 1) if \
                        self.limit_pf_hat_scaled == "micro" else pf_hat_scaled_raw
        supports      = np.r_[1, supports]
        y_true_means  = np.r_[np.nan, y_true_means] # for inspection
        y_true_means_raw = np.r_[np.nan, y_true_means_raw]
        pf_hat_means = np.r_[0, pf_hat_means]
        pi_hat_means = np.r_[np.nan, pi_hat_means]

        if as_dataframe:
            return pd.DataFrame({"pf_bin": pf_bins, 
                                 "pf_hat_scaled": pf_hat_scaled, 
                                 "pf_hat_scaled_raw": pf_hat_scaled_raw,
                                 "supports": supports,
                                 "y_true_means": y_true_means,
                                 "y_true_means_raw": y_true_means_raw, 
                                 "pf_hat_means": pf_hat_means,
                                 "pi_hat_means": pi_hat_means
                                })
        else:
            return pf_bins, pf_hat_scaled, supports
    
    def make_pf_calibrator(self, pf_hat, pf_hat_scaled, weights=None
                          ) -> Union[IsotonicRegression, None]:
        if self.method == "isotonic":
            isoreg = IsotonicRegression(y_min=self.min_pf_hat, y_max=1, 
                                        out_of_bounds="clip")
            isoreg.fit(pf_hat, pf_hat_scaled, sample_weight=weights)
            return isoreg
        else:
            # in order not to have to pickle the function that is returned by 
            # interp1d, just save the init kwargs, re-construct the interpolator
            # every time it's needed (takes a few ms), and even use these kwargs 
            # for hashing
            self._interp_kwargs_ = {
                "x": np.r_[pf_hat, 1.0],
                "y": np.r_[pf_hat_scaled, 1.0],
                "kind": self.method
            }
            # try if it works
            interpolate.interp1d(**self._interp_kwargs_, bounds_error=True)
            return None

    @property
    def interpolator(self):
        if hasattr(self, "_interp_kwargs_"):
            return interpolate.interp1d(**self._interp_kwargs_, bounds_error=True)
        else:
            return None
    
    @staticmethod
    def weighted_average(x, w) -> float:
        sel = ~(np.isnan(x) | np.isnan(w))
        return np.sum((x[sel]*w[sel])/np.sum(w[sel]))
    
    def fit(self, P_hat, y_true, r_pos_original=None):
        self.n_train_samples_ = P_hat.shape[0]
        pf_hat, pi_hat, p_hat = uetc.columns(P_hat)
        self.train_data_stats_ = (
            pd.DataFrame(P_hat, columns=["qf","qi","py"])
            .agg(["min","mean","max","median"], axis=0)
        )
        self.pf_scalings_ = self.scale_pf_with_pi_binning(
            pf_hat, pi_hat, y_true, r_pos_original=r_pos_original)
        self.calibrator_ = self.make_pf_calibrator(
            self.pf_scalings_["pf_bin_pos"], self.pf_scalings_["pf_hat_scaled"], 
            weights=self.pf_scalings_["support"] if self.weighted_isotonic \
                    else None)

        # test run
        _, self.clamped_during_training_ = self.transform(
            P_hat, return_had_to_clamp=True)
        
    def transform(self, P_hat, return_had_to_clamp: bool=False):
        pf_hat, pi_hat, p_hat = uetc.columns(P_hat)
        if self.method == "isotonic":
            pf_hat_scaled = self.calibrator_.transform(pf_hat)
        else:
            pf_hat_scaled = self.interpolator(pf_hat)
        scaling_factors = np.ones_like(pf_hat)
        scaling_factors[pf_hat != 0] = \
            pf_hat_scaled[pf_hat != 0] / pf_hat[pf_hat != 0]
        pi_hat_scaled   = pi_hat / scaling_factors
        assert np.allclose(p_hat, pf_hat_scaled*pi_hat_scaled)

        had_to_clamp = False
        if np.any(pi_hat_scaled > 1):
            had_to_clamp = True
            warnings.warn(f"{(pi_hat_scaled > 1).sum()} scaled pi_hat values "
                           "was/were >1 (now clamped)")
            # FOR DEBUGGING   #todo del  shouldn't get saved to disk
            # sel = pi_hat_scaled > 1
            # self.history = {
            #     "P_hat": P_hat,
            #     "pf_hat_scaled": pf_hat_scaled, "pi_hat_scaled": pi_hat_scaled,
            #     "problem": sel
            # }
            # --------------
            pi_hat_scaled = uetc.clamp(pi_hat_scaled, low=0, high=1)
        P_hat_scaled = np.hstack(
            [a[:,None] for a in (
                pf_hat_scaled, pi_hat_scaled, 
                pf_hat_scaled*pi_hat_scaled) # might deviate from py_hat, 
                                             # because of possible clamping
            ])
        
        if return_had_to_clamp:
            return P_hat_scaled, had_to_clamp
        else:
            return P_hat_scaled

    def plot_bin_results(self, property_to_plot="y_true_means", **kwargs):
        """
        Args:
            property_to_plot (str):
                Possible values {"y_true_means", "supports"}
            **kwargs: 
                passed to `plt.subplots`
        """
        if not hasattr(self, "bin_results_"):
            raise Exception("Calibrator not fitted yet. Cannot plot.")

        values_to_plot = {k: v[property_to_plot] for k, v 
                          in self.bin_results_.items()}

        k = list(values_to_plot.keys())[0]
        pf_bins = self.bin_results_[k]["pf_bins"]

        x_ticks = [b.mid for b in pf_bins]
        y_ticks = [b.mid for b in values_to_plot.keys()]

        values_mat = np.vstack([a for a in values_to_plot.values()])

        fig, ax = plt.subplots(1, **kwargs)
        ax.matshow(values_mat)
        ax.set_xticklabels([""]+[str(round(x, 4)) for x in x_ticks]) #todo ugly workaround
        ax.set_yticklabels([""]+[str(round(x, 4)) for x in y_ticks])
        ax.set_xlabel("$p_f$-bin mids")
        ax.set_ylabel("$p_i$-bin mids")
        ax.set_title(property_to_plot)

        return fig, ax

    def plot_qf_pf_curve(self):
        qf_grid = np.linspace(0,1, 200)
        f = self.calibrator_.transform if self.method == "isotonic" \
            else self.interpolator
        pf_grid = f(qf_grid)
        plt.plot(qf_grid, pf_grid)
        plt.plot([0,1],[0,1])

    def plot_qf_pf_curve_for_each_qi_bin(self, bin_as_label=False):
        grouped = self.pf_scalings_all_.groupby("pi_bin")
        for i, (grp_i, grp_df) in enumerate(grouped):
            plt.plot(grp_df["pf_hat_means"], grp_df["pf_hat_scaled"], 
                     label=grp_i if bin_as_label else i)
        plt.legend()
        plt.xlim(None, 1.02)


class PfPiNet:
    def __init__(self, 
                 pfnet_maker: Callable[..., nn.Module], 
                 pinet_maker: Callable[..., nn.Module],
                 cols_pf: List[int], cols_pi: List[int], 
                 pfnet_kwargs={}, pinet_kwargs={}, 
                 calibrate: bool=True, seed=None, 
                 n_bins_pf: int=5, n_bins_pi: int=5, 
                 n_train_net: Optional[int]=None,
                 double_precision: bool=True, 
                 lr: float=0.1, stop_patience: int=20, sched_patience: int=5,
                 sched_factor: float=.2, weight_decay: float=.01, 
                 stw_kwargs={}, cal_kwargs={},
                 verbose: bool=True):
        # for the record, i.e. to allow re-initiating this model, see state_dict
        self._init_argnames = locals().keys() # used in self.init_dict
        self.pfnet_maker  = pfnet_maker
        self.pfnet_kwargs = pfnet_kwargs
        self.pinet_maker  = pinet_maker
        self.pinet_kwargs = pinet_kwargs
        
        # which input columns of X are mapped to which net
        self.cols_pf = cols_pf
        self.cols_pi = cols_pi

        # optimizer settings
        self.lr = lr
        self.stop_patience = stop_patience
        self.sched_patience = sched_patience
        self.sched_factor = sched_factor
        self.weight_decay = weight_decay

        # calibrator settings
        self.calibrate = calibrate # if False, calibration is not performed
        self.n_bins_pf = n_bins_pf
        self.n_bins_pi = n_bins_pi
        self.cal_kwargs = cal_kwargs

        # other
        self.n_train_net = n_train_net
        self.double_precision = double_precision # used in ProbaProductNet
        self.stw_kwargs = deepcopy(stw_kwargs) # will be passed to SklearnTorchWrapper 
        for k in self.stw_kwargs:
            if k in ["net", "optimizer", "scheduler",  
                     "stopping_criteria", "rnd_state_split", "verbose"]:
                warnings.warn("stw_kwargs contains keywords which will be set "
                              "specifically", UserWarning)
        self.verbose = verbose
        
        # seed
        self.seed = seed # will be passed to SklearnTorchWrapper
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # net init #todo relu before
        self._pfnet = pfnet_maker(**pfnet_kwargs)
        self._pinet = pinet_maker(**pinet_kwargs)
        self._ppn   = ProbaProductNet(net_0=self._pfnet, net_1=self._pinet, 
                                      slice_0=self.cols_pf, slice_1=self.cols_pi, 
                                      double_precision=self.double_precision)
        
    @classmethod
    def load_state_dict(cls, state_dict, verbose=True):
        if verbose:
            print(f"loading {state_dict['id']}")

        # 
        instance = cls(**dill.loads(state_dict["init_kwargs"]))
        
        # restore ProductProbaNet
        net_state = pickle.loads(state_dict["net_state"])
        instance._ppn.load_state_dict(net_state)

        # restore calibrator
        cal = pickle.loads(state_dict["cal_state"])
        if cal is not None:
            instance.calibrator_ = cal

        # restore other attributes
        fit_meta_attrs_dict = pickle.loads(state_dict["fit_meta_attrs_dict"])
        if fit_meta_attrs_dict is not None:
            for k,v in fit_meta_attrs_dict.items():
                setattr(instance, k, v)

        return instance
    
    def state_dict(self):
        net_state = self._ppn.state_dict()
        sd = {
            "timestamp":   datetime.datetime.now(),
            "init_kwargs": dill.dumps(self.init_dict()),
            "net_state":   pickle.dumps(net_state),
            "fit_meta_attrs_dict": pickle.dumps(self.fit_meta_attrs_dict())
        }

        if self.calibrate and self.is_fitted:
            sd["cal_state"] = pickle.dumps(self.calibrator_)
            if self.calibrator_.method == "isotonic":
                sd["cal_hash"] = uetc.isotonic_regression_to_hash(
                    self.calibrator_.calibrator_)
            else:
                sd["cal_hash"] = uetc.dict_to_hash(self.calibrator_._interp_kwargs_)
        else:
            sd["cal_state"] = pickle.dumps(None)
            sd["cal_hash"]  = "none"
        sd["netparams_hash"] = uetc.torch_state_dict_to_hash(net_state)
        sd["netarch_hash"]   = self.net_architecture_hash()

        # 
        sd["id"] =  "neta-" + sd["netarch_hash"][:4] + "_"
        sd["id"] += "netp-" + sd["netparams_hash"][:4] + "_"
        sd["id"] += "cal-"  + sd["cal_hash"][:4] + "_"
        sd["id"] += sd["timestamp"].strftime(r"%Y%m%d")

        return sd

    def net_architecture_hash(self):
        """
        Generates a hash that identifies the architecture (not the weights) of 
        the ANN. This is not completely safe, e.g. as it relies - among others -
        on the __repr__ of ProbaProductNet. Depending on the implemenations of 
        all the modules, some properties might be missing (if they are not 
        listed in the repr)
        """
        s =  repr(self._ppn) # mostly what layers, what number of neurons etc.
        s += str(self.double_precision)
        s += str(self.cols_pf) + str(self.cols_pi)
        return uetc.str_to_hash(s)

    def fit_meta_attrs_dict(self):
        fmad = {
            "lr_history_": self.lr_history_,
            "train_loss_history_": self.train_loss_history_,
            "val_loss_history_": self.val_loss_history_,
            "score_steps_": self.score_steps_,
            "classes_": self.classes_,
            "fit_time_": self.fit_time_
        } if self.is_fitted else None
        return fmad
    
    def init_dict(self):
        return {k: getattr(self, k) for k in self._init_argnames if k != "self"}
    
    def fit(self, X, y, r_pos_original=None, class_weights: dict=None, 
            regression: bool=False):
        X_net, y_net = X[:self.n_train_net], y[:self.n_train_net]
        X_cal, y_cal = X[self.n_train_net:], y[self.n_train_net:]

        self._fit_nets(X_net, y_net, class_weights=class_weights, 
                       regression=regression)
        P_hat_uncalibrated = self.predict_all_probas(X_cal)

        if self.calibrate:
            self._fit_calibrator(P_hat_uncalibrated, y_cal, 
                                 r_pos_original=r_pos_original)
        self.fit_time_ = datetime.datetime.now()
        return self

    def _fit_nets(self, X, y, class_weights=None, regression: bool=False):
        if self.verbose:
            print("\nFitting net...")

        # init optimizer stuff and sklearntorch wrapper
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self._ppn.parameters()), 
            lr=self.lr, weight_decay=self.weight_decay)
        scd = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=self.sched_patience, factor=self.sched_factor, 
            verbose=self.verbose)
        stp = [uml.PlateauStoppingCriterion(patience=self.stop_patience)]
        crt = self.stw_kwargs.pop("criterion", nn.BCELoss)
        stw = uml.SklearnTorchWrapper(
            net=self._ppn, optimizer=opt, scheduler=scd, criterion=crt, 
            stopping_criteria=stp, rnd_state_split=self.seed, 
            double_precision=self.double_precision, verbose=self.verbose, 
            is_clf=(not regression), **self.stw_kwargs)

        # fit that thing
        stw.fit(X, y, class_weights=class_weights)
        opt.zero_grad()

        # keep some things, drop rest of stw
        self.lr_history_         = stw.lr_history_
        self.train_loss_history_ = stw.train_loss_history_
        self.val_loss_history_   = stw.val_loss_history_
        self.score_steps_        = stw.score_steps_
        self.classes_            = stw.classes_
    
    def _fit_calibrator(self, P_hat, y, r_pos_original=None):
        if self.verbose:
            print("\nFitting calibrator...")
        
        self.calibrator_ = PfPiCalibrator(self.n_bins_pf, self.n_bins_pi, 
                                          **self.cal_kwargs)
        self.calibrator_.fit(P_hat, y, r_pos_original=r_pos_original)
    
    def predict(self, X):
        p_hat = self.predict_all_probas(X, skip_calibrator=True)[:,[2]]
        return (p_hat > .5) * 1
    
    def predict_proba(self, X):
        p_hat = self.predict_all_probas(X, skip_calibrator=True)[:,[2]]
        return np.hstack([1-p_hat, p_hat])

    def predict_all_probas(self, X, skip_calibrator=False):
        # forward net
        P_hat = self._ppn.predict_all_probas(X).numpy()

        # scale with calibrator
        if self.calibrate and self.is_fitted and not skip_calibrator:
            P_hat = self.calibrator_.transform(P_hat)
        
        return P_hat
    
    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "fit_time_")
    
    def plot_learning_curves(self, ax=None, plot_lr: bool=False, 
                             sel=slice(None),**kwargs):
        if not self.is_fitted:
            raise Exception("Model is not fitted so no learning curves to plot")
        if plot_lr:
            raise NotImplementedError("plt_lr not implemented yet")
        if ax is None:
            fig, ax = plt.subplots(1, **kwargs)
        else:
            fig = None
        ax.plot(
            np.r_[self.score_steps_][sel], np.r_[self.train_loss_history_][sel], 
            label="train")
        if self.val_loss_history_[0] is not None:
            ax.plot(
                np.r_[self.score_steps_][sel], np.r_[self.val_loss_history_][sel], 
                label="val")
            ax.legend()
        return fig, ax

    def __repr__(self) -> str:
        model_id = self.state_dict()["id"]
        return f"PfPiNet: {model_id}"

    def state_id(self) -> str:
        return self.state_dict()["id"]


class PfPiLoss:
    """
    Args:
        weight: passed to `nn.BCELoss`
        qfmean (float): 
            target of mean of qf. Ignored if pullmean is None. Pulling the mean 
            of `qf` towards `qfmean` is supposed to ensure that `qf` always
            underestimates `pf` and thus `qi` will be shrunk in the calibration
            process, and thus won't go over 1 (not a proba anymore). A `qfmean`
            too close to 0 might hinder the learning of `py`. `qfmean=.1` 
            seems to work fine. `py` can still be learnt because the model 
            can compensate the constraint given by `qfmean` by increasing `qi`.
        pull* (float or None, optional): 
            weight of corresponding loss. If None, loss will not be computed 
            (thus no slow down of code execution).

            pullmean:
                Pulls the mean of `qf` towards `qfmean`. A `pullmean` of `.01` 
                seems to work fine / might be a good point to start from.

            pullcenter:
                Pulls all points towards `qfmean`, but each point with weight 
                `1-d` where `d` is the distance from some qf to the actual 
                mean of all qf values.

            pulldown:
                Pulls the mean of `qf` down, i.e. adds the loss 
                `pulldown * mean(qf)`. As the mean of qf needs to be >0 in order
                for the model to work, this objective can't be satisfied and 
                will thus always compete with the LogLoss on `py` => Rather 
                for demonstrational purposes* or with very small `pulldown` 
                (e.g. `.00001` but depends of course). However, if `qfmean` is 
                not None, pulldown is only applied if the mean(qf) > qfmean.

                * e.g. "What happens in calibration if `qf` overestimates `pf`?" 
                       (answer: `pi` gets >1; set `pulldown` < 0 and use PfPiNet
                       without hidden layers for best (illustrative) results)). 

            pullpf:
                Doesn't work, but kept for reference. In some cases `qf` is fairly 
                well calibrated, but `py` is completely off then. Thus, even if 
                it might work sometimes, it's much safer and much more reliable 
                to use the usual calibration method, i.e. `PfPiCalibrator`.

            pullmax, pullmin:
                Pulls `max(qf)` towards 1 and `min(qf)` towards 0 respectively.
    """
    def __init__(self, weight=None, pull=None, 
                 pullmean=None, pulldown=None, pullcenter=None, 
                 pullmax=None, pullmin=None, pullpf=None, pullbymin=None, 
                 stretchqi=None, stretchqf=None, pullsigmoid=None, 
                 sigmoid_shift=.1, qfmean=.1, qfmin=.001):
        self.weight = weight
        self.yloss  = nn.BCELoss(weight=weight)
        self.qfmean     = qfmean
        self.pullmean   = pullmean
        self.pulldown   = pulldown
        self.pullcenter = pullcenter
        self.pullmax = pullmax
        self.pullmin = pullmin
        self.pullpf = pullpf
        self.pullbymin = pullbymin
        self.qfmin = qfmin
        self.stretchqf = stretchqf
        self.stretchqi = stretchqi
        self.pullsigmoid = pullsigmoid
        self.sigmoid_shift = sigmoid_shift
        self.pull = pull

    def __call__(self, output: Iterable[torch.Tensor], target: torch.Tensor):
        if len(output) == 3:
            qf, qi, py  = output
            qi_detached = None
            qf_detached = None
        elif len(output) == 4:
            qf, qi, py, qi_detached = output
            qf_detached = None
        elif len(output) == 5:
            qf, qi, py, qf_detached, qi_detached = output
        else:
            raise ValueError()
        loss  = self.yloss(py.squeeze(), target)
        if self.pullmean is not None:
            loss += self.pullmean * ((qf.mean()-self.qfmean)**2)
        if self.pulldown is not None:
            if self.qfmean is None:
                loss += self.pulldown * qf.mean()
            elif self.qfmean < qf.mean():
                loss += self.pulldown * qf.mean()
        if self.pullcenter is not None:
            w = 1-(qf-qf.mean()) # 1 - distance to actual qf mean => points 
                                 # close to actual qf mean are pulled harder
            loss += self.pullcenter * ((w * (qf-self.qfmean))**2).mean()
        if self.pullmin is not None:
            loss += self.pullmin * ((qf.min()-0)**2)
        if self.pullmax is not None:
            loss += self.pullmax * ((qf.max()-1)**2)
        if self.pullpf is not None:
            eps = 1e-14
            # py = py.detach()
            pf = (py-py.min()+eps)/(py.max()-py.min()+eps)
            loss += self.pullpf * ((qf-pf)**2).mean()
        if (self.pullbymin is not None) and (qf.min() > self.qfmin):
            qftarget = qf.clone().detach() - (qf.clone().detach().min() - self.qfmin)
            loss += self.pullbymin * (qf - qftarget).mean()
        if self.stretchqf is not None:
            if qf_detached is None:
                raise Exception("if stretchqf is passed, outputs have to be "
                                "obtained via `output_getter_with_detached_qi`")
            loss += self.stretchqf * (qf_detached.min()**2) # (qfmin-0)**2
        if self.stretchqi is not None:
            if qi_detached is None:
                raise Exception("if stretchqi is passed, outputs have to be "
                                "obtained via `output_getter_with_detached_qi`")
            loss += self.stretchqi * ((qi_detached.max()-1)**2)
        if self.pullsigmoid is not None:
            if self.qfmean is None:
                d = 1.0
            elif self.qfmean < qf.mean():
                d = qf.mean() - self.qfmean
            else:
                d = 0
            
            if d > 0:
                qf_shifted = torch.sigmoid(
                    uml.inverse_sigmoid(qf.detach()) - (self.sigmoid_shift * d)
                )
                loss += self.pullsigmoid * (qf - qf_shifted).mean()
        if self.pull is not None:
            loss += self.pull * qf.mean() # not squared (in order to not pull 
                                          # qf=1 too much?)
        return loss
    
    @staticmethod
    def output_getter(net: ProbaProductNet, x: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        To be passed to `uml.SklearnTorchWrapper` (param `output_getter`) in 
        case `PfPiLoss` is to be used there.
        """
        return net._intermediates_and_product(x)

    @staticmethod
    def output_getter_with_detached_qi(
            net: ProbaProductNet, x: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = list(net._intermediates_and_product(x))

        # feed x through pi-net up to the last linear layer
        _, x_i = net._split_and_convert_x(x)
        pi_net: ScalarWeightedNet = net.net_1
        assert isinstance(pi_net, ScalarWeightedNet), \
            "output_getter_with_detached_qi only works with ScalarWeightedNet"
        
        z = pi_net.basenet(x_i)
        z = z.clone().detach() # by doing that, any loss computed on qi_detached
                               # will only affect the weights in `weight_and_sigmoid`
        qi_detached = pi_net.weight_and_sigmoid(z)
        outputs.append(qi_detached)
        return outputs

    @staticmethod
    def output_getter_with_detached_qi_qf(
            net: ProbaProductNet, x: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = list(net._intermediates_and_product(x))

        # feed x through pi-net up to the last linear layer
        x_f, x_i = net._split_and_convert_x(x)
        pf_net: ScalarWeightedNet = net.net_0
        pi_net: ScalarWeightedNet = net.net_1
        assert isinstance(pf_net, ScalarWeightedNet), \
            "output_getter_with_detached_qi_qf only works with ScalarWeightedNet"
        assert isinstance(pi_net, ScalarWeightedNet), \
            "output_getter_with_detached_qi_qf only works with ScalarWeightedNet"
        
        # qf
        zf = pf_net.basenet(x_f)
        zf = zf.clone().detach() # by doing that, any loss computed on qi_detached
                               # will only affect the weights in `weight_and_sigmoid`
        qf_detached = pf_net.weight_and_sigmoid(zf)
        outputs.append(qf_detached)

        # qi
        zi = pi_net.basenet(x_i)
        zi = zi.clone().detach() # by doing that, any loss computed on qi_detached
                               # will only affect the weights in `weight_and_sigmoid`
        qi_detached = pi_net.weight_and_sigmoid(zi)
        outputs.append(qi_detached)
        return outputs


class PfPiPipeline(Pipeline):
    
    #@if_delegate_has_method(delegate='_final_estimator')
    def predict_all_probas(self, X, **clf_kwargs):
        """
        Apply transforms, and predict_all_probas of the final estimator
        
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
            
        Returns
        -------
        P_hat : array-like of shape (n_samples, 3)
            First column is pf_hat, second column pi_hat, and third column is 
            p_hat (i.e. p(y==1)).
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_all_probas(Xt, **clf_kwargs)

    def state_dict(self) -> Dict[str, Any]:
        """
        save with these lines to disk:
        >>> with open(savepath, mode="wb") as f:
        >>>    pickle.dump(pipe.state_dict(), f)

        load with these lines:
        >>> with open(savepath, mode="rb") as f:
        >>>     state_dict = pickle.load(f)
        >>> pipe = PfPiPipeline.load_state_dict(state_dict)
        """
        # check if last step is PfPiNet
        last_step_is_ppn = isinstance(self.steps[-1][1], PfPiNet)
        assert last_step_is_ppn, "last step must be a PfPiNet"
            
        state_dict = {}
        
        # remove PfPiNet from pipe for now since it is not pickle-able as is
        ppn_name, ppn = self.steps[-1]
        self.steps = self.steps[:-1]
        
        # dump pickle string of rest of pipeline
        state_dict["pipe_without_ppn_step"] = pickle.dumps(self)
        
        # save ppn
        state_dict["ppn_state"] = ppn.state_dict()
        state_dict["ppn_name"]  = ppn_name
        
        # put last step (ppn) back in order to recondition this pipeline
        self.steps.append((ppn_name, ppn))

        return state_dict

    @staticmethod
    def load_state_dict(state_dict, verbose=True) -> "PfPiPipeline":
        pipe = pickle.loads(state_dict["pipe_without_ppn_step"])
        
        # pipe misses its last step: the PPN
        ppn_name, ppn_state = state_dict["ppn_name"], state_dict["ppn_state"]
        ppn = PfPiNet.load_state_dict(ppn_state, verbose=verbose)
        pipe.steps.append((ppn_name, ppn))
        
        return pipe


from sklearn.base import TransformerMixin, BaseEstimator
class DataFrameTransformer(TransformerMixin, BaseEstimator):
    """
    Resembles somewhat `sklearn.compose.ColumnTransformer`, but outputs the 
    transformation result as a DataFrame in order to be able to still access
    columns by their names. 

    Parameters
    ----------
    transformers : list of tuples
        List of (name, transformer, columns, keep_colname) tuples specifying the
        transformer objects to be applied to subsets of the data.

        name : str
            ...
        transformer : estimator
            ...
        columns : array-like of str
            Names of columns to be passed to this estimator. 
        keep_colnames : bool
            Whether to assign the column names of the input to the output of the
            transformer. If False, names will be generated as `{name}_{i}`, 
            where `i` is the position of the respective column.

    Issues
    ------
    not tested yet with CrossValidation Gridsearch
    """
    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, X, y=None):
        self._pass_to_transformers(X, func_name="fit", y=y)

    def fit_transform(self, X, y=None):
        return self._pass_to_transformers(X, func_name="fit_transform", y=y)

    def transform(self, X):
        return self._pass_to_transformers(X, func_name="transform")

    def _pass_to_transformers(self, X: pd.DataFrame, func_name: str, y=None
                             ) -> pd.DataFrame:
        """
        Pass to fit, fit_transform, or transform of transformers.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The data to be transformed by subset.
        func_name : {"transform","fit_transform","fit"}
            Name of the function of the transformers to call.
        
        Returns
        -------
        Xt : pd.DataFrame of \
                shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.
        """
        Xt = [] # outputs of the transformers
        for tf_name, tf_instance, in_cols, keep_colnames in self.transformers:
            X_in = X[in_cols].to_numpy()
            if "fit" in func_name: # fit or fit_transform => pass y
                f = getattr(tf_instance, func_name)
                tf_output = f(X_in, y=y)
            else:
                tf_output = tf_instance.transform(X_in)

            if tf_output is not None:
                n_cols_after_tf = tf_output.shape[1]
                if n_cols_after_tf != len(in_cols) and not keep_colnames:
                    raise RuntimeError("Number of columns changed. "
                                    "Column names can't be kept.")
                out_cols = in_cols if keep_colnames else \
                    [f"{tf_name}_{i}" for i in range(n_cols_after_tf)]
                Xt.append(pd.DataFrame(tf_output, columns=out_cols))

        if "transform" in func_name:
            return pd.concat(Xt, axis=1)


class PassthroughTransformer(TransformerMixin, BaseEstimator):
    """
    Passes X through as is. Use case: as transformer in DataFrameTransformer for
    columns that are not to be transformed.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        return X

    def transform(self, X):
        return X


class PfPiCalibrator2D:
    """
    Doesn't really work well as it is right now...

    Attributes:
        reference_points (dataframe): ...
        non_decreasing_ (bool): Exists if trained.
    """
    def __init__(self, raise_decreasing=False, min_pf_hat=None, 
                 r_pos_original=None, qf_bins=5, qi_bins=5, centering="mean"):
        self.raise_decreasing = raise_decreasing
        self.min_pf_hat = min_pf_hat
        self.r_pos_original = r_pos_original
        self.qf_bins = qf_bins
        self.qi_bins = qi_bins
        self.centering = centering
        
        # init
        self.reference_points = pd.DataFrame()
        
    @staticmethod
    def _make_percentiles(n_or_list):
        if type(n_or_list) is int:
            return np.linspace(0, 100, n_or_list+1)[1:-1]
        else: # assume it's already a list of percentiles
            return n_or_list
        
    def fit(self, P_hat, y=None):
        qf, qi, py = uetc.columns(P_hat)
        if y is None:
            y = py
            
        self.qf_bin_percentiles_ = self._make_percentiles(self.qf_bins)
        self.qi_bin_percentiles_ = self._make_percentiles(self.qi_bins)            

        # determine bins #todo if constant
        self.qf_bin_edges_ = self.determine_bin_edges(
            qf, mini_bin_size=0, main_bin_percentiles=self.qf_bin_percentiles_)
        self.qi_bin_edges_ = self.determine_bin_edges(
            qi, mini_bin_size=0, main_bin_percentiles=self.qi_bin_percentiles_)
        
        # make reference points
        for ii, (ilo, ihi, isel) in enumerate(self.iter_bins(qi, self.qi_bin_edges_)):
            for fi, (flo, fhi, fsel) in enumerate(self.iter_bins(qf, self.qf_bin_edges_)):
                sel = isel & fsel
                if self.centering == "mean":
                    fcenter, icenter = qf[sel].mean(), qi[sel].mean()
                elif self.centering == "mid":
                    fcenter, icenter = np.mean([flo, fhi]), np.mean([ilo, ihi])
                elif self.centering == "median":
                    fcenter, icenter = np.median(qf[sel]), np.median(qi[sel])
                else:
                    raise ValueError(f"{self.centering} not a valid value for "
                                     "centering")
                self.add_reference_point(
                    qi_bin = ii, qf_bin = fi, 
                    qi = icenter, qf = fcenter, 
                    py = uml.adjust_bin_means(
                        y[sel].mean(), r_pos_sampled=y.mean(), 
                        r_pos_original=self.r_pos_original),
                    n_samples = sel.sum())

        self.check_non_decreasing()
        self.make_limits()
        self.make_pf_hat()
        self.reference_points.loc[:, "is_limit"] = \
            self.reference_points["n_samples"] == 0
        
        # make interpolator
        rps = self.reference_points
        self.interpolator_ = LinearNDInterpolator(
            list(zip(rps["qf"], rps["qi"])), rps["pf_hat"])
        
    def transform(self, Q):
        qf, qi, py = uetc.columns(Q)
        pf = self.interpolator_(qf, qi)
        pi = py / pf
        return np.hstack([pf[:,None], pi[:,None], py[:,None]])

    def determine_bin_edges(self, x, mini_bin_size: int, 
                            main_bin_percentiles: List[float]):
        """
        Args:
            mini_bin_size (int):
                If > 0, add bins on the very left and right with `mini_bin_size` 
                samples.
            main_bin_percentiles (list of int/float):
                Percentiles where bin edges are supposed to be, excl. 0 and 100,
                e.g. `[25,50,75]` for quartiles as bins.
        """
        percentiles = [0] # list that will contain all edges represented by percentiles
        if mini_bin_size > 0:
            lower_mini_bin_edge = (x < np.sort(x)[mini_bin_size-1]).mean() * 100
            percentiles.append(lower_mini_bin_edge)
        percentiles.extend(main_bin_percentiles)
        if mini_bin_size > 0:
            upper_mini_bin_edge = (x < np.sort(x)[-mini_bin_size]).mean() * 100
            percentiles.append(upper_mini_bin_edge)
        percentiles.append(100)

        bin_edges = np.unique(np.percentile(x, percentiles))
        if len(bin_edges) < 2:
            bin_edges = np.r_[0,1]
        return bin_edges
    
    @staticmethod
    def iter_bins(x, bin_edges):
        """
        Yields lo, hi, sel
        """
        for i in range(len(bin_edges)-1):
            lo, hi = bin_edges[i], bin_edges[i+1]
            sel  = (lo <= x) if i==0 else (lo < x) # i=0: first bin
            sel &= (x <= hi)
            yield lo, hi, sel
    
    def add_reference_point(self, qi_bin, qf_bin, qi, qf, py, n_samples):
        """
        Args:
            py (float): obtained from, e.g., `mean(y_true)` or `mean(py_hat)`
        """
        new_row = pd.DataFrame(dict(
            qi_bin = qi_bin, qf_bin = qf_bin, qi = qi, qf = qf, 
            py = py, n_samples = n_samples
        ), index=[len(self.reference_points)])
        self.reference_points = pd.concat([self.reference_points, new_row])
    
    def check_non_decreasing(self):
        all_non_decreasing = True
        grouped = self.reference_points.groupby(["qi_bin"])
        for qi_bin, grp in grouped:
            if self.raise_decreasing:
                assert uetc.non_decreasing(grp["py"]), \
                    f"decreasing py in qi_bin {qi_bin}"
            elif not uetc.non_decreasing(grp["py"]):
                warnings.warn(f"decreasing py in qi_bin {qi_bin}")
                print(f"decreasing py in qi_bin {qi_bin}")
                all_non_decreasing = False
        self.non_decreasing_ = all_non_decreasing
        
    def make_pf_hat(self):
        self.reference_points.sort_values(["qi_bin","qf_bin"], inplace=True)
        self.reference_points = (
            self.reference_points
            .groupby("qi_bin")
            .apply(self._make_pf_hat_for_single_qi_bin)
        )
        self.reference_points.sort_index(inplace=True)
    
    def _make_pf_hat_for_single_qi_bin(self, df) -> pd.DataFrame:
        df.loc[:,"cummax_py"] = df["py"].cummax()
            
        # py_left  = df["cummax_py"].values[0]
        # py_right = df["cummax_py"].values[-1]

        # pf_hat  = df["cummax_py"] - py_left + min_pf_hat
        # pf_hat /= py_right - py_left + min_pf_hat
        # pf_hat  = df["cummax_py"] / py_right
        pf_hat = self.move_max(df["cummax_py"], newmax=1)
        
        df.loc[:,"pf_hat"] = pf_hat
        return df
    
    @staticmethod
    def move_max(x, newmax):
        xmin = x.min()
        l = newmax - xmin
        x_ = x - xmin
        x_ = (x_/x_.max()) * l + xmin
        return x_
    
    def make_limits(self):
        """
        Generate additional reference points such that we can later interpolate 
        within the entire range from 0 to 1 in both dimension, pi and pf.
        """
        RP = self.reference_points.copy()
        n_bins_qf = RP["qf_bin"].nunique()
        n_bins_qi = RP["qi_bin"].nunique()
        
        for qi_bin, grp in RP.groupby("qi_bin"):
            grp.sort_values("qf_bin", inplace=True)
            min_qf = grp.iloc[0]
            self.add_reference_point(qi_bin=qi_bin, qf_bin=-1, 
                                     qi=min_qf.qi, qf=0, py=min_qf.py, 
                                     n_samples=0)
            max_qf = grp.iloc[-1]
            self.add_reference_point(qi_bin=qi_bin, qf_bin=n_bins_qf, 
                                     qi=max_qf.qi, qf=1, py=max_qf.py, 
                                     n_samples=0)
            
            # on first or last qi_bin, add corner points
            if qi_bin==0:
                self.add_reference_point(qi_bin=-1, qf_bin=-1, 
                                         qi=0, qf=0, py=min_qf.py, n_samples=0)
                self.add_reference_point(qi_bin=-1, qf_bin=n_bins_qf, 
                                         qi=0, qf=1, py=max_qf.py, n_samples=0)
            if qi_bin==(n_bins_qi-1):
                self.add_reference_point(qi_bin=n_bins_qi, qf_bin=-1, 
                                         qi=1, qf=0, py=min_qf.py, n_samples=0)
                self.add_reference_point(qi_bin=n_bins_qi, qf_bin=n_bins_qf, 
                                         qi=1, qf=1, py=max_qf.py, n_samples=0)
                
        for qf_bin, grp in RP.groupby("qf_bin"):
            grp.sort_values("qi_bin", inplace=True)
            min_qi = grp.iloc[0]
            self.add_reference_point(qi_bin=int(min_qi.qi_bin)-1, qf_bin=qf_bin, 
                                     qi=0, qf=min_qi.qf, py=min_qi.py, 
                                     n_samples=0)
            max_qi = grp.iloc[-1]
            self.add_reference_point(qi_bin=int(max_qi.qi_bin)+1, qf_bin=qf_bin, 
                                     qi=1, qf=max_qi.qf, py=max_qi.py, 
                                     n_samples=0)

    def scatter_plot(self):
        pf_hat = self.reference_points["pf_hat"]
        ax = self.reference_points.plot.scatter(
            "qf", "qi", color=pf_hat, cmap="Reds", edgecolor="k", s=40)
        return ax

    def mesh_plot(self, res=200):
        rp = self.reference_points
        qf_ticks = np.linspace(min(rp["qf"]), max(rp["qf"]), num=res)
        qi_ticks = np.linspace(min(rp["qi"]), max(rp["qi"]), num=res)
        qf_grid, qi_grid = np.meshgrid(qf_ticks, qi_ticks) # 2D grid for interpolation
        pf_hat = self.interpolator_(qf_grid, qi_grid)
        plt.pcolormesh(qf_grid, qi_grid, pf_hat, shading="auto", cmap="RdYlGn_r")
        plt.plot(rp["qf"], rp["qi"], "xk", label="reference point")

