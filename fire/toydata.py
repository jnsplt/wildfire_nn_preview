from typing import List, Dict, Union, Optional, Tuple, Callable, Any
from dataclasses import dataclass

import warnings
import os
from copy import deepcopy
import pickle
import dill
from datetime import datetime
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import scipy.stats as scs
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import sklearn.metrics as skm

import fire.utils.etc as uetc
import fire.utils.plot as uplt
import fire.utils.ml as uml
import fire.utils.io as uio
import fire.models

import pdb


@dataclass
class Process:
    mu0: float # temperature
    mu1: float # rain
    mu2: float # ignition factor
    var0: float

    var1: float  = 0.0
    var2: float  = 0.0
    var01: float = 0.0 # if 0:  h0 and h1 independent
    var12: float = 0.0 # if 0:  h1 and h2 independent
    var_eps: float  = 0.0 # if 0: no noise
    p_spread: float = 0.0 # if 0: y not fed in again
    n_neighbors: int = 1 # raises ValueError. kept in order to not mess with
                         # old param dicts, which have the key "n_neighbors"
                         # even though it's set to 1
    h4: float        = 1.0 # p_i scaling factor, completely hidden, i.e. not in X
    pf_shift: float  = 0.0 # shifts pf, i.e. pf = sigm(h0-h1 + pf_shift)
    n_cycles: int    = 1 # "warm-up", number of times to run sim and wait for h3
    h2_trafo: Callable = uetc.clamp

    n_samples_factor: float = 1.0
    name: Optional[str]     = None
    rnd_seed: int = None

    def __post_init__(self):
        # init random number generator
        self.init_rng_from_seed() # needed before sanity check on h012

        if self.n_neighbors > 1:
            raise ValueError("n_neighbors > 1 ignored. Kept for compatibility.")

        # sanity check inputs
        assert self.var0 > 0, "var0 must be greater 0"
        assert not (np.isclose(self.var1, 0) and self.var2 > 0), \
            "Process with var1 == 0 but var2 > 0 is not defined"
        assert 0 <= self.p_spread <= 1, \
            "p_spread must be a proba within 0 and 1"

        # check if params are okay, by generating a few samples for h0, h1, 
        # and h2
        try:
            self._generate_h012(10)
        except ValueError as e:
            err_msg = "Passed variances can't be used. Original message " + \
                      "from trying to generate a few samples: " + str(e)
            if self.name is not None:
                err_msg += f" (in Process `{self.name}`)"
            raise ValueError(err_msg)

    @property
    def X_cols(self):
        return ["temperature","precipitation","ignition factor","neighboring fire"]

    @property
    def H_cols(self):
        return self.X_cols + ["p_i scaling"]

    def generate_H(self, n: int, ignore_factor: bool=False) -> np.ndarray:
        if not ignore_factor:
            n = int(n * self.n_samples_factor)
        h0, h1, h2 = self._generate_h012(n)
        h4         = np.ones_like(h0) * self.h4

        # compute h3 (neighboring fires) while keeping h0, h1, and h2 as they are
        h3_init = np.zeros_like(h0)
        H       = uetc.hstack_flat_arrays(h0, h1, h2, h3_init, h4)
        if self.p_spread > 0:
            for _ in range(self.n_cycles):
                h3 = self.compute_XZPy(H)[3]
                H  = uetc.hstack_flat_arrays(h0, h1, h2, h3, h4)

        return H
        
    def _generate_h012(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates h0 (temperature), h1 (rain), and h2 (anthropogenic factors).
        """
        if np.isclose(self.var1, 0): # __post_init__ asserts that var2 == 0
            h0_var = scs.multivariate_normal(
                mean=np.array([self.mu0]), 
                cov=np.array([[self.var0]]))
            # sample data from multivariate normal
            h0 = h0_var.rvs(size=n, random_state=self.rng)
            h1 = np.ones_like(h0) * self.mu1
            h2 = np.ones_like(h0) * self.mu2

        elif np.isclose(self.var2, 0):
            h01_var = scs.multivariate_normal(
                mean=np.array([self.mu0, self.mu1]), 
                cov=np.array([[self.var0 , self.var01],
                              [self.var01, self.var1 ]]))
            # sample data from multivariate normal
            H01    = h01_var.rvs(size=n, random_state=self.rng)
            h0, h1 = uetc.columns(H01)
            h2     = np.ones_like(h0) * self.mu2

        else:
            h012_var = scs.multivariate_normal(
                mean=np.array([self.mu0, self.mu1, self.mu2]), 
                cov=np.array([[self.var0 , self.var01, 0         ],
                              [self.var01, self.var1 , self.var12],
                              [0         , self.var12, self.var2 ]]))
            # sample data from multivariate normal
            H012 = h012_var.rvs(size=n, random_state=self.rng)
            h0, h1, h2 = uetc.columns(H012)

        return h0, h1, h2

    def compute_XZPy(self, H, add_noise: bool=True
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X   = self._compute_X(H, add_noise=add_noise)

        z_f = self._compute_z_f(H)
        z_i = self._compute_z_i(H)
        Z   = np.hstack([z_f[:,None], z_i[:,None]])

        p_f = self._compute_p_f(z_f, pf_shift=self.pf_shift)
        p_i = self._compute_p_i(z_i)
        p   = p_f * p_i
        P   = uetc.hstack_flat_arrays(p_f, p_i, p)

        y = p > self.rng.uniform(low=0, high=1, size=p.shape)
        return X, Z, P, y

    def _compute_X(self, H, add_noise: bool=True) -> np.ndarray:
        h0, h1, h2, h3, _ = uetc.columns(H)
        n = len(h0)

        # add noise
        std_eps = np.sqrt(self.var_eps) * float(add_noise)
        eps = self.rng.normal(loc=0, scale=std_eps, size=n)
        x0  = h0 + eps * self.var0

        eps = self.rng.normal(loc=0, scale=std_eps, size=n)
        x1  = h1 + eps * self.var1

        eps = self.rng.normal(loc=0, scale=std_eps, size=n)
        x2  = h2 + eps * self.var2

        x3 = h3 # no noise
        X  = uetc.hstack_flat_arrays(x0, x1, x2, x3)

        # replace constant columns by 0s
        for j in range(X.shape[1]):
            if uetc.is_constant(X[:,j], close=True):
                X[:,j] = 0
        
        return X
    
    @staticmethod
    def _compute_z_f(H) -> np.ndarray:
        h0, h1, _, _, _ = uetc.columns(H)
        return h0-h1

    def _compute_z_i(self, H) -> np.ndarray:
        _, _, h2, h3, h4 = uetc.columns(H)
        p_h2 = self.h2_trafo(h2)
        p_h3 = h3 * self.p_spread

        # probability of neither forest starting to burn without fire nearby nor
        # fire spreading from neighbor cell
        p_no_ignition = (1-p_h2) * (1-p_h3)

        # scale by h4...
        p_i = (1-p_no_ignition)
        p_i_scaled = h4 * p_i
        # ...but make sure it stays between 0 and 1
        p_i_scaled = uetc.clamp(p_i_scaled, 0, 1)

        return p_i_scaled

    @staticmethod
    def _compute_p_f(z_f, pf_shift: float=0.0) -> np.ndarray:
        return uetc.sigmoid(z_f + pf_shift)

    @staticmethod
    def _compute_p_i(z_i) -> np.ndarray:
        return uetc.clamp(z_i, 0, 1)

    def make_grid(self, size: int=8000, n_stds: float=4.0
                 ) -> np.ndarray:
        # calculate the number of grid points for each h0, h1, and h2
        points_per_dim = int(size**(1/3))
        
        # compute ticks for each of h0, h1, and h2
        std0 = np.sqrt(self.var0) + np.sqrt(self.var_eps)
        h0   = np.linspace(self.mu0 - n_stds*std0, self.mu0 + n_stds*std0, 
                           points_per_dim)
        std1 = np.sqrt(self.var1) + np.sqrt(self.var_eps)
        h1   = np.linspace(self.mu1 - n_stds*std1, self.mu1 + n_stds*std1, 
                           points_per_dim)
        std2 = np.sqrt(self.var2) + np.sqrt(self.var_eps)
        h2   = np.linspace(self.mu2 - n_stds*std2, self.mu2 + n_stds*std2, 
                           points_per_dim)

        # h3 and h4 ticks
        h3 = np.array([0,1]) if self.p_spread > 0 else np.array([0])
        h4 = np.array([self.h4])

        # 
        H_grid = uetc.grid_matrix(h0, h1, h2, h3, h4)
        return H_grid
    
    def init_rng_from_seed(self) -> None:
        self.rng = np.random.RandomState(self.rnd_seed)


class ProcessList:
    """
    Combines several processes to appear as one, and provides the option to 
    distort H non-linearly to yield X.

    Parameters
    ----------
    add_distortion : str, optional : {"neural","simple"} 
        if "neural":
            Sets up distortion modules for h0, h1, and h2, such that each h_i 
            will be mapped non-linearly to 3 features in X. X will thus have
            3+3+3+1=10 columns. 
        if "simple":
            x0 = h0 + tanh(h0-mu0+sqrt(var0))*alpha
            x1 = (h1-mu1)/2 + sigmoid(h1-mu1)*(h1-mu1)
            x2 = (h2-mu2)/2 + sigmoid(h2-mu2)*(h2-mu2)
            x3 = h3

        if None:
            X = H[:,:4] + noise (noise > 0 iff var_eps != 0)
    
    dist_alpha : float, optional
        Used in simple distortion (see add_distortion), ignored otherwise.

    nf_independent : bool, optional
        If True, `h3` is shuffled at the end of `generate_H`. Use `shuffle_seed`
        in `generate_H` to make this reproducible.


    #todo documentation below is outdated

    rnd_seed : int, optional 
        Seed to use for the setup of the distortion modules. If not None, the 
        global torch random state will be seeded by calling
        `torch.manual_seed(rnd_seed)`.
    maxae_lr_thresh : float, optional
        MaxAbsErr between p_true(H_true) and p_true(H_est), where H_est is 
        estimated by linear regression, must be greater than
        `maxae_lr_thresh * max(p_true(H_true))` for a distortion to be 
        considered suitable. This assures that the distortion is non-linear. 
        By default .2
    mae_lr_thres : float, optional
        MeanAbsErr between p_true(H_true) and p_true(H_est), where H_est is 
        estimated by linear regression, must be greater than
        `mae_lr_thres * max(p_true(H_true))` for a distortion to be 
        considered suitable. This assures that the distortion is non-linear. 
        By default .05
    maxae_rt_thresh : float, optional
        MaxAbsErr between p_true(H_true) and p_true(H_est), where H_est is 
        estimated by a regression tree, must be smaller than
        `maxae_rt_thresh * max(p_true(H_true))` for a distortion to be 
        considered suitable. This assures that little or no information is lost 
        by the distortion. By default .001
    """
    def __init__(self, processes: List[Process], 
                 add_distortion: Optional[str]=None, 
                 max_corr_lr: float=.8, min_corr_rt: float=.99, 
                 dist_max_tries: int=100, 
                 dist_verbose: bool=True, dist_rnd_seed: Optional[int]=None,
                 dist_alpha: float=2.0, nf_independent: bool=False):
        self._processes = processes
        self.add_distortion = add_distortion
        self.nf_independent = nf_independent

        # thresholds for suitability check of distortion
        self.max_corr_lr = max_corr_lr
        self.min_corr_rt = min_corr_rt

        assert isinstance(self._processes, list), \
            "param processes must be of type list"
        for elem in self._processes:
            assert isinstance(elem, Process), \
                "param processes must be a list of Process instances"
        self.init_rng_from_seed()

        if add_distortion is not None:
            self._dist_seed = dist_rnd_seed
            self._dist_rng = np.random.RandomState(self._dist_seed)
            if dist_rnd_seed is None:
                warnings.warn("dist_rnd_seed is None although distortion is on")

            if add_distortion == "neural":
                self._dist_max_tries = dist_max_tries
                self._dist_verbose   = dist_verbose
                self._setup_neural_distortion()
            elif add_distortion == "simple":
                self.dist_alpha = dist_alpha
            else:
                raise ValueError("illegal value for add_distortion")


    def generate_H(self, n: int, shuffle=True, shuffle_seed=None
                  ) -> Tuple[np.ndarray, List[int]]:
        """
        Generates H for each process in this instance of ProcessList.

        Parameters
        ----------
        n : int
            Number of samples to be generated by *each* process. Passed to each 
            of the Process instances `generate_H` method. May be altered by the 
            `n_samples_factor` attribute of each process. 
        independent_p_nf : float or None, optional
            passed to `Process.generate_H`, by default None.

        Returns
        -------
        Tuple[np.ndarray, List[int]]
            H : np.ndarray : shape (n_samples,5)
                H arrays generated by all the processes stacked vertically.
            pids : List[int] : length n_samples
                For each row in H the id of the process which generated the row.
                The id is the position of the respective process in the 
                ProcessList.
        """
        Hs   = []
        pids = []
        for pid, proc in enumerate(self):
            Hi = proc.generate_H(n)
            Hs.append(Hi)
            pids.extend([pid]*Hi.shape[0])
        H = np.vstack(Hs)

        if self.nf_independent:
            n_total = H.shape[0]
            rng = np.random.RandomState(shuffle_seed)
            shuffle_idx = rng.choice(range(n_total), size=n_total, replace=False)
            H[:,3] = H[shuffle_idx, 3]

        if shuffle:
            n_total = H.shape[0]
            rng = np.random.RandomState(shuffle_seed)
            shuffle_idx = rng.choice(range(n_total), size=n_total, replace=False)
            return H[shuffle_idx], list(np.array(pids)[shuffle_idx])
        else:
            return H, pids

    def compute_XZPy(self, H: np.ndarray, pids: List[int], add_noise: bool=True
                    ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]: 
        """
        Computes the ndarrays X, Z, P, and y for a given H.

        Parameters
        ----------
        H : np.ndarray : shape (n_samples, 5)
            As returned by `generate_H`.
        pids : List[int] : length H.shape[0], optional
            For each row in H the id of the process which generated the row, 
            by default None. The id is the position of the respective process in
            the ProcessList.
        add_noise : bool
            Passed to `Process.compute_XZPy()`, defaults to True.

        Returns
        -------
        Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]
            All `XZPy` outputs of the processes vertically stacked. See 
            docstring of Process.compute_XZPy for details on X, Z, P, and y.
        """
        n          = len(pids)
        pids       = np.array(pids) # to allow array bool operations
        X, Z, P, y = np.full((n,4), np.nan), np.full((n,2), np.nan), \
                     np.full((n,3), np.nan), np.full((n, ), np.nan)

        # whether to add noise within the Process instances or (if at all) later
        add_noise_in_process = add_noise and (self.add_distortion is None)

        for i, proc in enumerate(self._processes):
            mask = pids == i
            if np.any(mask):
                Hi = H[mask, :]
                Xi, Zi, Pi, yi = proc.compute_XZPy(
                    Hi, add_noise=add_noise_in_process)
                # put arrays into rows which correspond to the current pid
                X[mask] = Xi; Z[mask] = Zi; P[mask] = Pi; y[mask] = yi

        if self.add_distortion is not None:
            if self.add_distortion == "neural" and not self._neural_distortion_is_setup:
                X = None #todo What is this for again? For neural dist. setup?
            else:
                # override X, since X returned by indiv. processes is not needed
                X = self.distort_H(H)
                if add_noise:
                    X = self._add_noise_to_distorted_X(X, pids)

        return X, Z, P, y

    def _add_noise_to_distorted_X(self, X_dist: np.ndarray, pids: List[int]
                                 ) -> np.ndarray:
        """
        Adds noise to X as returned by `self.distort_H`.
        """
        self._assert_X_is_distorted(X_dist)
        X_dist = X_dist.copy()

        for pid in np.unique(pids):
            var_eps_i = self[pid].var_eps
            row_sel   = np.array(pids) == pid

            # add noise to all but the last columns
            # the standard scaler after distortion assures that each col of 
            # X_dist has variance 1 (except the last col; it's bool),
            # thus the variances of the features do not scale the noise
            # as in the "noising" of the individual processes
            n_cols = 9 if self.add_distortion == "neural" else 3
            Eps_i = self._dist_rng.normal(loc=0, scale=np.sqrt(var_eps_i), 
                                          size=(row_sel.sum(), n_cols))
            X_dist[row_sel, :-1] += Eps_i
        
        return X_dist

    def make_grid(self, size: int=8000, n_stds: float=4.0
                 ) -> Tuple[np.ndarray, List[int]]:
        """
        Make one grid for all processes.

        Parameters
        ----------
        size : int, optional
            Passed to each of the processes `make_grid` methods, by default 
            1_000_000
        n_stds : float, optional
            Passed to each of the processes `make_grid` methods, by default 4.0

        Returns
        -------
        Tuple[np.ndarray, List[int]]
            H_grid : np.ndarray : shape (n_grid_points, 5)
                As in `Process.make_grid()` but repeated (row-wise) n_processes
                times
            pids : List[int] : length n_grid_points * n_processes
                As required `ProcessList.compute_XZPy()`
        """
        # init vars
        h012_mins, h012_maxs = [np.inf]*3, [-np.inf]*3
        h3_uniques = set()
        h4_uniques = set()

        # calculate the number of grid points for each h0, h1, and h2
        points_per_dim = int(size**(1/3))

        # determine "range" of grid
        for i, proc in enumerate(self._processes):
            # get grid from each process
            H_grid_i = proc.make_grid(size=size, n_stds=n_stds)

            # update overall min and max values for h0, h1, and h2
            for j in range(3):
                hj           = H_grid_i[:,j]
                h012_mins[j] = min(hj.min(), h012_mins[j])
                h012_maxs[j] = max(hj.max(), h012_maxs[j])

            # update unique values for h3 and h4
            h3_uniques |= set(H_grid_i[:,3])
            h4_uniques |= set(H_grid_i[:,4])

        # make new grid with overall min max values
        h0 = np.linspace(h012_mins[0], h012_maxs[0], points_per_dim)
        h1 = np.linspace(h012_mins[1], h012_maxs[1], points_per_dim)
        h2 = np.linspace(h012_mins[2], h012_maxs[2], points_per_dim)
        h3 = np.array(list(h3_uniques))
        h4 = np.array(list(h4_uniques))

        H_grid = uetc.grid_matrix(h0, h1, h2, h3, h4)
        pids   = list(np.repeat(range(len(self)), H_grid.shape[0]))

        # copy and stack H_grid n_processes times, so that each process (with
        # its own p_spread) can compute X, Z, P, and y later
        H_grid = np.vstack([H_grid]*len(self))
        return H_grid, pids

    def init_rng_from_seed(self) -> None:
        for p in self._processes:
            p.init_rng_from_seed()

    @property
    def summary(self) -> pd.DataFrame: 
        df = pd.DataFrame(data={
            "mu0": [p.mu0 for p in self],
            "mu1": [p.mu1 for p in self],
            "mu2": [p.mu2 for p in self],
            "var0":  [p.var0 for p in self],
            "var1":  [p.var1 for p in self],
            "var2":  [p.var2 for p in self],
            "var01":   [p.var01   for p in self],
            "var12":   [p.var12   for p in self],
            "var_eps": [p.var_eps for p in self],
            "h2_trafo":  [p.h2_trafo for p in self],
            "p_spread":  [p.p_spread for p in self],
            "n_neighbors": [p.n_neighbors for p in self],
            "h4":        [p.h4       for p in self],
            "n_samples_factor": [p.n_samples_factor for p in self],
            "rnd_seed":         [p.rnd_seed         for p in self],
            "distortion": self.add_distortion
        }, index=[p.name for p in self])
        return df

    @property
    def X_cols(self):
        if self.add_distortion == "neural":
            return ["x00", "x01", "x02", "x10", "x11", "x12", "x20", "x21", 
                    "x22", "x3"]
        else:
            return self[0].X_cols

    @property
    def H_cols(self):
        return self[0].H_cols

    def __getitem__(self, i: int) -> Process:
        return self._processes[i]

    def __len__(self) -> int:
        return len(self._processes)

    @property
    def __iter__(self):
        return self._processes.__iter__

    @property
    def __next__(self):
        return self._processes.__next__

    def __repr__(self) -> str:
        s = "["
        processes = ",\n ".join([repr(p) for p in self._processes])
        s += processes + "]"
        return s

    def distort_H(self, H: np.ndarray) -> np.ndarray:
        """
        Maps H to X by distorting H. h3 is not distorted.
        """
        if self.add_distortion is None:
            raise Exception("distortion is disabled")

        if self.add_distortion == "neural":
            H_dist = [] # will hold three np.ndarray with each 3 cols
            for j in range(3):
                hj = H[:,[j]]
                hj_dist = self._neural_distortion_pipelines[j](hj)
                H_dist.append(hj_dist)
            
            h3 = H[:,[3]] # no distortion for h3
            H_dist.append(h3)
            X = np.hstack(H_dist)
            return X

        elif self.add_distortion == "simple":
            return self._simple_distortion(H)

    def _simple_distortion(self, H: np.ndarray, add_noise: bool=False
                          ) -> np.ndarray:
        if add_noise:
            raise NotImplementedError("add_noise=True not implemented")
        h0, h1, h2, h3, _ = uetc.columns(H)

        # define transformation functions
        ftanh = lambda h, mu, var: h + np.tanh(h-mu+np.sqrt(var))*self.dist_alpha
        fknee = lambda h, mu:     (h-mu)/2 + uetc.sigmoid(h-mu)*(h-mu)

        # mu values of first process
        mu0  = self._processes[0].mu0
        mu1  = self._processes[0].mu1
        mu2  = self._processes[0].mu2
        # var0 of first process
        var0 = self._processes[0].var0
        
        x0 = ftanh(h0, mu0, var0)
        x1 = fknee(h1, mu1)
        x2 = fknee(h2, mu2)
        x3 = h3 # no distortion on that one
        
        X = np.hstack([x0[:,None], x1[:,None], x2[:,None], x3[:,None]])
        return X

    def _setup_neural_distortion(self) -> None:
        self._neural_distortion_pipelines = [None]*3

        H, pids = self.generate_H(n=10000)
        if self._dist_verbose:
            print(f"Try to find suitable distortions for...")
        for j in range(3):
            self._setup_neural_distortion_j(H, pids, j)

    def _setup_neural_distortion_j(self, H: np.ndarray, pids: List[int], j: int
                                  ) -> None:
        hj = H[:,[j]] # nn.Module.forward expects shape (n_samples, >=1)

        # allow for reproducibility
        torch.manual_seed(self._dist_seed)
        
        # init randomly new nets until one is found that meets the requirements
        suitable_distortion_found = False
        for attempt in range(self._dist_max_tries):
            distort      = self.DistortionPipeline(hj, self._dist_rng)
            hj_distorted = distort(hj) # shape (n_samples, 3)

            suitable_distortion_found, corr_rt, corr_lr = \
                self._check_suitability(H, pids, j, hj_distorted)

            if self._dist_verbose:
                print(f"   h{j}: attempt {attempt+1:3d}/{self._dist_max_tries}"
                      f"   (corr rt: {corr_rt:.4f}, lr: {corr_lr:.4f})", 
                      end="\r")

            if suitable_distortion_found:
                if self._dist_verbose:
                    print(f"   h{j}: attempt {attempt+1:3d}/{self._dist_max_tries}"
                          f" => success! corr_rt: {corr_rt:.4f}, "
                          f"corr_lr: {corr_lr:.4f}")
                break
        
        if not suitable_distortion_found:
            raise RuntimeError("No suitable distortion was found. "
                               "Try to loosen distortion requirements. "
                               f"(max attempts: {self._dist_max_tries})")
        
        self._neural_distortion_pipelines[j] = distort

    def _check_suitability(self, H: np.ndarray, pids: List[int], j: int, 
                           hj_distorted: np.ndarray) -> bool:
        """
        Checks if the passed distortion module is suitable. A distortion is
        'suitable' if the distorted data 
        * is too non-linear such that linear regression estimates hj badly 
          (measured on MAE and MaxError of p(H_est) from p(H)), but
        * allows a non-linear learner (regression tree) to estimate hj very 
          closely (measured on MaxError of p(H_est) from p(H)).
        """
        hj = H[:,j]
        
        # check if hj can be recovered by a non-linear learner, i.e. regr. tree
        regtree    = DecisionTreeRegressor().fit(hj_distorted, hj)
        hj_est_rt  = regtree.predict(hj_distorted)
        corr_rt, _ = scs.pearsonr(hj, hj_est_rt)
        if corr_rt < self.min_corr_rt: 
            return False, corr_rt, None

        # check if hj can be recovered by a linear learner, i.e. linear regr. 
        linreg    = LinearRegression().fit(hj_distorted, hj)
        hj_est_lr = linreg.predict(hj_distorted)
        corr_lr, _ = scs.pearsonr(hj, hj_est_lr)
        if corr_lr > self.max_corr_lr: 
            return False, corr_rt, corr_lr

        # if arrived here, the distortion is suitable
        return True, corr_rt, corr_lr

    def _assert_X_is_distorted(self, X) -> None:
        ncols = X.shape[1]
        if self.add_distortion == "neural":
            assert ncols == 3+3+3+1, "X has an unexpected number of columns"
        elif self.add_distortion == "simple":
            # can't tell
            pass
    
    def plot_distortion(self, H: np.ndarray, X: np.ndarray, 
                        figsize=(11,3), fontsize=12, fontsize_legend=12,
                        legend_kwargs={}, simple_dist_scatter_size=.1):
        assert self.add_distortion is not None, \
            "Method only works if `self.add_distortion` is not None"
        self._assert_X_is_distorted(X)

        if self.add_distortion == "neural":
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
            for xcol_id in range(3+3+3):
                hcol_id = xcol_id // 3
                x  = X[:, xcol_id]
                h  = H[:, hcol_id]
                ax = axs[hcol_id]
                
                ax.scatter(h, x, s=.5, label=f"$x_{xcol_id}$")

                if xcol_id % 3 == 0: # first time in new ax
                    ax.set_xlabel(f"$h_{hcol_id}$", fontsize=fontsize)
                    if hcol_id == 0:
                        ax.set_ylabel(f"$x_j$", fontsize=fontsize)
                if xcol_id % 3 == 2:
                    ax.legend(markerscale=4, fontsize=fontsize)
            
            return fig, axs
        
        elif self.add_distortion == "simple":
            h0, h1, h2, _, _ = uetc.columns(H)
            x0, x1, x2, _    = uetc.columns(X)

            fig, ax = plt.subplots(1, figsize=figsize)
            h_min, h_max = np.min(H[:,:4]), np.max(H[:,:4])
            ax.plot([h_min, h_max], [h_min, h_max], 
                    color="k", linestyle="dashed", zorder=-10)

            ax.scatter(h0, x0, s=simple_dist_scatter_size, 
                       label=r"$h_0 \rightarrow x_0$")
            ax.scatter(h1, x1, s=simple_dist_scatter_size, 
                       label=r"$h_1 \rightarrow x_1$")
            ax.scatter(h2, x2, s=simple_dist_scatter_size, 
                       label=r"$h_2 \rightarrow x_2$")

            ax.set_xlabel("$h$", fontsize=fontsize)
            ax.set_ylabel("$x$", fontsize=fontsize)
            leg = ax.legend(markerscale=8, fontsize=fontsize_legend,
                            framealpha=1, edgecolor="k", **legend_kwargs)
            leg.get_frame().set_boxstyle('Square')
            return fig, ax        

    @property
    def _neural_distortion_is_setup(self):
        return all([s is not None for s in self._neural_distortion_pipelines])

    class DistortionPipeline:
        def __init__(self, h: np.ndarray, rng: np.random.RandomState):
            self.input_scaler = StandardScaler().fit(h)

            neg_slopes = rng.uniform(size=3)
            self.random_net = nn.Sequential(
                nn.Linear(1,5), nn.LeakyReLU(neg_slopes[0]), 
                nn.Linear(5,5), nn.LeakyReLU(neg_slopes[1]), 
                nn.Linear(5,5), nn.LeakyReLU(neg_slopes[2]), 
                nn.Linear(5,3))

            # for fitting of output_scaler feed through current pipeline
            h_scaled    = self.input_scaler.transform(h)
            h_distorted = self._pass_through_net(h_scaled)
            
            self.output_scaler = StandardScaler().fit(h_distorted)

        def _pass_through_net(self, h) -> np.ndarray:
            h_distorted = self.random_net(torch.Tensor(h)).detach().numpy()
            return h_distorted
        
        def __call__(self, h) -> np.ndarray:
            h_scaled    = self.input_scaler.transform(h)
            h_distorted = self._pass_through_net(h_scaled)
            return self.output_scaler.transform(h_distorted)


class ModelEvaluator:
    def __init__(self, X_train, H_train, y_train, X_test, H_test, y_test, 
                 process: Union[Process, ProcessList], 
                 pids_train: Optional[List[int]]=None,
                 pids_test: Optional[List[int]]=None,
                 grid_kwargs={}):
        # store parameters
        self.X_train, self.X_test = X_train, X_test
        self.H_train, self.H_test = H_train, H_test
        self.y_train, self.y_test = y_train, y_test
        
        if isinstance(process, ProcessList) and \
            (pids_train is None or pids_test is None):
            raise ValueError("If a ProcessList is passed, both pids_train and "
                             "pids_test must be passed as well")
        if isinstance(process, Process):
            process = ProcessList([process])
        if pids_train is None:
            pids_train = [0]*H_train.shape[0] # only one process
        if pids_test is None:
            pids_test = [0]*H_test.shape[0] # only one process
        self.process     = process
        self.pids_train  = pids_train
        self.pids_test   = pids_test
        self.grid_kwargs = grid_kwargs

        # compute probabilities for H
        self.P_train = process.compute_XZPy(H_train, pids=pids_train)[2]
        self.P_test  = process.compute_XZPy(H_test,  pids=pids_test)[2]

        # grid
        self.H_grid, self.pids_grid    = process.make_grid(**grid_kwargs)
        self.X_grid, _, self.P_grid, _ = process.compute_XZPy(
            self.H_grid, self.pids_grid, add_noise=False)
        
    def score(self, model):
        scores_on_ptrue = self.score_ptrue(model)
        scores_on_ytrue = self.score_ytrue(model)
        return {**scores_on_ptrue, **scores_on_ytrue}
    
    def score_ptrue(self, model, **kwargs
                   ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Returns a dict like
        {"ptrue": {
            "train": {
                "mse": MeanSqError (float),
                ... },
            ... },
        ... }
        """
        all_probas_method = get_all_probas_method_from_model(model)
        if all_probas_method is not None:
            P_hat_train = all_probas_method(self.X_train, **kwargs)
            P_hat_test  = all_probas_method(self.X_test, **kwargs)
            P_hat_grid  = all_probas_method(self.X_grid, **kwargs)
            proba_cols_to_iter = [0,1,2]
        else:
            # fake shape of all-proba-P-array (cols: p_f, p_i, p)
            P_hat_train = model.predict_proba(self.X_train)[:,[1,1,1]]
            P_hat_test  = model.predict_proba(self.X_test)[:,[1,1,1]]
            P_hat_grid  = model.predict_proba(self.X_grid)[:,[1,1,1]]
            proba_cols_to_iter = [2]

        proba_names = ["ptrue_f","ptrue_i","ptrue"]

        scores = {proba_names[i]: {
            "train": uml.compute_regression_scores(self.P_train[:,i], 
                                                   P_hat_train[:,i]), 
            "test": uml.compute_regression_scores( self.P_test[:,i],  
                                                   P_hat_test[:,i]), 
            "grid": uml.compute_regression_scores( self.P_grid[:,i],  
                                                   P_hat_grid[:,i])
        } for i in proba_cols_to_iter}

        return scores
    
    def score_ytrue(self, model) -> Dict[str, Dict[str, Dict[str, float]]]:
        y_hat_train = model.predict(self.X_train)
        y_hat_test  = model.predict(self.X_test)
        p_hat_train = model.predict_proba(self.X_train)[:,1]
        p_hat_test  = model.predict_proba(self.X_test)[:,1]
        
        calibration_score_kwargs = {"n_bins": 20}
        train_scores = uml.compute_classification_scores(
            self.y_train, y_hat_train, p_hat_train, y_proba=p_hat_train, 
            hard_scores=False, stats=False, kwcalib=calibration_score_kwargs)
        test_scores  = uml.compute_classification_scores(
            self.y_test, y_hat_test, p_hat_test, y_proba=p_hat_test, 
            hard_scores=False, stats=False, kwcalib=calibration_score_kwargs)
        return {"ytrue": {"train": train_scores, "test": test_scores}}
    
    def plot(self, model, n_bins=20, split: str="test",
             fname: Optional[str]=None, show: bool=True, figsize=None):
        """
        Plots pp- and reliability-plots.

        Parameters
        ----------
        model : sklearn model
            [description]
        n_bins : int, optional
            Number of bins to use in the reliability plot, by default 20
        split : str, optional
            One of "test", "train", or "grid", by default "test"
        fname : str, optional
            If not None, the plot will be saved under this path, by default None
        show : bool, optional
            If False, the plot will not be shown in interactive environments 
            (e.g. jupyter), by default True
        figsize : optional
            As in matplotlibs figure kwargs, by default None
        """
        X = getattr(self, f"X_{split}")
        p = getattr(self, f"P_{split}")[:,2]
        y = getattr(self, f"y_{split}")
        p_hat  = model.predict_proba(X)[:,1]
        fig, _ = reliability_plots(y, p, p_hat, n_bins=n_bins, 
                                   add_clipped=True, figsize=figsize)
        if fname is not None:
            plt.savefig(fname)
        if not show:
            plt.close(fig)
        
    def print_and_plot(self, model, figsize=None):
        scores = self.score(model)
        print("# True Proba Scores")
        print(pd.DataFrame(scores["ptrue"]))
        print()
        print("# True Label Scores")
        print(pd.DataFrame(scores["ytrue"]))
        print()
        self.plot(model, figsize=figsize)
        return scores

    @staticmethod
    def df_from_scores_dicts(scores: Dict[str, Dict]):
        """
        Converts a dictionary which maps 
        `model_name (str) -> score_dict (dict)` , where `score_dict` is a 
        dictionary as returned by `ModelEvaluator.score()` to a `pd.DataFrame`.
        """
        df_rows = []
        for model, model_scores in scores.items():
            rows_of_current_model = []
            for target in model_scores.keys():
                rows_of_current_model.append(
                    pd.DataFrame({**{"model": model, "target": target}, 
                                 **scores[model][target]})
                    .reset_index()
                    .rename(columns={"index": "measure"})
                    .set_index(["model","measure","target"])
                    .unstack().unstack()
                )
            df_rows.append(pd.concat(rows_of_current_model, axis=1))
        return pd.concat(df_rows)


def scenarios(process_params: List[Dict], noise_options: List[bool]=[False,True]
             ) -> Tuple[int, bool, List[Dict]]:
    """
    Scenarios:
        1) Temperature only, only one process, all defaults
        2) Add rain
        3) Add correlation between temperature and rain
        4) Add variance to h2
        5) Add correlation between h2 and rain
        6) Add second process
        7) Add neighboring fires
        8) Add "regional" p_i-biases and allow for different p_spread 
           among the processes

    Returns:
        scenario_number (int): number indicating the scenario
        with_noise (bool): 
            whether noise (var_eps) was set to 0 (False) or not (True)
        scenario_params (dict): 
            altered version of `process_params`, altering depending on scenario
    """
    last_name = process_params[-1]["name"]
    for with_noise in noise_options:
        noise_key = [] if with_noise else ["var_eps"]
        
        # 1) Temperature only, only one process, all defaults
        scenario_params = [p for p in process_params 
                           if p["name"] == last_name][0]
        scenario_params = uetc.remove_keys(
            scenario_params, ["var1","var2","var01","var12","p_spread","h4",
                              "n_samples_factor"] + noise_key)
        scenario_params["mu1"] = 0.0
        yield 1, with_noise, [scenario_params]
        
        # 2) Add rain
        scenario_params = [p for p in process_params 
                           if p["name"] == last_name][0]
        scenario_params = uetc.remove_keys(
            scenario_params, ["var2","var01","var12","p_spread","h4",
                              "n_samples_factor"] + noise_key)
        yield 2, with_noise, [scenario_params]
        
        # 3) Add correlation between temperature and rain
        scenario_params = [p for p in process_params 
                           if p["name"] == last_name][0]
        scenario_params = uetc.remove_keys(
            scenario_params, ["var2","var12","p_spread","h4","n_samples_factor"] 
                             + noise_key)
        yield 3, with_noise, [scenario_params]
        
        # 4) Add variance to h2
        scenario_params = [p for p in process_params 
                           if p["name"] == last_name][0]
        scenario_params = uetc.remove_keys(
            scenario_params, ["var12","p_spread","h4","n_samples_factor"] 
                             + noise_key)
        yield 4, with_noise, [scenario_params]
        
        # 5) Add correlation between h2 and rain
        scenario_params = [p for p in process_params 
                           if p["name"] == last_name][0]
        scenario_params = uetc.remove_keys(
            scenario_params, ["p_spread","h4","n_samples_factor"] + noise_key)
        yield 5, with_noise, [scenario_params]
        
        # 6) Add second process
        scenario_params = [uetc.remove_keys(p, ["h4","p_spread"] + noise_key) 
                           for p in process_params]
        yield 6, with_noise, scenario_params

        # 7) Add neighboring fires
        scenario_params = [uetc.remove_keys(p, ["h4"] + noise_key) 
                           for p in process_params]
        yield 7, with_noise, harmonize_p_spreads(scenario_params)
        
        # 8) Add "regional" p_i-biases
        scenario_params = [uetc.remove_keys(p, noise_key) 
                           for p in process_params]
        yield 8, with_noise, scenario_params

def make_scenario_params(params_blueprint: List[Dict[str,Any]], scenario: int, 
                         add_noise: bool=False, pf_shift: Optional[float]=None
                        ) -> List[Dict[str,Any]]:
    for i, _, scenario_params in scenarios(
        params_blueprint, noise_options=[add_noise]):
        if i == scenario:
            if pf_shift is not None:
                for j,p in enumerate(scenario_params):
                    if "pf_shift" in p.keys():
                        raise RuntimeError("params already contain pf_shift.")
                    scenario_params[j]["pf_shift"] = pf_shift
            return scenario_params

def harmonize_p_spreads(proc_params: List[Dict]) -> List[Dict]:
    proc_params = deepcopy(proc_params)
    n_samples_factors = np.r_[[p["n_samples_factor"] for p in proc_params]]
    p_spreads         = np.r_[[p["p_spread"] for p in proc_params]]
    oneforall_p_spread = np.sum(p_spreads * n_samples_factors 
                                / n_samples_factors.sum())
    for i in range(len(proc_params)):
        proc_params[i]["p_spread"] = oneforall_p_spread
    return proc_params

def scale_h2_in_process_params(process_params: Dict[str, Any], 
                               factor_mu: float, factor_vars: float, 
                               copy=True):
    if copy:
        process_params = deepcopy(process_params)
    
    mu = process_params["mu2"]
    process_params["mu2"] = mu * factor_mu
    
    if "var2" in process_params.keys():
        var = process_params["var2"]
        var = (np.sqrt(var) * factor_vars)**2
        process_params["var2"] = var
        
    if "var12" in process_params.keys():
        cov = process_params["var12"]
        cov = np.sign(cov) * (np.sqrt(np.abs(cov)) * factor_vars)**2
        process_params["var12"] = cov
        
    return process_params

def run_tests(process_params: List[Dict[str,Any]], 
              modelgenerators: Dict[str, Callable], 
              process_list_params: Dict[str, Any]={},
              r_partial_fits: List[float]=[.33,.66,1.0],
              n_trials: int=5, n_samples: int=20_000, 
              grid_kwargs={"size": 8000, "n_stds": 4.0}, 
              results_root=None, rnd_seed: Optional[int]=None, 
              scenario_name: Optional[str]=None, 
              results_dir_suffix: str=""):
    """
    Args:
        modelgenerators (dict): Dictionary which maps 
            `model_name` to `Callable` with signature
            `f() -> "SklearnModel"`.
    """
    processes = [Process(**params) for params in process_params]
    processes = ProcessList(processes, **process_list_params)
    score_dfs = [] # will hold one df for each model, trial, and partial fit
    models    = []
    fits_successful = []

    if rnd_seed is not None:
        rng = np.random.RandomState(rnd_seed)
        trial_seeds = list(rng.randint(0, 2**32 - 1, size=n_trials)) #rnd
    else:
        trial_seeds = [None]*n_trials

    # stuff to disk
    if results_root is not None:
        # create directory
        subdir = datetime.now(tz=None).strftime(r"%Y-%m-%d_%H-%M-%S")
        subdir += results_dir_suffix
        results_dir = os.path.join(results_root, subdir)
        if os.path.exists(results_dir):
            raise ValueError("Results directory (with timestamp appended) "
                             "already exists.")
        else:
            os.makedirs(results_dir)
        
        # dill settings to disk
        settings = {
            "process_params": process_params, "modelgenerators": modelgenerators, 
            "process_list_params": process_list_params, 
            "r_partial_fits": r_partial_fits, "n_trials": n_trials,
            "n_samples": n_samples, "rnd_seed": rnd_seed #todo distortion params
        }
        with open(os.path.join(results_dir, "settings.d"), "wb") as f:
            dill.dump(settings, f)

        # write short txt file on settings
        settings_summary = [
            "PROCESSES\n",
            processes.summary.to_string()+"\n\n",
            "TEST SETTINGS\n"
            f"   modelgenerators: ({len(modelgenerators)}) {list(modelgenerators.keys())}",
            f"   r_partial_fits:  {r_partial_fits}",
            f"   n_trials:        {n_trials}",
            f"   n_samples:       {n_samples}",
            f"   rnd_seed:        {rnd_seed}"
        ]
        uio.write_lines(settings_summary, os.path.join(results_dir, "settings.txt"))
    
    # calculate amount of work to be done
    n_models         = len(modelgenerators)
    n_fits_per_model = n_trials * len(r_partial_fits)
    
    # verbosity
    print(f"Models:         {n_models}")
    print(f"Fits per model: {n_fits_per_model}")
    progr = uetc.ProgressDisplay(n_fits_per_model*n_models)
    progr.start_timer().print_status()
    
    model_id = 0
    for i_trial, seed in zip(range(n_trials), trial_seeds):
        # generate data for current run 
        H, pids    = processes.generate_H(n_samples) #rnd
        X, _, P, y = processes.compute_XZPy(H, pids, add_noise=True) #rnd
        
        # split train/test 
        train_sel  = uetc.rnd_bool(n=n_samples, n_true=n_samples//2, 
                                   seed=seed) #rnd
        pids_train = list(np.array(pids)[ train_sel])
        pids_test  = list(np.array(pids)[~train_sel])
        X_train, H_train, y_train = X[ train_sel], H[ train_sel], y[ train_sel]
        X_test,  H_test,  y_test  = X[~train_sel], H[~train_sel], y[~train_sel]
        
        # save plots of probas and data
        if results_root is not None and i_trial==0:
            # proba distribution plot
            fig, _ = plot_scenario_probas(P, n_bins=20, figsize=(12,6), 
                                          zoom_y=True)
            fname = os.path.join(results_dir, f"probas_trial{i_trial}.png")
            plt.savefig(fname)
            plt.close(fig)
            
            # pairplot
            fig, ax = pairplot(X, y, processes.X_cols)
            fname = os.path.join(results_dir, f"data_trial{i_trial}.png")
            plt.savefig(fname)
            plt.close(fig)

            # pairplot for each process
            if len(processes) > 1:
                for i in set(pids):
                    sel_i   = np.array(pids) == i
                    Xi, yi  = X[sel_i], y[sel_i]
                    fig, ax = pairplot(Xi, yi, processes.X_cols)
                    pname   = processes[i].name
                    fname_suffix = "_"+pname if pname is not None else ""
                    fname = os.path.join(
                        results_dir, f"data_trial{i_trial}_pid{i}{fname_suffix}.png")
                    plt.savefig(fname)
                    plt.close(fig)
        
        # for each model...
        for mname, mgen in modelgenerators.items():
            progr_note = f"(working on {mname})"
            progr.print_status(note = progr_note)

            # init model
            # ...but first set numpy and torch seed to seed of current trial
            torch.manual_seed(seed)
            np.random.seed(seed)
            model_untouched = mgen() #rnd (potential)
            
            # fit model, score model, save plot
            for r in r_partial_fits:
                # get training data (first r*100%)
                n_train_samples    = int(r * X_train.shape[0])
                X_train_partial    = X_train[:n_train_samples]
                H_train_partial    = H_train[:n_train_samples]
                y_train_partial    = y_train[:n_train_samples]
                pids_train_partial = np.array(pids_train)[:n_train_samples]

                # init model evaluator #todo how to pass partial training set to 
                # ModelEval
                modeval = ModelEvaluator(
                    X_train_partial, H_train_partial, y_train_partial, 
                    X_test, H_test, y_test, process=processes, 
                    pids_train=pids_train_partial, pids_test=pids_test, 
                    grid_kwargs=grid_kwargs)

                # fit model
                model = deepcopy(model_untouched)
                start_time_fit = time.time()
                try:
                    model.fit(X_train_partial, y_train_partial)
                    success = True
                    fit_duration = time.time() - start_time_fit
                except Exception:
                    success = False
                    fit_duration = None
                fits_successful.append(success)
                models.append(model)

                # compute scores
                start_time_score = time.time()
                scores           = {mname: modeval.score(model)}
                score_duration   = time.time() - start_time_score

                # postprocess and augment score data
                scores = modeval.df_from_scores_dicts(scores)
                scores.loc[:, ("trial","","")]    = i_trial
                scores.loc[:, ("n_train","","")]  = n_train_samples
                scores.loc[:, ("n_score","","")]  = modeval.X_train.shape[0]  \
                                                    + modeval.X_test.shape[0] \
                                                    + modeval.X_grid.shape[0]
                scores.loc[:, ("model_id","","")] = model_id
                scores.loc[:, ("fit_duration","","")]   = fit_duration
                scores.loc[:, ("score_duration","","")] = score_duration
                score_dfs.append(scores)
                
                # save plot
                if results_root is not None:
                    # reliability and ptrue/phat-scatter plots
                    for split in ["test","train"]:
                        fname =  f"{model_id}_{mname}_trial{i_trial}"
                        fname += f"_ntrain{n_train_samples}_{split}"
                        modeval.plot(model, show=False, split=split, 
                            fname=os.path.join(results_dir, fname+".png"), 
                            figsize=(12,6))
                    
                    # all probas plot and P_hat bar plot
                    all_probas_method = get_all_probas_method_from_model(model)
                    if all_probas_method is not None:
                        P_test = P[~train_sel]
                        P_hat  = all_probas_method(X_test)

                        fig, _ = plot_all_probas(P_test, P_hat, figsize=(12,6))
                        plt.savefig(os.path.join(results_dir, 
                                                 fname+"_all_pp_scatter.png"))
                        plt.close(fig)

                        fig, _ = plot_scenario_probas(
                            P_hat, n_bins=20, figsize=(12,6), zoom_y=True)
                        plt.savefig(os.path.join(results_dir, 
                                                 fname+"_all_pp_bar.png"))
                        plt.close(fig)

                progr.update_and_print(note = progr_note)
                model_id += 1
            #end partial
        #end model
    progr.stop()
    
    # finalize scores df
    all_scores = (
        pd.concat(score_dfs)
        .astype({("trial","",""): int, ("n_train","",""): int, 
                 ("n_score","",""): int, ("model_id","",""): int})
        .reset_index()
        .set_index("model_id", append=False)
        .assign(scenario_name=scenario_name)
    )

    # save stuff
    if results_root is not None:
        # scores df
        all_scores.to_pickle(path=os.path.join(results_dir,"all_scores.p"))
        # models
        with open(os.path.join(results_dir, "models.d"), "wb") as f:
            dill.dump(models, f)
        # success flags
        with open(os.path.join(results_dir, "fits_successful.d"), "wb") as f:
            dill.dump(fits_successful, f)

        # model performance comparison boxplot
        for split in ["test","grid"]:
            for measure in ["mae","rmse"]:
                for corr_method in ["pearson","spearman"]:
                    g = plot_tests_summary(all_scores, focus_split=split,
                                           focus_measure=measure, 
                                           corr_method=corr_method)
                    fname = "model_performances"
                    fname += f"_{split}_{measure}_{corr_method}"
                    g.savefig(os.path.join(results_dir, fname+".png"))
                    plt.close(g.fig)

        # save flag file, that the run terminated successfuly
        uio.write_lines([], os.path.join(results_dir, "finished"))
    return all_scores, models, fits_successful

def get_all_probas_method_from_model(model):
    """
    Returns a function with signature `f(X: np.ndarray) -> np.ndarray`,
    where X is of shape (n_samples,4) and the returned value is P, shape 
    (n_samples,3), where the columns are `p_f`, `p_i`, and `p`, for a given 
    model.
    """
    if isinstance(model, (uml.SklearnTorchWrapper)):
        if hasattr(model, "predict_all_probas"):
            return model.predict_all_probas
        elif hasattr(model.net, "predict_all_probas"):
            f = lambda x: model.net.predict_all_probas(torch.Tensor(x)).numpy()
            return f
        elif hasattr(model.net, "factors_and_product_2d"):
            f = lambda x: model.net.factors_and_product_2d(torch.Tensor(x)).numpy()
            return f
    elif isinstance(model, fire.models.PfPiNet):
        return model.predict_all_probas
    return None

def _measure_corrs_from_scores_df(scores, splits=["grid","test"], 
                                  sorting_split="grid", sorting_measure="rmse",
                                  method: str="pearson"):
    """
    [summary]

    Parameters
    ----------
    scores : [type]
        [description]
    splits : list, optional
        One or more of {"grid","test","train"}, by default ["grid","test"]
    sorting_split : str, optional : {"grid","test","train"}
        Split to sort by, by default "grid"
    sorting_measure : str, optional : {"rmse","mae","mse","pcor","scor"}
        Measure to sort by, by default "rmse"
    method : str, optional : {"pearson", "kendall", "spearman"}
        Correlation method. Passed to pandas.DataFrame.corr(), by default 
        "pearson"

    Returns
    -------
    [type]
        [description]
    """
    return (
        scores
        [[col for col in scores.columns if col[0] in splits]]
        .corr(method=method)
        .sort_values((sorting_split,"ptrue",sorting_measure), ascending=False)
    )

# PLOT FUNCTIONS --------------------------------------------------------------
def plot_tests_summary(scores, focus_split: str="grid", 
                       focus_measure: str="rmse", corr_method: str="pearson"):
    """
    [summary] #todo

    Parameters
    ----------
    scores : [type]
        [description]
    
    focus_split : str : {"grid", "test"}

    Returns
    -------
    seaborn plot
        Facetted Boxplot with
            hue: n_train
            facetted by: measure (e.g. "grid / ptrue / mae")
            y: measure value
            x: model (categorical)
    """
    # compute how correlated all the measures are to grid/ptrue/mae
    # to sort the boxplots by these (better correlated -> more on the LHS)
    measure_corrs = _measure_corrs_from_scores_df(
        scores, sorting_split=focus_split, sorting_measure=focus_measure,
        method=corr_method)
    corr_to_split_ptrue_measure = (
        measure_corrs
        .reset_index()
        .rename(columns={"level_0": "split"}) # grid and test
        .assign(measure = lambda df: df["split"] + " / " + df["target"] + " / " 
                                     + df["measure"])
        .set_index("measure")
        [(focus_split,"ptrue",focus_measure)]
        .rename("corr_to_split_ptrue_measure")
    )

    # prepare scores for plotting
    scores_melt = (
        scores
        [[("model","",""),("n_train","",""),("trial","",""),
          (focus_split,"ptrue",focus_measure), 
          ("test","ytrue","logl"), ("test","ytrue","bssl"), 
          ("test","ytrue","ace"), ("test","ytrue","acecl"),
          ("test","ytrue","ece"), ("test","ytrue","ececl"),
          ("test","ytrue","mce"),
          ("test","ytrue","avp"), ("test","ytrue","auroc")]]
        .melt(id_vars=["model","n_train","trial"])
        .rename(columns={None: "split"}) # grid and test
        .assign(measure = lambda df: df["split"] + " / " + df["target"] + " / " 
                                     + df["measure"])
        .merge(corr_to_split_ptrue_measure, on="measure")
        .assign(abs_corr_to_split_ptrue_measure = lambda df:
            df["corr_to_split_ptrue_measure"].abs())
        .sort_values("abs_corr_to_split_ptrue_measure", ascending=False)
        .assign(corr_rounded = lambda df: df["corr_to_split_ptrue_measure"].round(2),
                measure = lambda df: df["measure"] + "\n(r = " + 
                                     df["corr_rounded"].astype(str) + ")")

    )

    g = sns.FacetGrid(scores_melt, col="measure", col_wrap=4, height=3, 
                      aspect=1.5, sharey=False)
    g.map(sns.boxplot, "model", "value", "n_train", 
          order=scores_melt["model"].unique(), 
          hue_order=scores_melt["n_train"].unique().sort(), showfliers=False)
    
    return g

def pairplot(X, y, varnames=None, n_bins=20, figsize=(12,12), 
             max_scatter_points=None, scatter_style=dict(s=2, alpha=.3),
             left_side_only: bool=True, float_fmt="%.2f"):
    d = X.shape[1]
    if varnames is None:
        varnames = [f"x{i}" for i in range(d)]

    _X = X[:max_scatter_points]
    _y = y[:max_scatter_points]
        
    fig, axs = plt.subplots(nrows=d, ncols=d, sharex=False, sharey=False, 
                            figsize=figsize)
    for icol in range(d):
        for irow in range(d):
            ax = axs[irow, icol]
            if irow < icol and left_side_only:
                fig.delaxes(ax)
            else:
                if irow != icol: # scatter
                    for b in [False,True]:
                        ax.scatter(_X[_y == b, icol], _X[_y == b, irow], 
                                c={True: "tab:red", False: "tab:gray"}[b], 
                                **scatter_style)
                else: # hist
                    if not uetc.is_constant(X[:,icol]):
                        _hist_to_ax(X[:,icol], ax, y=y.astype(int), 
                                    n_bins=n_bins, means=True, legend=irow==d-1)
                        ax.get_yaxis().set_visible(False)

                if irow == d-1:
                    ax.set_xlabel(varnames[icol])
                    ax.xaxis.set_major_formatter(FormatStrFormatter(float_fmt))
                else:
                    ax.tick_params(
                        axis='x',     # changes apply to the x-axis
                        which='both', # both major and minor ticks are affected
                        bottom=False, # ticks along the bottom edge are off
                        labelbottom=False)
                if icol == 0:
                    ax.set_ylabel(varnames[irow])
                    ax.yaxis.set_major_formatter(FormatStrFormatter(float_fmt))
                else:
                    ax.tick_params(
                        axis='y', 
                        which='both',
                        left=False,
                        labelleft=False)
                ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.subplots_adjust(wspace=.1, hspace=.1)
    return fig, ax

def _hist_to_ax(x, ax, y=None, n_bins=30, means=False, legend: bool=True, 
                div_by: str="max", min_max: Optional[Tuple[float,float]]=None,
                on_top: bool=False, max_y_height: float=1.0, 
                bar_base: float=0.0, set_label: bool=True):
    """
    inplace, but returns ax anyway

    Args:
        min_max (tuple of ints, opt): 
            min x value and max x value. If None, inferred from x.
    """
    colors = {0: "tab:gray", 1: "tab:red"} # not used if y_given == False

    y_given = y is not None
    if not y_given:
        y = np.zeros_like(x)

    # x = X[:,irow]
    if min_max is None:
        x_min, x_max = x.min(), x.max()
    else:
        x_min, x_max = min_max
    bins      = np.linspace(x_min, x_max, n_bins+1)
    bin_width = (x_max-x_min)/n_bins
    bin_mids  = bins[:-1] + bin_width/2

    D = pd.DataFrame({"x": x, "y": y})

    D_binned = (
        D.assign(x_binned = lambda df: pd.cut(df.x, bins, labels=bin_mids, 
                 include_lowest=True))
        .groupby(["x_binned","y"])
        .size().rename("n"))
    D_binned = D_binned / {"max": D_binned.max(), "sum": D_binned.sum()}[div_by]
    D_binned = D_binned.unstack()

    for _y in np.unique(y):
        add_kwargs = {"color": colors[_y] if y_given else "0.7"}
        if not on_top:
            add_kwargs["zorder"] = 0
        h = D_binned[_y] * (max_y_height - bar_base)
        ax.bar(x=D_binned.index.values, height=h, width=bin_width, 
               label=f"$y = {_y}$" if set_label else None, 
               alpha=.5, bottom=bar_base, **add_kwargs)
    
    if means:
        xmeans = D.groupby("y").mean()
        ylims = ax.get_ylim()
        for _y in np.unique(y):
            ax.plot([xmeans.loc[_y]]*2, [0,1.05], linestyle="-", linewidth=1, 
                    zorder=1, color=colors[_y])
        ax.set_ylim(*ylims)
    
    if legend and y_given:
        leg = ax.legend(markerscale=8, fontsize=10, framealpha=1, edgecolor="k")
        leg.get_frame().set_boxstyle('Square')
        for lh in leg.legendHandles:
            lh.set_alpha(1) # no alpha as for the bars
    # ax.set_ylim(-.05,1.05)
    return ax

def _make_heatmap_ax(fig, ax, mat, ax_title=None, xlabel=None, ylabel=None):
    pos = ax.matshow(mat, cmap="RdYlBu_r")
    fig.colorbar(pos, ax=ax)
    if ax_title is not None:
        ax.set_title(ax_title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_ylim(ax.get_ylim()[::-1])
    return ax

def pp_plot(p_true, p_hat, y_true, jitter=0, shuffle=True, 
            figsize=(20,7), alpha=.3, set_lims=True):
    p_true, p_hat, y_true = np.array(p_true), np.array(p_hat), np.array(y_true)
    n     = len(p_true)
    order = list(range(n))
    if shuffle:
        order = np.random.choice(order, size=n, replace=False)
        
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], c="k", linestyle="--", lw=1) # add diagonal
    colors = ["r" if y else "b" for y in y_true[order]]
    ax.scatter(uplt.add_jitter(p_true[order], jitter), 
               uplt.add_jitter(p_hat[order], jitter), 
               s=.1, alpha=alpha, c=colors)
    ax.set_xlabel("true prob")
    ax.set_ylabel("est prob")
    ax.set_aspect("equal")
    if set_lims:
        ax.set_xlim(-.05,1.05)
        y_lim = min(p_hat.max()+.2, 1.05)
        ax.set_ylim(-.05,y_lim)
    return fig, ax

def plot_all_probas(P_true, P_hat, categories=None, rolling_mean_win: int=250, 
                    n_bins: int=20, hist_on_top: bool=False, figsize=(12,6),
                    scatter_alpha=.5, line_alpha=.75, rstd_alpha=.5, 
                    zoom_factors: List[float]=[1.0, 1.0, 1.0], 
                    auto_zoom: bool=False, scatter_size: float=5.0, 
                    scatter_colors=[None]*3,
                    axs=None, zorder_diag=10):
    """
    Plots three scatter plots of (x) est. probas vs (y) true probas: (1) p_f,
    (2) p_i, and (3) p.

    Args:
        cat (array-like of int): For each sample a category: 0, 1, ...
        n_bins (int, optional): If 0, no histogram will be drawn.
        rolling_mean_win (int, optional): If 0, no rolling mean will be drawn.
    """
    if P_true.shape != P_hat.shape or P_true.shape[1] != 3:
        raise ValueError("P_true or P_hat have wrong shapes")

    axs_passed = axs is not None
    if axs_passed:
        fig = axs[0].get_figure()
    else:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    for i in range(3):
        if categories is None:
            axs[i].scatter(P_hat[:,i], P_true[:,i], s=scatter_size, 
                           c=scatter_colors[i], alpha=scatter_alpha)
        else:
            categories = np.array(categories)
            for cat in np.unique(categories):
                sample_sel = categories == cat
                n_samples_for_cat = np.sum(sample_sel)
                axs[i].scatter(P_hat[sample_sel, i], P_true[sample_sel,i], 
                               s=scatter_size, alpha=scatter_alpha, 
                               c=["tab:green","tab:blue"][cat],
                               label=f"cat {cat} ({n_samples_for_cat} samples)")
            if i == 2:
                axs[i].legend()

        # draw diagonal
        if not axs_passed: # no need to do that twice
            axs[i].plot([0, 1], [0, 1], "k--", lw=1, zorder=zorder_diag)

        # draw histogram
        if n_bins > 0:
            # add histogram of p_hat
            _hist_to_ax(P_hat[:,i], axs[i], n_bins=n_bins, 
                        min_max=(0,1/zoom_factors[i]), on_top=hist_on_top,
                        max_y_height=1/zoom_factors[i])

        # draw rolling mean
        if rolling_mean_win > 0:
            uplt.rolling_mean(
                P_hat[:,i], P_true[:,i], window_size=rolling_mean_win, 
                line_alpha=line_alpha, fill_alpha=rstd_alpha, 
                color="tab:orange", ax=axs[i])
        axs[i].set_aspect("equal")
        axs[i].set_xlabel([r"$\hat{p}_f$", r"$\hat{p}_i$", r"$\hat{p}_y$"][i])
        if i == 0:
            axs[i].set_ylabel("true probability")
        if auto_zoom:
            max_val = np.max([P_hat[:,i], P_true[:,i]])
            if max_val < .5:
                zoom_factors = deepcopy(zoom_factors)
                zoom_factors[i] = 1/max_val
        axs[i].set_xlim(-.05/zoom_factors[i], 1.05/zoom_factors[i])
        axs[i].set_ylim(-.05/zoom_factors[i], 1.05/zoom_factors[i])
    return fig, axs

def reliability_plots(y_true, p_true, p_hat, 
                      n_bins=20, rolling_mean_win: int=250,
                      add_clipped: bool=False, figsize=None):
    """
    Plots two reliability plots, one with p_true (left, scatter) and one with 
    y_true (right, bar plot).
    """
    if add_clipped and uetc.is_constant(p_hat):
        # warnings.warn("add_clipped set to False, since p_hat is constant")
        add_clipped = False
    
    nrows = 1 + add_clipped # 1 or 2
    fig, axs = plt.subplots(nrows=nrows, ncols=2, sharex=False, sharey=False, 
                            figsize=figsize) #todo third col for histogram (count p_hat)

    for irow in range(nrows):
        # scatter plot wrt p_true
        ax = axs[irow,0] if add_clipped else axs[0]
        ax.scatter(p_hat, p_true, s=5, alpha=.5)
        ax.plot([0, 1], [0, 1], "k--", lw=1) # add diagonal
        if rolling_mean_win > 0:
            order = np.argsort(p_hat)
            p_hat_sorted  = p_hat[order]
            p_true_sorted = p_true[order]
            rmean_phat  = uetc.rolling_mean(p_hat_sorted, rolling_mean_win)
            rmean_ptrue = uetc.rolling_mean(p_true_sorted, rolling_mean_win)
            ax.plot(rmean_phat, rmean_ptrue, "r-", lw=1.5, alpha=.75)
            # add histogram of p_hat
            _hist_to_ax(p_hat, ax, n_bins=n_bins, 
                        min_max=(0,1) if irow==0 else None)
        if irow == 0:
            ax.set_aspect("equal")
        else:
            ax.set_xlim(*uetc.expand_range(np.min(p_hat), np.max(p_hat), rel=.05))
        if irow==1 or not add_clipped:
            ax.set_xlabel("estimated probability")
        ax.set_ylabel("true probability")

        # bar plot wrt y_true
        ax = axs[irow,1] if add_clipped else axs[1]
        bin_range = (0,1) if irow==0 else (np.min(p_hat), np.max(p_hat))
        y_proba_bins, y_true_means, supports = uml.reliability_bins(
            y_true, p_hat, n_bins=n_bins, bin_range=bin_range)
        uml.reliability_plot(y_proba_bins, y_true_means, supports, ax=ax,
            zoom_x = irow==1)
        if irow==0 and add_clipped:
            ax.set_xlabel(None)
        if irow==1:
            y_true_means_no_na = y_true_means[~np.isnan(y_true_means)]
            all_y_values       = np.r_[p_true, y_true_means_no_na]
            ylim_low  = np.min(all_y_values)
            ylim_high = np.max(all_y_values)
            ylim_low, ylim_high = uetc.expand_range(ylim_low, ylim_high, rel=.05)
            bin_range = uetc.expand_range(*bin_range, rel=.05)
            xspan     = bin_range[1] - bin_range[0]
            yspan     = ylim_high - ylim_low
            for icol in range(2):
                ax = axs[1,icol]
                ax.set_aspect(xspan/yspan, adjustable='box') # square
                ax.set_xlim(*bin_range)
                ax.set_ylim(ylim_low, ylim_high)

    return fig, axs

def plot_scenario_probas(p, n_bins=15, figsize=None, zoom_y=False):
    """
    Plots three histograms, (1) p_f, (2) p_i, and (3) p.
    """
    bin_edges, step = np.linspace(0, 1, num=n_bins+1, retstep=True)
    bin_centers = bin_edges[:-1] + step/2
    
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize, 
                            sharey = not zoom_y)
    for j,ax in enumerate(axs):
        bin_df = (
            pd.DataFrame({"p_true": p[:,j]})
            .assign(p_binned = lambda df: 
                    pd.cut(df["p_true"], bins=bin_edges, labels=bin_centers,
                           include_lowest=True))
            .groupby("p_binned")
            .agg({"p_true": ["count"]})
        )

        p_bins = bin_df.index.values.to_numpy()
        counts = bin_df[("p_true","count")].to_numpy()
        ratios = counts / counts.sum()
        
        ax.bar(p_bins, ratios, width=step, align="center", 
               edgecolor="k", color="0.8")
        
        p_mean = p[:,j].mean()
        ax.plot([p_mean,p_mean], [-.05,1.05], "--", c="royalblue", 
                linewidth=1.25, alpha=1, label="mean")
        p_median = np.median(p[:,j])
        ax.plot([p_median,p_median], [-.05,1.05], "-", c="royalblue", 
                linewidth=1.25, alpha=1, label="median")
        
        ylim_lo, ylim_hi = 0, 1
        if zoom_y:
            ylim_hi = ratios.max()
            ylim_lo, ylim_hi = uetc.expand_range(ylim_lo, ylim_hi, rel=.05)
            yspan   = ylim_hi - ylim_lo
            ax.set_aspect(1.1/yspan, adjustable='box') # square
        else:
            ax.set_aspect("equal")
            
        ax.set_ylim(ylim_lo, ylim_hi)
        ax.set_xlim(-.05, 1.05)
        ax.set_xlabel(["$p_f$","$p_i$",r"$p = p_f~p_i$"][j])
        if j==0:
            ax.set_ylabel("ratio")
            ax.legend()
    
    return fig, axs

def undersample(procl: ProcessList, N: int, r_pos: float=.25, seed=None,
                tqdm_mininterval=.5, cache: bool=True):
    """
    Args:
        N (int):
            number of samples to return
        r_pos (float, opt): 
            Target ratio of positive samples after undersampling.

    Returns:
        H, pids, X, Z, P, y, r_pos_original

        r_pos_original (float): 
            Fraction of positive samples in all samples that were generated.
    """
    rng = np.random.RandomState(seed)
    make_seed = lambda: rng.randint(low=0, high=2**32 - 1)
    
    n_pos_original = 0
    n_neg_original = 0
    n = 0

    odds_pos = uetc.proba_to_odds(r_pos)
    odds_neg = 1/odds_pos
    
    H, pids, X, Z, P, y = [], [], [], [], [], []
    pbar = tqdm(total=N, desc="undersampling", mininterval=tqdm_mininterval)
    is_first_iter = True
    while n < N:
        # generate data
        _H, _pids      = procl.generate_H(n=10_000, 
                                          shuffle=True, # shuffle must be on
                                          shuffle_seed=make_seed())
        if cache and is_first_iter:
            cache_key = _make_undersampling_cache_key(_H, N, r_pos, seed)
            print("cache key (undersampling):", cache_key)
            if _key_exists_in_undersampling_cache(cache_key):
                print("loading undersampling results from cache")
                return _load_from_undersampling_cache(cache_key, procl)
            
        _X, _Z, _P, _y = procl.compute_XZPy(_H, _pids)
        _pids = np.array(_pids)
        _y    = _y.astype(bool)
        
        # stats
        n_pos_original += _y.sum()
        n_neg_original += (~_y).sum()
        
        # undersample
        idx_pos, idx_neg = np.nonzero(_y)[0], np.nonzero(~_y)[0]
        idx_neg  = idx_neg[:int(round(odds_neg * len(idx_pos)))]
        idx_keep = np.sort(np.r_[idx_pos, idx_neg])
        n += len(idx_keep)
        pbar.update(len(idx_keep))
        
        H.append(_H[idx_keep])
        pids.append(_pids[idx_keep])
        X.append(_X[idx_keep])
        Z.append(_Z[idx_keep])
        P.append(_P[idx_keep])
        y.append(_y[idx_keep])

        is_first_iter = False
    pbar.close()
        
    # stack batch results
    H = np.vstack(H)
    pids = np.concatenate(pids)
    X = np.vstack(X)
    Z = np.vstack(Z)
    P = np.vstack(P)
    y = np.concatenate(y)
    
    # shuffle (and drop some samples if n > N)
    rng = np.random.RandomState(seed)
    idx_rnd = rng.choice(np.arange(n), size=N, replace=False)
    H, pids, X, Z, P, y = (
        H[idx_rnd], pids[idx_rnd], X[idx_rnd], Z[idx_rnd], 
        P[idx_rnd], y[idx_rnd]
    )
    
    r_pos_original = n_pos_original / (n_pos_original + n_neg_original)
    n_original     = n_pos_original + n_neg_original
    print(f"generated {n_original} samples")

    if cache:
        # if the function arrived here, the variable cache_key exists
        print("undersampling results are cached")
        _cache_undersampling_results(cache_key, H, pids, X, Z, P, y, 
                                     r_pos_original, procl)
    
    return H, pids, X, Z, P, y, r_pos_original

def _make_undersampling_cache_key(_H, N, r_pos, seed) -> str:
    """
    if the same settings were used before, _H was generated identically
    before as well. 
    => hash _H, look for that hash in cache, if results exist in cache
    load results 
    """
    d = {"_H": _H, "N": N, "r_pos": r_pos, "seed": seed}
    return uetc.dict_to_hash(d)

def _make_undersampling_cache_fpath(key: str) -> str:
    return os.path.join("_toy_undersampling_cache", key+".pickle")

def _key_exists_in_undersampling_cache(key: str) -> bool:
    cache_fpath = _make_undersampling_cache_fpath(key)
    return os.path.exists(cache_fpath)

def _load_from_undersampling_cache(key: str, procl: ProcessList):
    """
    same return as `undersample`

    procl: see _cache_undersampling_results
    """
    cache_fpath = _make_undersampling_cache_fpath(key)
    with open(cache_fpath, "rb") as f:
        from_cache = pickle.load(f)

    H    = from_cache["H"]
    pids = from_cache["pids"]
    X    = from_cache["X"]
    Z    = from_cache["Z"]
    P    = from_cache["P"]
    y    = from_cache["y"]
    r_pos_original = from_cache["r_pos_original"]
    rnd_states = from_cache["rnd_states"]

    for p, rs in zip(procl, rnd_states):
        p.rng.set_state(rs)

    return H, pids, X, Z, P, y, r_pos_original

def _cache_undersampling_results(key, H, pids, X, Z, P, y, r_pos_original,
                                 procl: ProcessList):
    """
    procl needed to cache the results along with the states of the RandomState
    instances in procl. When loading results from cache, the RandomStates must
    be set to the states at which they would have arrived if they generated 
    the samples without caching to not mess with subsequent generate tasks.  
    """
    os.makedirs("_toy_undersampling_cache", exist_ok=True)
    cache_fpath = _make_undersampling_cache_fpath(key)
    rnd_states = [p.rng.get_state() for p in procl]
    d = {"H": H, "pids": pids, "X": X, "Z": Z, "P": P, "y": y, 
         "r_pos_original": r_pos_original, "rnd_states": rnd_states}
    with open(cache_fpath, "wb") as f:
        pickle.dump(d, f)
