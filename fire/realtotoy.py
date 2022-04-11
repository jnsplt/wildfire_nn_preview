import warnings
import pickle
import os

from typing import List, Dict, Any, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale, power_transform

import fire.utils.etc as uetc
import fire.toydata as toy
from realexperiments.utils import make_nf


X_COLS = ["x0","x1","x2"]


def main_probas(y, nf) -> pd.DataFrame:
    """
    Compute some basic probabilities that should be approximated when 
    mimicking some dataset with toydata.
    * p(y)
    * p(y | nf)
    * p(y | not nf)
    * p(nf)
    """
    y, nf = np.array(y) > 0, np.array(nf) > 0
    p_y = y.mean()
    p_y_nf = y[nf].mean()
    p_y_not_nf = y[~nf].mean()
    p_nf = nf.mean()
    return pd.DataFrame({
        "p_y": [p_y], "p_y_nf": [p_y_nf], 
        "p_y_not_nf": [p_y_not_nf], "p_nf": [p_nf]
    })

def compare_main_probas(Xy_real, nf_real, Xy_toy, nf_toy) -> pd.DataFrame:
    df = pd.concat([
        main_probas(Xy, nf).assign(id = id) 
        for Xy, nf, id 
        in zip([Xy_real, Xy_toy], [nf_real, nf_toy], ["real", "toy"])
    ]).set_index("id")
    return df

def make_toydata_params(Xy, gmm, i, nf_col, ps_scaling, pi_scaling, var12_scaling):
    """
    Xy: incl. column `cluster`
    """
    Xy = Xy.assign(x_nf = Xy[nf_col] > 0)

    sample_ratios = Xy.groupby("cluster").size() / Xy.shape[0]
    if i not in sample_ratios.index:
        return None # no samples

    n_samples_in_cluster = len(Xy.query("cluster == @i"))
    if n_samples_in_cluster < 500:
        warnings.warn(f"Cluster {i} too small ({n_samples_in_cluster})")
        return None

    py_not_nf = Xy.query("~x_nf").groupby("cluster")["fire"].mean()[i]
    py = Xy.groupby("cluster")["fire"].mean()[i]
    try:
        py_nf = Xy.query("x_nf").groupby("cluster")["fire"].mean()[i]
    except KeyError as e:
        warnings.warn(f"no data on nf found in cluster{i}, {e}")
        return None

    mu0, mu1 = gmm.means_[i, 0], gmm.means_[i, 1]
    pf = uetc.sigmoid(mu0-mu1) #todo make for not_nf means

    var12 = gmm.covariances_[i, 1,2] * var12_scaling
    pi_not_nf = py_not_nf / pf
    try:
        mu2 = float(uetc.inverse_sigmoid((pi_not_nf * pi_scaling)+.00000001))
    except ValueError as e:
        raise ValueError(f"mu2 couldn't be determined, pi_not_nf={pi_not_nf}... {e}")

    p_spread = (py_not_nf - py_nf) / (py_not_nf-pf) * ps_scaling
    if p_spread > 1:
        warnings.warn(f"p_spread = {p_spread} > 1 for i={i}. Setting to 1.0.")
        p_spread = 1.0

    params = {
        "name": i, 
        "mu0": mu0,
        "mu1": mu1,
        "mu2": mu2, 
        
        "var0": gmm.covariances_[i, 0,0],
        "var1": gmm.covariances_[i, 1,1],
        "var2": gmm.covariances_[i, 2,2],

        "var01": gmm.covariances_[i, 0,1],
        "var12": var12,
        "var_eps":  0,

        "h2_trafo": uetc.sigmoid,
        
        "p_spread": p_spread,
        "n_neighbors": 1, 
        "n_samples_factor": sample_ratios[i], #todo use gmm.weights_[i] instead
        "h4": 1.0,
        "rnd_seed": i
    }
    return params

def estimate_params(Xy, n_clusters: int, 
                    z_scaling: float, x0_shift: float=0, 
                    ps_scaling: float=1, pi_scaling:float=1, 
                    var12_scaling: float=1, seed=1
                   ) -> List[Dict[str,Any]]:
    """
    [summary]

    Args:
        Xy ([type]): not preprocessed yet (not standardized yet)
        n_clusters (int): [description]
        z_scaling (float): [description]
        ps_scaling (float, optional): [description]. Defaults to 1.
        pi_scaling (float, optional): [description]. Defaults to 1.
        x0_shift (float, optional): [description]. Defaults to 0.
        var12_scaling (float, optional): [description]. Defaults to 1.
        seed (int, optional): [description]. Defaults to 1.

    Returns:
        list: [description]
    """
    Xy = preprocess(Xy, keep_all_cols=True, 
                    z_scaling=z_scaling, x0_shift=x0_shift)
    gm = GaussianMixture(n_components=n_clusters, random_state=seed+1)
    gm.fit(Xy[X_COLS].to_numpy())

    Xy = Xy.assign(cluster = gm.predict(Xy[X_COLS].to_numpy()))
    process_params = [
        p for p in [make_toydata_params(
            Xy, gm, i, nf_col="x3", 
            ps_scaling=ps_scaling,
            pi_scaling=pi_scaling,
            var12_scaling=var12_scaling) 
        for i in range(gm.n_components)] 
        if p is not None and p["n_samples_factor"] > 0
    ]
    return process_params

def sample_Xy(params, expset, seed=1, n_samples=1_000_000, verbose=True):
    params = toy.make_scenario_params(params, expset["scenario"], 
                                      add_noise=expset["add_noise"])
    procl = [toy.Process(**p) for p in params]
    procl = toy.ProcessList(procl, dist_rnd_seed=seed+2, 
                            nf_independent=expset.pop("nf_independent", False),
                            **expset["dist_kwargs"])
    if verbose:
        print(f"{len(procl)} processes")

    # sample data
    H, pids    = procl.generate_H(n=n_samples, shuffle=True, shuffle_seed=seed+3)
    X, Z, P, y = procl.compute_XZPy(H, pids)
    pids       = np.array(pids)

    Xy = pd.DataFrame(X, columns=["x0","x1","x2","x3"]).assign(fire = y)
    return Xy, P, procl

def preprocess(Xy, keep_all_cols=False, z_scaling=1, x0_shift=0, x_cols=None):
    if x_cols is None:
        x_cols = X_COLS
    if not keep_all_cols:
        cols = x_cols + (["cluster"] if "cluster" in Xy.columns else [])
        Xy = Xy[cols]
    Xy = Xy.assign(
        **{x: (lambda x: lambda df: scale(df[[x]])*z_scaling)(x) # scale power_transform
           for x in x_cols})
    Xy.loc[:, x_cols[0]] += x0_shift
    return Xy

def params_to_pickle(params: List[Dict[str,Any]], settings: Dict, 
                     proba_comparison: pd.DataFrame) -> None:
    params_hash = uetc.list_of_dicts_to_hash(params)
    os.makedirs(os.path.join("toyexperiments","cluster_params"), exist_ok=True)
    fpath = os.path.join("toyexperiments","cluster_params", params_hash+".pickle")
    with open(fpath, "wb") as f:
        d = {"params": params, "settings": settings, 
             "proba_comparison": proba_comparison}
        pickle.dump(d, f)
    print(f"Written to {fpath}")
    return params_hash

def elbow_plot(Xy: pd.DataFrame, n_samples=1000, **kwargs):
    """
    **kwargs: passed to `preprocess`
    """
    k = []; e = []
    cols = ["x0","x1","x2"]
    Xy = preprocess(Xy, **kwargs)

    for i in tqdm(range(2, 25)):
        for r in range(5):
            gm = GaussianMixture(n_components=i, random_state=r)
            gm.fit(Xy[cols].sample(n=n_samples, random_state=r).to_numpy())
            e.append(gm.lower_bound_)
            k.append(i)
    plt.scatter(k, e)
    plt.grid()

def make_Xy_look_like_toydata(Xy: pd.DataFrame, x_cols: List[str]
                             ) -> pd.DataFrame:
    """
    [summary]

    Args:
        Xy (pd.DataFrame): [description]
        x_cols (List[str]): 
            e.g. ["x_t2m_hist_mean","x_tp_hist_mean","x_lightn_f"]

    Returns:
        pd.DataFrame: with columns
            x0: float, x1: float, x2: float, x3: bool, fire: bool
    """
    Xy = (
        Xy.assign(x_nf = make_nf(Xy))
        [x_cols + ["x_nf","fire"]])
    Xy.columns = ["x0","x1","x2","x3","fire"]
    return Xy

def un_undersample_Xy(Xy, knr, seed) -> pd.DataFrame:
    """
    Undersample positives with `knr` as "keep-positive-rate", such that
    the ratio of positives in Xy matches that when doing simple random sampling
    (no undersampling).
    """
    sel_neg = ~Xy["fire"]
    sel_pos =  Xy["fire"]
    sel_rnd = np.random.RandomState(seed+4).choice([True, False], 
                                                    size=len(Xy), 
                                                    p=[knr, 1-knr])
    sel = sel_neg | (sel_pos & sel_rnd)
    return Xy.loc[sel]
