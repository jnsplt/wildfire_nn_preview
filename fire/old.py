import torch.nn as nn
import torch.functional as F
import torch

import numpy as np
import scipy.stats as scs
import sklearn.metrics as skm

import fire.utils.etc as uetc

def repl_zero_by_neg_one(x):
    x = x.copy()
    x[np.isclose(x, 0)] = -1
    return x

def adjust_oversample_proba(estimated_proba, original_frac: float, 
                            oversampled_frac: float):
    """
    https://yiminwu.wordpress.com/2013/12/03/how-to-undo-oversampling-explained/
    """
    original_odds    = original_frac    / (1-original_frac)
    oversampled_odds = oversampled_frac / (1-oversampled_frac)
    estimated_odds   = estimated_proba  / (1-estimated_proba)
    adjusted_odds    = estimated_odds * original_odds / oversampled_odds
    adjusted_proba   = 1 / (1 + 1/adjusted_odds)
    return adjusted_proba

# high values can still come up even if p_hat is always too low (or high)
def precision_threshold_correlation(y_true, p_hat, min_samples=100):
    precs, _, thrs = skm.precision_recall_curve(y_true, p_hat)
    precs = precs[:-1] # sklearn adds a 1 to the end. Not needed
    
    # Throw away precision scores that represent too few samples
    max_thr = np.sort(p_hat)[-min_samples]
    select  = thrs < max_thr
    thrs    = thrs[select]
    precs   = precs[select]
    
    # compute correlation of precision scores and corresponding thresholds
    r, _ = scs.pearsonr(thrs, precs)
    return r

def precision_series(y_true, y_score, stepsize: int=100):
    n = len(y_true)
    desc_order = np.argsort(-y_score)
    y_true     = y_true[desc_order]
    y_score    = y_score[desc_order]
    thresh_idx = np.arange(stepsize, n, stepsize)
    
    # append n-1 to thresh_idx if it's not already the last element
    if thresh_idx[-1] != (n-1):
        # last element of precision vector should always reflect the precision
        # for the case when always predicting 1.
        thresh_idx = np.r_[thresh_idx, n-1] 
    
    # compute precisions
    n_trues_over_thrs   = np.cumsum(y_true)[thresh_idx]
    n_samples_over_thrs = np.arange(0,n)[thresh_idx]
    precisions = n_trues_over_thrs / n_samples_over_thrs
    
    thresholds = y_score[thresh_idx]
    return precisions, thresholds


class ProbabilityFunctions:
    @staticmethod
    def clipped_linear(start=0, slope=1, p_max=1):
        def f(z):
            z = (z+start) * slope
            return (uetc.clamp(z)*p_max).flatten()
        return f
    
    @staticmethod
    def sigmoidal(c=0, s=1, p_max=1):
        """
        sigmoid( (z+c)*s ) * p_max
        c: center, or "where f(x) is 0.5" (if p_max=1)
        s: the higher s, the steeper f(z)
        """
        def f(z):
            return (uetc.sigmoid((z-c)*s) * p_max).flatten()
        return f


# TOYDATA
def p_fire_i1(r: Union[np.ndarray, int], t: Union[np.ndarray, int]
             ) -> np.ndarray:
    """
    The probability of fire assuming there was an ignition.
    
    Args:
        r (int): rain level (0-5)
        t (int): temperature level (0-5)
    """
    r,t = np.atleast_1d(r,t)
    
    p = (t*2/10) - (r*2.5/10) # (temp on scale 0-1) - (rain on scale 0-1.25)
    p[p>1] = 1
    p[p<0] = 0
    return p

def sample(p_ignition: float, p_temp: Dict[int,float], p_rain: Dict[int,float], 
           n: int=1, add_to_p_fire: float=0.0, 
           seed=None) -> tuple:
    """
    Generates toydata examples.

    Args:
        p_ignition (scalar): Probability of an ignition source being onsite.
        p_temp, p_rain (dict): Mapping from temperature and rain level 
            respectively to the respective probability (dict[level] -> proba).
        n (int): Number of examples to sample.
    """
    rng = np.random.RandomState(seed)
    r = rng.choice(list(p_rain.keys()), p=list(p_rain.values()), size=n)
    t = rng.choice(list(p_temp.keys()), p=list(p_temp.values()), size=n)
    i = rng.choice([True,False],  p=[p_ignition, 1-p_ignition], size=n)
    p = np.round(p_fire_i1(r,t) + add_to_p_fire, 5)
    p[p>1] = 1
    f = rng.uniform(size=n) < p
    f = f*i
    return r,t,i,p,f

def generate_dataset(*args, **kwargs) -> pd.DataFrame:
    """
    Wrapper for `sample`. Returns samples as dataframe. All args passed to 
    `sample`.
    """
    r,t,i,p,f = sample(*args, **kwargs)
    return pd.DataFrame({"rain": r, "temp": t, "fire": f, 
                         "_ignition": i, "_p_fire_i1": p}) # _... hidden vars

def plot_proba_dict(proba_dict: Dict[int, float]):
    return plt.bar(x=proba_dict.keys(), height=proba_dict.values())

def plot(x_train, y_train, x_test=None, y_test=None, model=None, p_fun=None, 
         scenario: "Scenario"=None, plot_all_p: bool=False, plot_x_proba: bool=False, 
         hist_bins: int=0, pad: float=0, title_prefix: str=""):
    """
    #todo docstring Parameters bad
    
    Parameters
    ----------
    model :
        Model instance in sklearn fashion. Must have `predict_proba` method.
    p_fun : Callable
        Function that takes data X of shape (n_samples, n_features) and outputs 
        an array of true probabilities of shape (n_samples,). Ignored if 
        scenario is not None.
    hist_bins : int, optional, default 0
        Number of bins in between 0 and 1 to use to show histogram. 0 means 
        no histogram.
    pad : float, optional
        Expands x axis to left and right
    """
    if x_train.shape[1] > 1 and scenario is None:
        warnings.warn("X with more than one dim is not supported (unless for scenarios)")
    if scenario is not None and scenario.scenario > 1:
        raise NotImplementedError()
    
    x_min, x_max = x_train.min(), x_train.max()
    n = len(x_train)
    
    fig, ax = plt.subplots(1, figsize=(14,8))
    if scenario is None:
        ax.scatter(x=x_train, y=y_train, marker=".", c="steelblue", zorder=2, 
                   label="Train set")
        ax.set_xlabel("Precipitation")
    else:
        if scenario.scenario == 1:
            ax.scatter(x=x_train[:,0], y=y_train, marker=".", c="steelblue", 
                       zorder=2, label="Train set")
        
    if x_test is not None:
        ax.scatter(x=x_test, y=y_test + np.sign(y_test-.5)*.03, 
                   marker=".", c="slategray", zorder=2, label="Test set")
        
    if hist_bins > 0:
        bins      = np.linspace(x_min, x_max, hist_bins+1)
        bin_width = (x_max-x_min)/hist_bins
        bin_mids  = bins[:-1] + bin_width/2
        
        if scenario is None:
            D = pd.DataFrame({"x": x_train.flatten(), "y": y_train})
        else:
            D = pd.DataFrame({"x": x_train[:,0].flatten(), "y": y_train})

        D_binned = (
            D.assign(x_binned = lambda df: pd.cut(df.x, bins, labels=bin_mids))
            .groupby(["x_binned","y"])
            .size().rename("n"))
        D_binned = D_binned / D_binned.max()
        D_binned = D_binned.unstack()
        
        ax.bar(x=D_binned.index.values, height=D_binned[0], width=bin_width, 
               label="y = 0", alpha=.25, zorder=0)
        ax.bar(x=D_binned.index.values, height=D_binned[1], width=bin_width, 
               label="y = 1", alpha=.25, zorder=1)
        
    if scenario is not None:
        p_fun = scenario.p_all_fun
        X_grid, H_grid = scenario.make_grid(10000, pad=.05)
        
    if model is not None:
        if scenario is None:
            x_grid = np.arange(x_min-pad, x_max+pad, .01)[:,None]
            p_hats = model.predict_proba(x_grid)[:,1] # p_hat(y=1|x)
            ax.plot(x_grid, p_hats, c="red", zorder=5, label="Est. proba", lw=2)
        else:
            p_hats = model.predict_proba(X_grid)[:,1] # p_hat(y=1|x)
            order  = np.argsort(X_grid[:,0])
            # ax.scatter(X_grid[order,0], p_hats[order], color="red", zorder=5, 
            #            label=r"$\hat{p}(y~|~X)$")
            ax.plot(X_grid[order,0], p_hats[order], c="red", zorder=5, 
                    label=r"$\hat{p}(y~|~X)$", lw=2)
        
    if p_fun is not None:
        if scenario is None:
            x_grid  = np.arange(x_min-pad, x_max+pad, .01)[:,None]
            p_trues = p_fun(x_grid)
            ax.plot(x_grid, p_trues, c="k", linestyle="--", lw=1.5, zorder=6, 
                    label="True proba")
        else:
            if scenario.scenario == 1:
                p_trues = p_fun(H_grid)
                if plot_x_proba:
                    order  = np.argsort(X_grid[:,0])
                    ax.plot(X_grid[order,0], p_trues[order,2], c="k", alpha=.2, 
                            linestyle="-", lw=5, zorder=6, label="$p(y~|~X)$")
                ax.plot(H_grid[:,0], p_trues[:,2], c="k", linestyle="--", lw=1.5, 
                        zorder=7, label="$p(y~|~H)$")
                if plot_all_p:
                    ax.plot(H_grid[:,0], p_trues[:,0], c="seagreen", linestyle="--", 
                            lw=1.5, zorder=7, label="$p(f~|~H)$")
                    ax.plot(H_grid[:,0], p_trues[:,1], c="darkkhaki", linestyle="--", 
                            lw=1.5, zorder=7, label="$p(i~|~H)$")
            else:
                # p_trues, labels = [], []
                raise NotImplementedError()
    
    
    ratio_true = y_train.sum() / n
    ax.set_title(f"{title_prefix}$n = {n},~r = {ratio_true:.4f}$")
    ax.set_ylim((-.1, 1.1))
    ax.legend()
    
    return fig, ax

def plot_level_proba_heatmap2(p_temp: Dict[int,float], p_rain: Dict[int,float],
                              figsize=None):
    weather_prob_mat = np.full((len(p_rain), len(p_temp)), np.nan)
    fire_prob_mat    = np.full((len(p_rain), len(p_temp)), np.nan)
    for rlvl, rproba in p_rain.items():
        for tlvl, tproba in p_temp.items():
            weather_prob_mat[rlvl,tlvl] = rproba * tproba
            fire_prob_mat[rlvl,tlvl]    = p_fire_i1(rlvl,tlvl)

    assert np.isclose(np.sum(weather_prob_mat), 1), "probas do not sum to 1."
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    axs[0] = _make_heatmap_ax(fig, axs[0], weather_prob_mat, ax_title = 
        "$p(t,~r)$\nJoint probability of rain-temperature-combinations\n",
        xlabel = "\ntemperature level", ylabel = "rain level")
    axs[1] = _make_heatmap_ax(fig, axs[1], fire_prob_mat, ax_title = 
        "$p(f~|~t,~r,~i=1)$\nProbability of fire given an ignition source\n",
        xlabel = "\ntemperature level", ylabel = "rain level")
    return fig, axs

# ML / TOYDATA

def _add_train_val_test_accessors(mldata_class) -> "MLDataClass":
    """
    For usage as class decorator (class MLData)
    """
    for prop_grp, col_prop in [("X", "x_cols"), 
                               ("y", "y_col"), 
                               ("p", "p_col")]:
        for split in [None, "train", "val", "test"]:
            if split is None:
                prop_name = prop_grp # e.g. X
                getter = _make_getter(col_prop)
            else:
                prop_name = f"{prop_grp}_{split}" # e.g. X_train
                getter = _make_getter(col_prop, split)

            prop = property(fget=getter)
            setattr(mldata_class, prop_name, prop)
    return mldata_class

def _make_getter(col_prop, split=None):
    if split is None:
        def getter(self):
            return self.data_strat[
                    getattr(self, col_prop)
                ].to_numpy()
    else:
        def getter(self):
            return self.data_strat.loc[
                    getattr(self, f"{split}_sel"), 
                    getattr(self, col_prop)
                ].to_numpy()
    return getter

@_add_train_val_test_accessors
class MLData:
    """
    Args:
        stratify (str (opt)): Either "oversample" or "undersample" or None.
        train_ratio (float): The ratio of data that will be put into the 
            training set. Rest is used for test and train.
        val_ratio (float): The ratio of data that will be put into the 
            validation set. Rest is used for test and train.
    """
    def __init__(self, df: pd.DataFrame, x_cols: List[str], y_col: str, 
                 train_ratio: float=.5, val_ratio: float=.25, 
                 mini_size: int=2000, set_mean_tol = .01, 
                 p_col: Optional[str]=None, 
                 stratify: Optional[str]=None, sample_weights: bool=False,
                 seed: Optional=None):
        self._data = df
        self.x_cols = x_cols
        self.y_col  = y_col
        self.p_col  = p_col
        self.mini_size = mini_size
        self.n, self.xdim = self.data.shape[0], len(x_cols)
        self.stratify = stratify
        self.use_sample_weights = sample_weights
        
        if stratify is not None and sample_weights is True:
            raise ValueError("use either strat or sample_weights, not both")
        
        rng = np.random.RandomState(seed).randint
        max_seed = 2**32 - 1
        
        # stratify
        if stratify is None:
            self._strat = range(self.n)
        else:
            raise NotImplementedError("stratify not implemented")
            oversample  = {"oversample": True, "undersample": False}[stratify]
            self._strat = uml.stratified_idx(
                self.data[self.y_col], oversample=oversample, seed=rng(max_seed))
        self.n_strat = self.data_strat.shape[0]

        # generate splits
        train_sel_bool = uetc.rnd_bool(
            n = self.n_strat, n_true=int(self.n_strat * train_ratio), 
            seed=rng(max_seed)) # bool mask for all_ids
        all_ids        = np.arange(self.n_strat)  # array of ints
        self.train_sel = all_ids[ train_sel_bool] # array of ints
        val_test_ids   = all_ids[~train_sel_bool] # array of ints
        
        test_ratio    = 1-val_ratio-train_ratio
        test_ratio_of_val_test = test_ratio / (1-train_ratio)
        test_sel_bool = uetc.rnd_bool(
            n = len(val_test_ids), 
            n_true = int(len(val_test_ids) * test_ratio_of_val_test), 
            seed = rng(max_seed)) # bool mask for val_test_ids
        self.test_sel = val_test_ids[ test_sel_bool] # array of ints
        self.val_sel  = val_test_ids[~test_sel_bool] # array of ints

        # sample weights
        self._set_sample_weights()
        
        # properties such as self.X_train are generated by decorator
        # @add_train_val_test_accessors

        self._print_overview()
        return
        
    @property
    def data_strat(self) -> pd.DataFrame:
        return self.data.loc[self._strat]
    
    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy()

    def _print_overview(self) -> None:
        # print split sizes
        print("split-sizes\n"
            f"    train: {len(self.y_train)}, val: {len(self.y_val)}, "
            f"test: {len(self.y_test)}")

        # print class-ratios per split
        mean_or_none = lambda x: x.mean().round(4) if len(x) > 0 else None
        y_mean_train = mean_or_none(self.y_train) # prop generated by decorator
        y_mean_val   = mean_or_none(self.y_val)
        y_mean_test  = mean_or_none(self.y_test)
        all_means = [m for m in [y_mean_train, y_mean_val, y_mean_test] if m]
        max_diff  = max(*all_means) - min(*all_means)
        print("class-ratios (mean of y)\n"
            f"    train: {y_mean_train}, val: {y_mean_val}, test: {y_mean_test}\n"
            f"    max-diff: {max_diff:.4f}")

        # sample weights
        print(f"sample weights computed: {self.use_sample_weights}")
        if self.use_sample_weights:
            max_w = max(self.sample_weights_train)
            min_w = min(self.sample_weights_train)
            print(f"   in train split: max weight: {max_w}, min weight: {min_w}")

        # stratification
        use_strat = self.stratify is not None
        print(f"classes balanced by stratification: {use_strat}")
        if use_strat:
            print(f"    strategy: {self.stratify}")
            print(f"    number of rows originally: {self.data.shape[0]}")
            print(f"    number of rows stratified: {self.data_strat.shape[0]}")


    def _set_sample_weights(self) -> None:
        self._set_sample_weights_for_split("train", self.train_sel)
        self._set_sample_weights_for_split("val", self.val_sel)
        self._set_sample_weights_for_split("test", self.test_sel)


    def _set_sample_weights_for_split(self, split_name: str, split_sel: List[int]
                                     ) -> None:
        """
        Determines the sample weights for all examples in a split by using the
        inverse priors "rule" (equivalent to `1-class_ratio`), and sets the
        respective attributes (e.g. sample_weights_train and class_weights_train).
        
        sample_weights_{split_name} (array-like of float): Array of sample 
            weights of the length of the split, or None.
        class_weights_{split_name} (Dict[Any, float]): Dictionary that maps 
            class-label -> weight for each corresponding example, or None.
        """
        if self.use_sample_weights:
            split_df = self.data.loc[split_sel]
            class_ratios  = split_df.groupby(self.y_col).size() / len(split_df)
            class_weights = pd.Series(uml.inverse_priors(class_ratios), 
                                    index=class_ratios.index, name="w")
            sample_weights = split_df.merge(
                class_weights.to_frame().reset_index(), on=self.y_col)["w"]
            class_weight_dict = {
                class_label: weight for class_label, weight 
                in zip(class_weights.index.values, class_weights.to_list())}
        else:
            sample_weights = None
            class_weight_dict = None
        
        setattr(self, "sample_weights_"+split_name, sample_weights)
        setattr(self, "class_weights_"+split_name, class_weight_dict)


from sklearn.gaussian_process import GaussianProcessClassifier
class GPCBatched(GaussianProcessClassifier):
    def __init__(self, batch_size: int=1000, show_progress: bool=True, 
                 *args, **kwargs):
        self.batch_size    = batch_size
        self.show_progress = show_progress
        super().__init__(*args, **kwargs)
        
    def predict_proba(self, X):
        P = []
        if self.show_progress:
            n_batches = int(np.ceil(X.shape[0]/self.batch_size))
            print(self.batch_size, X.shape[0], n_batches, "\n\n")
            progr = uetc.ProgressDisplay(n_batches)
            progr.start_timer()
        for batch in self._iter_batches(X):
            P.append(super(GPCBatched, self).predict_proba(batch))
            if self.show_progress:
                progr.update_and_print()
        if self.show_progress:
            progr.stop()
        P = np.vstack(P)
        return P
    
    def predict(self, X):
        y = []
        if self.show_progress:
            n_batches = int(np.ceil(X.shape[0]/self.batch_size))
            progr = uetc.ProgressDisplay(n_batches)
            progr.start_timer()
        for batch in self._iter_batches(X):
            y.append(super(GPCBatched, self).predict(batch))
            if self.show_progress:
                progr.update_and_print()
        if self.show_progress:
            progr.stop()
        y = np.r_[y]
        return y
    
    def _iter_batches(self, X):
        n = X.shape[0]
        b = self.batch_size
        for i in range(0, n, b):
            yield X[i:i+b]


class ProbaCorridor(nn.Module):
    """
    Args:
        leaky (bool): If True, LeakyHardtanh will be used as activation, 
            otherwise nn.Hardtanh. Defaults to False.
        slope (float): Slope of leaky parts in LeakyHardtanh, ignored if 
            leaky is False.
    """
    def __init__(self, leaky: bool=False, slope: float=.1):
        #todo max_lower_bound, min_upper_bound (for init)
        super(ProbaCorridor, self).__init__()
        self.linear = nn.Linear(1, 1, bias=True)
        
        # init weight and bias such that wx+b \in [0,1] for any weight and 
        # bias, given x \in [0,1]
        b_init = torch.rand(self.linear.bias.shape) / 4 # torch.rand is uniform in [0,1]
        w_init = torch.rand(self.linear.weight.shape) * (1-b_init.item())
        self.linear.bias   = nn.Parameter(b_init)
        self.linear.weight = nn.Parameter(w_init)
        
        if leaky:
            self.act = LeakyHardtanh(id_start=0, id_stop=1, slope=slope, 
                                     inplace=False)
        else:
            # assure that the output is always within [0,1], even if weights 
            # get crazy during training
            self.act = nn.Hardtanh(0, 1)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        return x
        
    @property
    def corridor(self) -> Tuple[float,float]:
        w = self.linear.weight.item()
        b = self.linear.bias.item()
        p_min = w*0 + b 
        p_max = w*1 + b
        p_min = self.act(torch.Tensor([p_min])).item()
        p_max = self.act(torch.Tensor([p_max])).item()
        return p_min, p_max
    
    @property
    def bias(self):
        return self.linear.bias
    
    @property
    def weight(self):
        return self.linear.weight
        

class FeatureFusionNet(nn.Module):
    def __init__(self, n_f: int=1, n_i: int=0, pc_kwargs={}):
        super(FeatureFusionNet, self).__init__()
        # flammability feature branch
        self.n_f   = n_f
        self.f_fc  = nn.Linear(n_f, 1, bias=True)
        self.f_act = nn.Sigmoid()
        
        # ignition src feature branch
        self.n_i   = n_i
        # if n_i is 0, always pass 1 as input to the i-branch
        if n_i > 0:
            self.i_fc  = nn.Linear(n_i, 1, bias=True)
            self.i_act = nn.Sigmoid()
        self.i_pc = ProbaCorridor(**pc_kwargs)
        
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
        if self.n_i > 0:
            x_i = self.i_fc(x_i)
            x_i = self.i_act(x_i)
        else:
            if x_i is not None:
                raise ValueError("x_i must be None, since n_i is 0")
            x_i = torch.Tensor([[1]])
        x_i = self.i_pc(x_i)
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


# toydata
def identify_scenario(settings: List[Dict[str,Any]]) -> int:
    if len(settings) > 1: # 7 or 8
        if "h4" not in settings[0]:
            return 7
        else:
            return 8
    elif "var1" not in settings[0]:
        return 1
    elif "var01" not in settings[0]:
        return 2
    elif "var2" not in settings[0]:
        return 3
    elif "var12" not in settings[0]:
        return 4
    elif "p_spread" not in settings[0]:
        return 5
    else:
        return 6




class ProcessList:
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
        print("-"*20)

        # do not consider neighboring fires as they are not distorted anyway
        no_neighboring_fire = H[:,3] == 0
        H = H[no_neighboring_fire]
        pids = np.array(pids)[no_neighboring_fire]

        hj     = H[:,j]
        p_true = self.compute_XZPy(H, pids, add_noise=False)[2][2]

        # calculate the absolute threshold values
        abs_maxae_rt_thresh = np.max(p_true) * self.maxae_rt_thresh
        abs_maxae_lr_thresh = np.max(p_true) * self.maxae_lr_thresh
        abs_mae_lr_thresh   = np.max(p_true) * self.mae_lr_thresh
        
        # check if hj can be recovered by a non-linear learner, i.e. regr. tree
        regtree   = DecisionTreeRegressor().fit(hj_distorted, hj)
        hj_est_rt = regtree.predict(hj_distorted)
        # compute p for H and for H_est (H with hj replaced by hj_est_rt)
        H_est_rt      = H.copy()
        H_est_rt[:,j] = hj_est_rt
        p_est_rt      = self.compute_XZPy(H_est_rt, pids, add_noise=False)[2][2]
        maxabserr_rt  = uml.maximum_absolute_error(p_true, p_est_rt)
        print(f"MaxAErt  {maxabserr_rt / np.max(p_true):.8f} > {self.maxae_rt_thresh}")
        if maxabserr_rt > abs_maxae_rt_thresh: 
            return False

        # check if hj can be recovered by a non-linear learner, i.e. regr. tree
        linreg    = LinearRegression().fit(hj_distorted, hj)
        hj_est_lr = linreg.predict(hj_distorted)
        # compute p for H and for H_est (H with hj replaced by hj_est_rt)
        H_est_lr      = H.copy()
        H_est_lr[:,j] = hj_est_lr
        p_est_lr      = self.compute_XZPy(H_est_lr, pids, add_noise=False)[2][2]
        maxabserr_lr  = uml.maximum_absolute_error(p_true, p_est_lr)
        mae_lr        = skm.mean_absolute_error(p_true, p_est_lr)
        print(f"MaxAElin {maxabserr_lr / np.max(p_true):.8f} < {self.maxae_lr_thresh}")
        print(f"MAElin   {mae_lr / np.max(p_true):.8f} < {self.mae_lr_thresh}")
        if (maxabserr_lr < abs_maxae_lr_thresh) or (mae_lr < abs_mae_lr_thresh):
            print("lr out")
            return False
        
        # if arrived here, the distortion is suitable
        print(" "*20 + "success")
        return True


class Times(nn.Module):
    def __init__(self, m: float):
        super().__init__()
        self.m = m
    
    def forward(self, x):
        return x*self.m


class Plus(nn.Module):
    def __init__(self, n: float):
        super().__init__()
        self.n = n
    
    def forward(self, x):
        return x + self.n