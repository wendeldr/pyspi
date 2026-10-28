import jpype as jp
import numpy as np
from pyspi import utils
import copy
import logging

from pyspi.base import Undirected, Directed, Unsigned, parse_univariate, parse_bivariate, parse_multivariate
from pyspi._gaussian import gaussian_result, gaussian_diagnostics, gaussian_marginal_entropy
from pyspi._moments import moment_workspace
from pyspi._information_history import history
from pyspi._naming import compose, parameter, word


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


class JIDTBase(Unsigned):

    _native_gaussian_measure = None
    _estimators = ("gaussian", "kernel", "kozachenko")

    @classmethod
    def requires_java(cls, estimator="gaussian", **kwargs):
        return not (estimator == "gaussian" and cls._native_gaussian_measure)

    def issigned(self):
        return "signed" in self.labels

    @parse_multivariate
    def multivariate(self, data):
        if self._estimator == "gaussian" and self._native_gaussian_measure:
            return gaussian_result(data, self._native_gaussian_measure, **getattr(self, "_gaussian_parameters", {}))
        return super().multivariate(data)

    @parse_multivariate
    def diagnostics(self, data):
        if self._estimator == "gaussian" and self._native_gaussian_measure:
            return gaussian_diagnostics(data, self._native_gaussian_measure, **getattr(self, "_gaussian_parameters", {}))
        return {}

    # List of (currently) modifiable parameters
    _NNK_PROP_NAME = "k"
    _AUTO_EMBED_METHOD_PROP_NAME = "AUTO_EMBED_METHOD"
    _DYN_CORR_EXCL_PROP_NAME = "DYN_CORR_EXCL"
    _KERNEL_WIDTH_PROP_NAME = "KERNEL_WIDTH"
    _K_HISTORY_PROP_NAME = "k_HISTORY"
    _K_TAU_PROP_NAME = "k_TAU"
    _L_HISTORY_PROP_NAME = "l_HISTORY"
    _L_TAU_PROP_NAME = "l_TAU"
    _K_SEARCH_MAX_PROP_NAME = "AUTO_EMBED_K_SEARCH_MAX"
    _TAU_SEARCH_MAX_PROP_NAME = "AUTO_EMBED_TAU_SEARCH_MAX"
    _BIAS_CORRECTION = "BIAS_CORRECTION"
    _NORMALISE = "NORMALISE"
    _SEED = "NOISE_SEED"

    _base_class = jp.JPackage("infodynamics.measures.continuous")

    def __init__(
        self, estimator="gaussian", kernel_width=0.5, prop_k=4, dyn_corr_excl=None
    ):
        if estimator not in self._estimators:
            raise ValueError(f"{type(self).__name__} supports estimators {self._estimators}.")
        self._estimator = estimator
        self._kernel_width = kernel_width
        self._prop_k = prop_k
        self._dyn_corr_excl = dyn_corr_excl
        self._entropy_calc = self._getcalc("entropy")

        self._measure_name = self.identifier
        self._name_variants, self._name_parameters = [word(estimator)], []
        if estimator == "kraskov":
            self._name_parameters.append(parameter("neighbours", prop_k))
            self.labels = self.labels + ["nonlinear"]
        elif estimator == "kernel":
            self._name_parameters.append(parameter("width", kernel_width, 1))
            self.labels = self.labels + ["nonlinear"]
        elif estimator == "symbolic":
            if not isinstance(self, TransferEntropy):
                raise NotImplementedError(
                    "Symbolic estimator is only available for transfer entropy."
                )
            self.labels = self.labels + ["symbolic", "nonlinear"]
            self._dyn_corr_excl = None
        else:
            self.labels = self.labels + ["linear"]
            self._dyn_corr_excl = None

        # Dynamic correlation exclusion is a Theiler window: chosen per pair
        # ("AUTO") or a fixed number of samples.
        if self._dyn_corr_excl == "AUTO":
            self._name_variants.append("theilerauto")
        elif self._dyn_corr_excl:
            self._name_parameters.append(parameter("theiler", int(self._dyn_corr_excl)))
        self._rename()

    def _rename(self, variants=(), parameters=()):
        self._name_variants.extend(variants)
        self._name_parameters.extend(parameters)
        self.identifier = compose(self._measure_name, *self._name_variants, *self._name_parameters)

    def __getstate__(self):
        state = dict(self.__dict__)

        unserializable_objects = ["_entropy_calc", "_calc"]

        for k in unserializable_objects:
            if k in state.keys():
                del state[k]

        return state

    def __setstate__(self, state):
        """Re-initialise the calculator"""
        # Re-initialise
        self.__dict__.update(state)
        self._entropy_calc = self._getcalc("entropy")

    def __deepcopy__(self, memo):
        newone = type(self).__new__(type(self))
        memo[id(self)] = newone
        newone.__setstate__(copy.deepcopy(self.__getstate__(), memo))
        return newone

    def _setup(self, calc):
        if self._estimator == "kernel":
            calc.setProperty(self._KERNEL_WIDTH_PROP_NAME, str(self._kernel_width))
        elif self._estimator == "kraskov":
            calc.setProperty(self._NNK_PROP_NAME, str(self._prop_k))

        calc.setProperty(self._BIAS_CORRECTION, "false")
        calc.setProperty(self._SEED, "42")

        return calc

    def _getkey(self):
        if self._estimator == "kernel":
            return (self._estimator, self._kernel_width)
        elif self._estimator == "kraskov":
            return (self._estimator, self._prop_k)
        else:
            return (self._estimator,)

    def _getcalc(self, measure):
        if not self.requires_java(estimator=self._estimator):
            return None
        if measure == "entropy":
            if self._estimator == "kernel":
                calc = self._base_class.kernel.EntropyCalculatorMultiVariateKernel()
            elif self._estimator == "kozachenko":
                calc = (
                    self._base_class.kozachenko.EntropyCalculatorMultiVariateKozachenko()
                )
            else:
                # KSG MI/TE own their estimator; no separate entropy calculator
                # is used. Gaussian and symbolic construction returned above.
                return None
        elif measure == "MutualInfo":
            if self._estimator == "kernel":
                calc = self._base_class.kernel.MutualInfoCalculatorMultiVariateKernel()
            elif self._estimator == "kraskov":
                calc = (
                    self._base_class.kraskov.MutualInfoCalculatorMultiVariateKraskov1()
                )
            else:
                raise ValueError(f"No Java MI estimator for {self._estimator}.")
        elif measure == "TransferEntropy":
            if self._estimator == "kernel":
                calc = self._base_class.kernel.TransferEntropyCalculatorKernel()
            elif self._estimator == "kraskov":
                calc = self._base_class.kraskov.TransferEntropyCalculatorKraskov()
            else:
                raise ValueError(f"No Java TE estimator for {self._estimator}.")
        else:
            raise TypeError(f"Unknown measure: {measure}")

        return self._setup(calc)

    # No Theiler window yet (can it be done?)
    @parse_univariate
    def _compute_entropy(self, data, i=None):
        if self._estimator == "gaussian" and self._native_gaussian_measure:
            return gaussian_marginal_entropy(data)[i]
        work = moment_workspace(data)
        return work.arrays.get(("information_entropy", self._getkey(), i),
                               lambda: np.asarray(self._entropy_of(work.source[i, :, None]))).item()

    # No Theiler window is available in the JIDT estimator
    @parse_bivariate
    def _compute_joint_entropy(self, data, i, j):
        if self._estimator == "gaussian" and self._native_gaussian_measure:
            return gaussian_result(data, "je")[i, j]
        work = moment_workspace(data)
        return work.arrays.get(("information_joint_entropy", self._getkey(), min(i,j), max(i,j)),
                               lambda: np.asarray(self._entropy_of(work.source[[i,j]].T))).item()

    def _entropy_of(self, observations):
        observations = np.asarray(observations, dtype=float)
        if not observations.shape[1]:
            return 0.
        self._entropy_calc.initialise(observations.shape[1])
        self._entropy_calc.setObservations(jp.JArray.of(observations))
        return float(self._entropy_calc.computeAverageLocalOfObservations())

    # No Theiler window is available in the JIDT estimator
    def _compute_conditional_entropy(self, X, Y):
        return self._entropy_of(np.concatenate([X, Y], axis=1))-self._entropy_of(Y)

    def _target_conditional_entropy(self, data, channel, lags):
        """Share aligned target-only terms between DI and stochastic interaction."""
        work = moment_workspace(data)
        lags = tuple(lags)
        start = max((0, *lags))
        def compute():
            past = history(work, start, lags)[channel]
            future = work.source[channel, start:, None]
            return np.asarray(self._compute_conditional_entropy(future, past))
        return work.arrays.get(("information_target_conditional_entropy", self._getkey(), lags, channel), compute).item()

    def _set_theiler_window(self, data, i, j):
        if self._dyn_corr_excl == "AUTO":
            work = moment_workspace(data)
            def compute():
                # Preserve the existing Bartlett definition and integer
                # truncation; compute each channel's ACF once. The bounded
                # workspace invalidates this table when Data changes.
                acfs = np.array([utils.acf(row) for row in work.source])
                values = 2 * (acfs @ acfs.T)
                np.fill_diagonal(values, -1.)
                return values
            theiler = work.arrays.get(("information_auto_theiler",), compute)
            self._calc.setProperty(
                self._DYN_CORR_EXCL_PROP_NAME, str(int(theiler[i, j]))
            )
        elif self._dyn_corr_excl is not None:
            self._calc.setProperty(
                self._DYN_CORR_EXCL_PROP_NAME, str(int(self._dyn_corr_excl))
            )


class JointEntropy(JIDTBase, Undirected):

    name = "Joint entropy"
    identifier = "je"
    labels = ["signed", "infotheory", "unordered", "undirected"]
    _native_gaussian_measure = "je"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return self._compute_joint_entropy(data, i=i, j=j)


class ConditionalEntropy(JIDTBase, Directed):

    name = "Conditional entropy"
    identifier = "ce"
    labels = ["signed", "infotheory", "unordered", "directed"]
    _native_gaussian_measure = "ce"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        if self._estimator == "gaussian":
            return gaussian_result(data, "ce")[i, j]
        return self._compute_joint_entropy(data, i=i, j=j) - self._compute_entropy(
            data, i=i
        )


class MutualInfo(JIDTBase, Undirected):
    name = "Mutual information"
    identifier = "mi"
    labels = ["unsigned", "infotheory", "unordered", "undirected"]
    _native_gaussian_measure = "mi"
    _estimators = ("gaussian", "kernel", "kraskov")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calc = self._getcalc("MutualInfo")

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.update(state)
        self._calc = self._getcalc("MutualInfo")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        """Compute mutual information between Y and X"""
        if self._estimator == "gaussian":
            return gaussian_result(data, "mi")[i, j]
        self._set_theiler_window(data, i, j)
        self._calc.initialise(1, 1)

        try:
            src, targ = data.to_numpy(squeeze=True)[[i, j]]
            self._calc.setObservations(
                jp.JArray(jp.JDouble)(src), jp.JArray(jp.JDouble)(targ)
            )
            return self._calc.computeAverageLocalOfObservations()
        except:
            logging.warning(
                "MI calcs failed. Maybe check input data for Cholesky factorisation?"
            )
            return np.nan


class TimeLaggedMutualInfo(JIDTBase, Directed):
    name = "Time-lagged mutual information"
    identifier = "tlmi"
    labels = ["unsigned", "infotheory", "temporal", "directed"]
    _native_gaussian_measure = "tlmi"
    _estimators = ("gaussian", "kernel", "kraskov")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calc = self._getcalc("MutualInfo")

    def __setstate__(self, state):
        """Re-initialise the calculator"""
        super().__setstate__(state)
        self.__dict__.update(state)
        self._calc = self._getcalc("MutualInfo")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        if self._estimator == "gaussian":
            return gaussian_result(data, "tlmi")[i, j]
        self._set_theiler_window(data, i, j)
        self._calc.initialise(1, 1)
        try:
            src, targ = data.to_numpy(squeeze=True)[[i, j]]
            src = src[:-1]
            targ = targ[1:]
            self._calc.setObservations(
                jp.JArray(jp.JDouble, 1)(src), jp.JArray(jp.JDouble, 1)(targ)
            )
            return self._calc.computeAverageLocalOfObservations()
        except:
            logging.warning(
                "Time-lagged MI calcs failed. Maybe check input data for Cholesky factorisation?"
            )
            return np.nan


class TransferEntropy(JIDTBase, Directed):

    name = "Transfer entropy"
    identifier = "te"
    labels = ["unsigned", "embedding", "infotheory", "temporal", "directed"]
    _native_gaussian_measure = "te"
    _estimators = ("gaussian", "kernel", "kraskov", "symbolic")

    @classmethod
    def requires_java(cls, estimator="gaussian", **kwargs):
        return estimator not in ("gaussian", "symbolic")

    def __init__(
        self,
        auto_embed_method=None,
        k_search_max=None,
        tau_search_max=None,
        k_history=1,
        k_tau=1,
        l_history=1,
        l_tau=1,
        pattern_length=None,
        pattern_spacing=None,
        prediction_shift=None,
        **kwargs,
    ):

        if "estimator" not in kwargs.keys() or kwargs["estimator"] == "gaussian":
            self.identifier = "gc"
        for name, value in (("k_history", k_history), ("k_tau", k_tau),
                            ("l_history", l_history), ("l_tau", l_tau)):
            _positive_integer(value, name)
        if auto_embed_method is not None:
            _positive_integer(k_search_max, "k_search_max")
            if kwargs.get("estimator", "gaussian") != "kernel":
                _positive_integer(tau_search_max, "tau_search_max")
            if kwargs.get("estimator", "gaussian") == "gaussian" and auto_embed_method != "MAX_CORR_AIS":
                raise ValueError("Native Gaussian automatic embedding supports MAX_CORR_AIS.")
        self._gaussian_parameters = dict(auto_embed_method=auto_embed_method, k_search_max=k_search_max,
                                        tau_search_max=tau_search_max, k_history=k_history, k_tau=k_tau,
                                        l_history=l_history, l_tau=l_tau)
        super().__init__(**kwargs)
        self._calc = self._getcalc("TransferEntropy")
        if self._estimator == "symbolic":
            if auto_embed_method is not None or (k_history, k_tau, l_history, l_tau) != (1, 1, 1, 1):
                raise ValueError("Symbolic TE uses pattern_length, pattern_spacing and prediction_shift, not Gaussian histories or auto embedding.")
            length = 3 if pattern_length is None else pattern_length
            spacing = 1 if pattern_spacing is None else pattern_spacing
            shift = 1 if prediction_shift is None else prediction_shift
            for name, value in (("pattern_length", length), ("pattern_spacing", spacing), ("prediction_shift", shift)):
                _positive_integer(value, name)
            if not 2 <= length <= 20:
                raise ValueError("pattern_length must be 2..20; length one contains no ordinal information.")
            self._symbolic_parameters = dict(pattern_length=length, pattern_spacing=spacing, prediction_shift=shift)
            self._rename(parameters=(parameter("patternlength", length), parameter("spacing", spacing),
                                     parameter("shift", shift)))
            return
        if any(value is not None for value in (pattern_length, pattern_spacing, prediction_shift)):
            raise ValueError("Ordinal pattern parameters apply only to estimator='symbolic'.")

        # k/l are the target/source history lengths, k_tau/l_tau their delays.
        if auto_embed_method is not None:
            searched = [parameter("maxtargethistory", k_search_max)]
            if self._estimator != "kernel":
                searched.append(parameter("maxtargetdelay", tau_search_max))
            self._rename(("autoembed", word(auto_embed_method)), searched)
        elif self._estimator != "kernel":
            self._rename(parameters=(parameter("targethistory", k_history), parameter("targetdelay", k_tau),
                                     parameter("sourcehistory", l_history), parameter("sourcedelay", l_tau)))
        else:
            self._rename(parameters=(parameter("targethistory", k_history),))
        self._configure_calc()

    @parse_multivariate
    def multivariate(self, data):
        if self._estimator == "symbolic":
            from pyspi._symbolic import symbolic_result
            return symbolic_result(data, **self._symbolic_parameters)
        return super().multivariate(data)

    @parse_multivariate
    def diagnostics(self, data):
        if self._estimator == "symbolic":
            from pyspi._symbolic import symbolic_diagnostics
            return symbolic_diagnostics(data, **self._symbolic_parameters)
        return super().diagnostics(data)

    def _configure_calc(self):
        if self._calc is None:
            return
        parameters = self._gaussian_parameters
        if parameters["auto_embed_method"] is not None:
            settings = {self._AUTO_EMBED_METHOD_PROP_NAME: parameters["auto_embed_method"],
                        self._K_SEARCH_MAX_PROP_NAME: parameters["k_search_max"]}
            if self._estimator != "kernel":
                settings[self._TAU_SEARCH_MAX_PROP_NAME] = parameters["tau_search_max"]
        else:
            settings = {self._K_HISTORY_PROP_NAME: parameters["k_history"]}
            if self._estimator != "kernel":
                settings.update({self._K_TAU_PROP_NAME: parameters["k_tau"],
                                 self._L_HISTORY_PROP_NAME: parameters["l_history"],
                                 self._L_TAU_PROP_NAME: parameters["l_tau"]})
        for name, value in settings.items():
            self._calc.setProperty(name, str(value))

    def __setstate__(self, state):
        """Re-initialise the calculator"""
        # Re-initialise
        super().__setstate__(state)
        self.__dict__.update(state)
        self._calc = self._getcalc("TransferEntropy")
        self._configure_calc()

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        """
        Compute transfer entropy from i->j
        """
        if self._estimator == "gaussian":
            return gaussian_result(data, "te", **self._gaussian_parameters)[i, j]
        if self._estimator == "symbolic":
            from pyspi._symbolic import symbolic_result
            return symbolic_result(data, **self._symbolic_parameters)[i, j]
        self._set_theiler_window(data, i, j)
        self._calc.initialise()
        src, targ = data.to_numpy(squeeze=True)[[i, j]]
        try:
            self._calc.setObservations(
                jp.JArray(jp.JDouble, 1)(src), jp.JArray(jp.JDouble, 1)(targ)
            )
            return self._calc.computeAverageLocalOfObservations()
        except Exception as err:
            logging.warning(f"TE calcs failed: {err}.")
            return np.nan


class CrossmapEntropy(JIDTBase, Directed):

    name = "Source-history conditional entropy"
    identifier = "xme"
    labels = ["signed", "infotheory", "temporal", "directed"]
    _native_gaussian_measure = "xme"

    def __init__(self, history_length=10, **kwargs):
        _positive_integer(history_length, "history_length")
        super().__init__(**kwargs)
        self._rename(parameters=(parameter("history", history_length),))
        self._history_length = history_length
        self._gaussian_parameters = dict(history_length=history_length)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        if self._estimator == "gaussian":
            return gaussian_result(data, "xme", **self._gaussian_parameters)[i, j]
        k = self._history_length
        work = moment_workspace(data)
        past = history(work, k, range(1, k+1))[i]
        # Keep XME's existing source-first column order for the retained
        # estimators' seeded noise; the corrected lag set now has exactly k.
        joint = np.column_stack([past, work.source[j, k:]])
        conditioning = work.arrays.get(("information_source_history_entropy", self._getkey(), k, i),
                                      lambda: np.asarray(self._entropy_of(past))).item()
        return self._entropy_of(joint)-conditioning


class CausalEntropy(JIDTBase, Directed):

    name = "Causally conditioned entropy"
    identifier = "cce"
    labels = ["signed", "infotheory", "temporal", "directed"]
    _native_gaussian_measure = "cce"

    def __init__(self, n=5, **kwargs):
        _positive_integer(n, "n")
        super().__init__(**kwargs)
        self._n = n
        self._gaussian_parameters = dict(n=n)
        # n is the number of causally conditioned time steps.
        self._rename(parameters=(parameter("horizon", n),))

    def _causal_entropy(self, data, i, j):
        work = moment_workspace(data)
        def compute():
            value = 0.
            for m in range(1, self._n+1):
                past = history(work, m-1, range(1, m))[j]
                source = history(work, m-1, range(m))[i]
                future = work.source[j, m-1:, None]
                value += self._compute_conditional_entropy(future, np.column_stack([past, source]))
            return np.asarray(value)
        return work.arrays.get(("information_causal_entropy", self._getkey(), self._n, i, j), compute).item()

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        if self._estimator == "gaussian":
            return gaussian_result(data, "cce", **self._gaussian_parameters)[i, j]
        return self._causal_entropy(data, i, j)


class DirectedInfo(CausalEntropy, Directed):

    name = "Directed information"
    identifier = "di"
    labels = ["unsigned", "infotheory", "temporal", "directed"]
    _native_gaussian_measure = "di"

    def __init__(self, n=5, **kwargs):
        super().__init__(n=n, **kwargs)

    def _target_entropy_sum(self, data, j):
        work = moment_workspace(data)
        def compute():
            value = 0.
            for m in range(1, self._n+1):
                value += self._target_conditional_entropy(data, j, range(1, m))
            return np.asarray(value)
        return work.arrays.get(("information_target_entropy_sum", self._getkey(), self._n, j), compute).item()

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        """Compute directed information from i to j"""

        if self._estimator == "gaussian":
            return gaussian_result(data, "di", **self._gaussian_parameters)[i, j]

        return self._target_entropy_sum(data, j)-self._causal_entropy(data, i, j)


class StochasticInteraction(JIDTBase, Undirected):

    name = "Stochastic interaction"
    identifier = "si"
    labels = ["unsigned", "infotheory", "temporal", "undirected"]
    _native_gaussian_measure = "si"

    def __init__(self, delay=1, **kwargs):
        _positive_integer(delay, "delay")
        super().__init__(**kwargs)
        self._delay = delay
        self._rename(parameters=(parameter("delay", delay),))
        self._gaussian_parameters = dict(delay=delay)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        if self._estimator == "gaussian":
            return gaussian_result(data, "si", **self._gaussian_parameters)[i, j]
        x, y = data.to_numpy()[[i, j]]
        xy = np.concatenate([x, y], axis=1)
        tau = self._delay

        H_joint = self._compute_conditional_entropy(xy[tau:], xy[:-tau])
        H_src = self._target_conditional_entropy(data, i, (tau,))
        H_targ = self._target_conditional_entropy(data, j, (tau,))

        return H_src + H_targ - H_joint


class IntegratedInformation(Undirected, Unsigned):

    name = "Integrated information"
    identifier = "phi"
    labels = ["linear", "unsigned", "infotheory", "temporal", "undirected"]

    def __init__(self, phitype="star", delay=1, normalization=0):
        self._phitype = phitype
        self._delay = delay
        self._normalization = normalization
        # phi_native normalizes only when normalization == 1.
        self.identifier = compose(self.identifier, word(phitype), *(["normalized"] if normalization == 1 else []),
                                  parameter("delay", delay))

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        """
        Compute integrated information using a native Python implementation.
        """
        try:
            from pyspi.lib.phi_native import phi_comp

            # Prepare partition for bivariate analysis
            P = np.array([1, 2])
            X = data.to_numpy(squeeze=True)[[i, j]]

            # Set up parameters for native implementation
            params = {"tau": self._delay}
            options = {
                "type_of_phi": self._phitype,
                "type_of_dist": "Gauss",
                "normalization": self._normalization
            }

            return phi_comp(X, P, params, options)

        except Exception as e:
            logging.error(f"Integrated information computation failed: {e}")
            return np.nan
