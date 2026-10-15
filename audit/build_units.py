"""Build ``audit/units.yaml``, the calculation-unit catalogue of the audit.

A *unit* is one calculation pyspi performs at the level it is specified:
a statistics class together with the estimator/method/hyperparameter choice
that selects a distinct code path. Variants that only relabel the same
computation (``squared``, ``mean``/``max`` over a cached array, frequency band
selection of a cached spectrum) are listed under the parent unit's
``identifiers`` instead of getting their own row.

The catalogue is derived from the live package the same way the regression
harness does (``tests/regression/harness.catalogue``), so every identifier the
Calculator registers from ``pyspi/config.yaml`` and from
``tests/regression/extra_config.yaml`` is assigned to exactly one unit.
Infrastructure units (preprocessing, estimators shared by several classes)
are appended by hand at the end of this file.

Run with the project venv::

    .venv/bin/python audit/build_units.py            # writes audit/units.yaml
    .venv/bin/python audit/build_units.py --check    # exits non-zero if the file is stale
"""
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import yaml

AUDIT_DIR = Path(__file__).resolve().parent
REPO = AUDIT_DIR.parent
sys.path.insert(0, str(REPO / "tests"))

from regression import harness  # noqa: E402

EXTRA_CONFIG = REPO / "tests" / "regression" / "extra_config.yaml"
RETAINED_FILE = AUDIT_DIR / "env" / "retained_metrics.txt"

# Classes that override ``multivariate`` (one call produces the whole matrix).
MULTIVARIATE_CLASSES = {
    "Covariance", "Precision", "PairwiseDistance", "ConvergentCrossMapping",
    "PowerEnvelopeCorrelation", "InterDependenceScore",
    "CoherenceMagnitude", "CoherencePhase", "ImaginaryCoherence",
    "PhaseLockingValue", "PhaseLagIndex", "WeightedPhaseLagIndex",
    "DebiasedSquaredPhaseLagIndex", "DebiasedSquaredWeightedPhaseLagIndex",
    "PairwisePhaseConsistency", "PhaseSlopeIndex", "GroupDelay",
    "SpectralGrangerCausality", "PairwisePhaseConsistency",
    "weighted_PhaseLagIndex", "debiased_weighted_squared_PhaseLagIndex",
}
# Bivariate classes whose per-pair cache stores both (i, j) and (j, i), so the
# second direction is free: the effective pair basis is unordered.
UNORDERED_CACHE_CLASSES = {"CrossCorrelation", "Cointegration", "Barycenter", "JointEntropy"}


def _slug(value):
    return str(value).replace(".", "-").replace(" ", "")


def unit_key(module, cls, params):
    """``(unit_suffix, method_or_estimator, hyperparameters)`` for one catalogue entry.

    Entries that map to the same ``(module, cls, unit_suffix)`` share a unit.
    """
    p = dict(params or {})
    if cls in ("Covariance", "Precision"):
        est = p.get("estimator", "EmpiricalCovariance")
        return est, est, {"estimator": est}
    if cls == "CrossCorrelation":
        stat, sig = p.get("statistic", "max"), p.get("sigonly", True)
        return f"{stat}_sig-{sig}", "scipy.signal.correlate", {"statistic": stat, "sigonly": sig}
    if cls in ("SpearmanR", "KendallTau"):
        return "", cls, {}
    if cls == "PairwiseDistance":
        m = p.get("metric", "euclidean")
        return m, m, {"metric": m}
    if cls in ("DistanceCorrelation", "HilbertSchmidtIndependenceCriterion"):
        b = p.get("biased", False)
        return f"biased-{b}", "hyppo " + ("biased" if b else "unbiased"), {"biased": b}
    if cls in ("CrossDistanceCorrelation", "CrossMultiscaleGraphCorrelation"):
        lag = p.get("max_lag", 1)
        return f"maxlag-{lag}", f"hyppo max_lag={lag}", {"max_lag": lag}
    if cls in ("DynamicTimeWarping", "LongestCommonSubsequence"):
        gc = p.get("global_constraint")
        return f"constraint-{_slug(gc)}", f"tslearn global_constraint={gc}", {"global_constraint": gc}
    if cls == "SoftDynamicTimeWarping":
        return "", "tslearn soft_dtw gamma=1 (constraint ignored)", {}
    if cls == "Barycenter":
        mode = p.get("mode", "euclidean")
        return mode, f"tslearn {mode} barycenter", {"mode": mode}
    if cls == "ConvergentCrossMapping":
        E = p.get("embedding_dimension")
        return f"E-{E}", f"pyEDM CCM E={E}", {"embedding_dimension": E}
    if module == ".statistics.infotheory":
        est = p.get("estimator", "gaussian")
        hp = {"estimator": est}
        if est == "kernel":
            hp["kernel_width"] = p.get("kernel_width", 0.5)
        if est == "kraskov":
            hp["prop_k"] = p.get("prop_k", 4)
        if p.get("dyn_corr_excl"):
            hp["dyn_corr_excl"] = p["dyn_corr_excl"]
        suffix = est + ("_DCE" if p.get("dyn_corr_excl") else "")
        if cls == "CrossmapEntropy":
            hp["history_length"] = p.get("history_length", 10)
            suffix += f"_k{hp['history_length']}"
        if cls == "TransferEntropy":
            if est == "symbolic":
                return "symbolic", "JIDT TransferEntropyCalculatorGaussian (symbolic falls through)", hp
            for k in ("auto_embed_method", "k_search_max", "tau_search_max",
                      "k_history", "k_tau", "l_history", "l_tau"):
                if k in p:
                    hp[k] = p[k]
            if p.get("auto_embed_method"):
                suffix += f"_auto-{p['auto_embed_method']}_kmax-{p.get('k_search_max')}_taumax-{p.get('tau_search_max')}"
            else:
                suffix += (f"_k-{p.get('k_history', 1)}_kt-{p.get('k_tau', 1)}"
                           f"_l-{p.get('l_history', 1)}_lt-{p.get('l_tau', 1)}")
        if cls == "IntegratedInformation":
            hp = {"phitype": p.get("phitype", "star"), "delay": p.get("delay", 1),
                  "normalization": p.get("normalization", 0)}
            return f"{hp['phitype']}_norm-{hp['normalization']}", f"phi_native {hp['phitype']}", hp
        return suffix, f"JIDT {est}", hp
    if cls == "SpectralGrangerCausality":
        method = p.get("method", "nonparametric")
        if method == "nonparametric":
            return "nonparametric", "spectral_connectivity pairwise_spectral_granger_prediction", {"method": method}
        order = p.get("order")
        return f"parametric_order-{order}", f"nitime GrangerAnalyzer order={order}", {"method": method, "order": order, "max_order": p.get("max_order", 50)}
    if module in (".statistics.spectral", ".statistics.wavelet"):
        return "", cls, {}
    if cls == "LinearModel":
        m = p["model"]
        return m, f"sklearn {m}", {"model": m}
    if cls == "GPModel":
        k = p.get("kernel", "RBF")
        return k, f"sklearn GP {k}", {"kernel": k}
    if cls == "Cointegration":
        method = p.get("method", "johansen")
        if method == "johansen":
            d, k = p.get("det_order", 1), p.get("k_ar_diff", 1)
            return f"johansen_order-{d}_ardiff-{k}", "statsmodels coint_johansen", {"method": method, "det_order": d, "k_ar_diff": k}
        return "aeg", "statsmodels coint (Engle-Granger)", {"method": method}
    if cls == "PowerEnvelopeCorrelation":
        orth, log = p.get("orth", False), p.get("log", False)
        return f"orth-{orth}_log-{log}", "mne_connectivity envelope_correlation", {"orth": orth, "log": log}
    if cls in ("AdditiveNoiseModel", "ConditionalDistributionSimilarity",
               "RegressionErrorCausalInference", "InformationGeometricConditionalIndependence",
               "HellerHellerGorfine", "MultiscaleGraphCorrelation", "GromovWasserstainTau",
               "InterDependenceScore"):
        return "", cls, {}
    raise ValueError(f"no unit rule for {module}.{cls} params={params}")


CLASS_SHORT = {
    "Covariance": "cov", "Precision": "prec", "CrossCorrelation": "xcorr",
    "SpearmanR": "spearmanr", "KendallTau": "kendalltau",
    "PairwiseDistance": "pdist", "DistanceCorrelation": "dcorr",
    "HilbertSchmidtIndependenceCriterion": "hsic", "HellerHellerGorfine": "hhg",
    "MultiscaleGraphCorrelation": "mgc", "CrossDistanceCorrelation": "dcorrx",
    "CrossMultiscaleGraphCorrelation": "mgcx", "DynamicTimeWarping": "dtw",
    "LongestCommonSubsequence": "lcss", "SoftDynamicTimeWarping": "softdtw",
    "Barycenter": "bary", "GromovWasserstainTau": "gwtau",
    "AdditiveNoiseModel": "anm", "ConditionalDistributionSimilarity": "cds",
    "RegressionErrorCausalInference": "reci",
    "InformationGeometricConditionalIndependence": "igci",
    "ConvergentCrossMapping": "ccm",
    "JointEntropy": "je", "ConditionalEntropy": "ce", "CausalEntropy": "cce",
    "CrossmapEntropy": "xme", "DirectedInfo": "di", "StochasticInteraction": "si",
    "MutualInfo": "mi", "TimeLaggedMutualInfo": "tlmi", "TransferEntropy": "te",
    "IntegratedInformation": "phi",
    "CoherenceMagnitude": "cohmag", "CoherencePhase": "phase", "ImaginaryCoherence": "icoh",
    "PhaseLockingValue": "plv", "PhaseLagIndex": "pli", "WeightedPhaseLagIndex": "wpli",
    "DebiasedSquaredPhaseLagIndex": "dspli",
    "DebiasedSquaredWeightedPhaseLagIndex": "dswpli", "PairwisePhaseConsistency": "ppc",
    "DirectedCoherence": "dcoh", "PartialDirectedCoherence": "pdcoh",
    "GeneralizedPartialDirectedCoherence": "gpdcoh", "DirectedTransferFunction": "dtf",
    "DirectDirectedTransferFunction": "ddtf", "PhaseSlopeIndex": "psi", "GroupDelay": "gd",
    "SpectralGrangerCausality": "sgc",
    "weighted_PhaseLagIndex": "wspli", "debiased_weighted_squared_PhaseLagIndex": "dwspli",
    "LinearModel": "lmfit", "GPModel": "gpfit", "Cointegration": "coint",
    "PowerEnvelopeCorrelation": "pec", "InterDependenceScore": "ids",
}

# Workflow batch names: units of one family are traced/checked/profiled together.
FAMILY_RULES = [
    (lambda m, c, s: c in ("Covariance", "Precision") and s in ("MinCovDet", "EllipticEnvelope"), "cov_prec_robust"),
    (lambda m, c, s: c in ("Covariance", "Precision") and s.startswith("GraphicalLasso"), "cov_prec_glasso"),
    (lambda m, c, s: c in ("Covariance", "Precision"), "cov_prec_simple"),
    (lambda m, c, s: c == "CrossCorrelation", "xcorr"),
    (lambda m, c, s: c in ("SpearmanR", "KendallTau"), "rank_corr"),
    (lambda m, c, s: c == "PairwiseDistance", "pdist"),
    (lambda m, c, s: c in ("DistanceCorrelation", "HilbertSchmidtIndependenceCriterion"), "dcorr_hsic"),
    (lambda m, c, s: c in ("HellerHellerGorfine", "MultiscaleGraphCorrelation"), "hhg_mgc"),
    (lambda m, c, s: c in ("CrossDistanceCorrelation", "CrossMultiscaleGraphCorrelation"), "lagged_dcorr_mgc"),
    (lambda m, c, s: c in ("DynamicTimeWarping", "LongestCommonSubsequence", "SoftDynamicTimeWarping"), "dtw_family"),
    (lambda m, c, s: c == "Barycenter", "barycenters"),
    (lambda m, c, s: c == "GromovWasserstainTau", "gwtau"),
    (lambda m, c, s: c in ("AdditiveNoiseModel", "ConditionalDistributionSimilarity",
                           "RegressionErrorCausalInference", "InformationGeometricConditionalIndependence"), "cdt_pairwise"),
    (lambda m, c, s: c == "ConvergentCrossMapping", "ccm"),
    (lambda m, c, s: c == "IntegratedInformation", "phi"),
    (lambda m, c, s: m == ".statistics.infotheory" and s.startswith("kraskov"), "it_kraskov"),
    (lambda m, c, s: m == ".statistics.infotheory" and s.startswith("kozachenko"), "it_kozachenko"),
    (lambda m, c, s: m == ".statistics.infotheory" and s.startswith("kernel"), "it_kernel"),
    (lambda m, c, s: m == ".statistics.infotheory", "it_gaussian"),
    (lambda m, c, s: c in ("CoherenceMagnitude", "CoherencePhase", "ImaginaryCoherence") and m == ".statistics.spectral", "spectral_coherence"),
    (lambda m, c, s: c in ("PhaseLockingValue", "PhaseLagIndex", "WeightedPhaseLagIndex", "DebiasedSquaredPhaseLagIndex",
                           "DebiasedSquaredWeightedPhaseLagIndex", "PairwisePhaseConsistency") and m == ".statistics.spectral", "spectral_phase_sync"),
    (lambda m, c, s: c in ("DirectedCoherence", "PartialDirectedCoherence", "GeneralizedPartialDirectedCoherence",
                           "DirectedTransferFunction", "DirectDirectedTransferFunction"), "spectral_directed"),
    (lambda m, c, s: c in ("PhaseSlopeIndex", "GroupDelay") and m == ".statistics.spectral", "spectral_psi_gd"),
    (lambda m, c, s: c == "SpectralGrangerCausality", "sgc"),
    (lambda m, c, s: m == ".statistics.wavelet" and c == "PhaseSlopeIndex", "wavelet_psi"),
    (lambda m, c, s: m == ".statistics.wavelet", "wavelet_measures"),
    (lambda m, c, s: c == "LinearModel", "lmfit"),
    (lambda m, c, s: c == "GPModel", "gpfit"),
    (lambda m, c, s: c == "Cointegration", "coint"),
    (lambda m, c, s: c == "PowerEnvelopeCorrelation", "pec"),
    (lambda m, c, s: c == "InterDependenceScore", "ids"),
]


def family_of(module, cls, suffix):
    for rule, name in FAMILY_RULES:
        if rule(module, cls, suffix):
            return name
    raise ValueError(f"no family for {module}.{cls}/{suffix}")


def pair_basis_of(cls):
    if cls in MULTIVARIATE_CLASSES:
        return "multivariate"
    if cls in UNORDERED_CACHE_CLASSES:
        return "unordered"
    return "ordered"


INFRASTRUCTURE_UNITS = [
    dict(unit_id="preproc.detrend", module="data", pyspi_class="Data",
         method_or_estimator="scipy.signal.detrend (linear, axis=1)", family="preprocessing_base",
         notes="Optional step before z-scoring (pyspi/data.py); the EEG workflow uses detrend=True."),
    dict(unit_id="preproc.zscore", module="data", pyspi_class="Data",
         method_or_estimator="scipy.stats.zscore(ddof=1, axis=1) after truncation and detrend", family="preprocessing_base",
         notes="Default normalise=True; order truncate -> detrend -> zscore."),
    dict(unit_id="preproc.two_array_api", module="base", pyspi_class="parse_bivariate",
         method_or_estimator="Data() + add_process twice (second array not z-scored)", family="preprocessing_base",
         notes="Only affects the direct two-array API, not Calculator."),
    dict(unit_id="base.pair_handling", module="base", pyspi_class="Directed/Undirected",
         method_or_estimator="Directed loops all ordered pairs; Undirected then mirrors the upper triangle", family="preprocessing_base",
         notes="Classes overriding multivariate bypass the mirror (REVIEW M1; tests/README.md bug 5 before b2f6d7e)."),
    dict(unit_id="spectral.multitaper_csm", module="spectral", pyspi_class="NonparametricSpectral",
         method_or_estimator="spectral_connectivity Multitaper(NW=3, 5 DPSS) + Connectivity cross-spectral matrix", family="spectral_csm",
         notes="Shared estimator for every multitaper SPI; expectation is over the 5 tapers."),
    dict(unit_id="spectral.band_selection", module="spectral", pyspi_class="NonparametricSpectral",
         method_or_estimator="fmin/fmax selection on the FFT grid (inclusive multivariate, strict bivariate/PSI/GD)", family="spectral_csm",
         notes="Narrow EEG bands at T=512/1024 leave 0-3 bins."),
    dict(unit_id="spectral.wilson_factorisation", module="spectral", pyspi_class="NonparametricSpectralBivariate",
         method_or_estimator="spectral_connectivity minimum_phase_decomposition (Wilson, tol 1e-8, 60 iterations)", family="spectral_directed",
         notes="Shared by dtf/dcoh/pdcoh/gpdcoh/ddtf and nonparametric sgc."),
    dict(unit_id="wavelet.morlet_cwt", module="wavelet", pyspi_class="mne",
         method_or_estimator="mne_connectivity cwt_morlet with cwt_freqs=linspace(0.2,0.5,125), n_cycles=freqs/7", family="wavelet_measures",
         notes="n_cycles=freqs/7 makes every wavelet one sample long."),
    dict(unit_id="infotheory.est.gaussian", module="infotheory", pyspi_class="JIDTBase",
         method_or_estimator="JIDT Gaussian entropy/MI/TE calculators (nats)", family="it_gaussian", notes=""),
    dict(unit_id="infotheory.est.kozachenko", module="infotheory", pyspi_class="JIDTBase",
         method_or_estimator="JIDT Kozachenko-Leonenko entropy (nats, 1e-8 noise, NORMALISE=true)", family="it_kozachenko", notes=""),
    dict(unit_id="infotheory.est.kernel", module="infotheory", pyspi_class="JIDTBase",
         method_or_estimator="JIDT box-kernel entropy/MI/TE (bits, width 0.5 or 0.25)", family="it_kernel", notes=""),
    dict(unit_id="infotheory.est.ksg1", module="infotheory", pyspi_class="JIDTBase",
         method_or_estimator="JIDT Kraskov KSG1 MI / conditional MI (nats, k=4, 1e-8 noise, NUM_THREADS=all)", family="it_kraskov", notes=""),
    dict(unit_id="infotheory.theiler_window", module="infotheory", pyspi_class="JIDTBase",
         method_or_estimator="DYN_CORR_EXCL = int(2 * dot(acf(src), acf(targ))) per pair", family="it_kraskov", notes=""),
]


# Identifiers in the EEG selection that config.yaml does not instantiate. Each is
# built from the stated class and params, and the identifier is asserted to match.
RETAINED_EXTRA_SPIS = {
    "xcorr_max_sig-False": (".statistics.basic", "CrossCorrelation", {"statistic": "max", "sigonly": False}),
    "xcorr-sq_max_sig-False": (".statistics.basic", "CrossCorrelation", {"statistic": "max", "sigonly": False, "squared": True}),
}


def load_retained():
    if not RETAINED_FILE.is_file():
        return []
    return [l.strip() for l in RETAINED_FILE.read_text().splitlines() if l.strip() and not l.startswith("#")]


def band_stem(identifier):
    """Identifier without its band suffix, e.g. ``cohmag_multitaper_mean``."""
    for marker in ("_multitaper_", "_wavelet_", "_nonparametric_", "_parametric_"):
        if marker in identifier:
            head, tail = identifier.split(marker, 1)
            return head + marker + tail.split("_fs-")[0]
    return identifier


def build():
    harness.ensure_jvm()
    harness.import_statistics_modules()
    units = OrderedDict()
    retained_entries = []
    for ident, (module, cls, params) in RETAINED_EXTRA_SPIS.items():
        spi = harness.build_spi(dict(module=module, cls=cls, params=params))
        assert spi.identifier == ident, (spi.identifier, ident)
        retained_entries.append(dict(identifier=ident, module=module, cls=cls, params=params,
                                     labels=sorted(set(spi.labels)), deps=[]))
    for source, cfg in (("config", None), ("extra", EXTRA_CONFIG), ("retained", retained_entries)):
        entries = cfg if source == "retained" else harness.catalogue(cfg)[0]
        for e in entries:
            suffix, method, hp = unit_key(e["module"], e["cls"], e["params"])
            short = CLASS_SHORT[e["cls"]]
            mod = e["module"].split(".")[-1]
            uid = f"{mod}.{short}" + (f".{suffix}" if suffix else "")
            if source == "extra":
                uid = f"{mod}.{short}"  # the nine extra wavelet classes
            u = units.get(uid)
            if u is None:
                u = units[uid] = dict(
                    unit_id=uid, module=mod, pyspi_class=e["cls"], method_or_estimator=method,
                    hyperparameters=hp, source=source, identifiers=[], identifier_params={},
                    representative=e["identifier"], family=family_of(e["module"], e["cls"], suffix),
                    pair_basis=pair_basis_of(e["cls"]), labels=e["labels"], deps=e["deps"],
                )
            if e["identifier"] not in u["identifiers"]:
                u["identifiers"].append(e["identifier"])
                u["identifier_params"][e["identifier"]] = e["params"]
    # A unit row describes its base identifier: a "-sq" relabel never labels the row when the
    # unsquared identifier is in the same unit (README "Two score columns", rule R1).
    for u in units.values():
        rep = u["representative"]
        if rep.endswith("-sq") and rep[:-3] in u["identifiers"]:
            u["representative"] = rep[:-3]
    # Retained EEG selection: exact identifier match first, then band-stem match.
    retained = load_retained()
    by_stem = {}
    for u in units.values():
        for ident in u["identifiers"]:
            by_stem.setdefault(band_stem(ident), set()).add(u["unit_id"])
    unmatched = []
    for ident in retained:
        hits = {u["unit_id"] for u in units.values() if ident in u["identifiers"]}
        if not hits:
            hits = by_stem.get(band_stem(ident), set())
        if len(hits) != 1:
            unmatched.append((ident, sorted(hits)))
            continue
        u = units[next(iter(hits))]
        u.setdefault("eeg_identifiers", []).append(ident)
    for u in units.values():
        u["in_eeg_selection"] = bool(u.get("eeg_identifiers"))
        u.setdefault("eeg_identifiers", [])
    for inf in INFRASTRUCTURE_UNITS:
        u = dict(inf)
        u.update(hyperparameters={}, source="infrastructure", identifiers=[], identifier_params={},
                 representative=None, pair_basis="n/a", labels=[], deps=[], eeg_identifiers=[],
                 in_eeg_selection=False)
        units[u["unit_id"]] = u
    doc = dict(
        schema=1,
        generated_by="audit/build_units.py",
        sources=dict(config=str(harness.DEFAULT_CONFIG.relative_to(REPO)),
                     extra=str(EXTRA_CONFIG.relative_to(REPO)),
                     retained=str(RETAINED_FILE.relative_to(REPO)) if RETAINED_FILE.is_file() else None),
        counts=dict(units=len(units),
                    config_identifiers=sum(len(u["identifiers"]) for u in units.values() if u["source"] == "config"),
                    extra_identifiers=sum(len(u["identifiers"]) for u in units.values() if u["source"] == "extra"),
                    retained_only_identifiers=sum(len(u["identifiers"]) for u in units.values() if u["source"] == "retained"),
                    retained_identifiers=len(retained),
                    retained_matched=sum(len(u["eeg_identifiers"]) for u in units.values()),
                    infrastructure=len(INFRASTRUCTURE_UNITS)),
        retained_unmatched=[dict(identifier=i, candidates=c) for i, c in unmatched],
        units=list(units.values()),
    )
    return doc


def dump(doc):
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if audit/units.yaml is stale")
    ap.add_argument("--out", default=str(AUDIT_DIR / "units.yaml"))
    a = ap.parse_args()
    text = dump(build())
    out = Path(a.out)
    if a.check:
        if not out.is_file() or out.read_text() != text:
            print(f"{out} is stale; rerun build_units.py", file=sys.stderr)
            sys.exit(1)
        print("units.yaml up to date")
        return
    out.write_text(text)
    doc = yaml.safe_load(text)
    print(f"wrote {out}: {doc['counts']}")
    if doc["retained_unmatched"]:
        print("UNMATCHED retained identifiers:")
        for r in doc["retained_unmatched"]:
            print("  ", r)
    fams = {}
    for u in doc["units"]:
        fams.setdefault(u["family"], []).append(u["unit_id"])
    print(f"{len(fams)} families:")
    for f, ids in fams.items():
        print(f"  {f}: {len(ids)} units")


if __name__ == "__main__":
    main()
