"""Emit one JSON briefing pack per family for the audit workflows.

Each pack lists the family's units (identifiers, params, representative), the
supplement text line ranges and GitBook page to read, the pyspi source files,
and the third-party code paths. Written to ``audit/briefing/<family>.json``;
regenerate after ``build_units.py``.
"""
import json
import re
from pathlib import Path

import yaml

AUDIT = Path(__file__).resolve().parents[1]
REPO = AUDIT.parent
DOCS = Path("/media/dan/Data/git/pyspiDocumentation")
SUPP_TXT = DOCS / "supplement" / "supplement.txt"
GITBOOK = DOCS / "information-about-pyspi" / "spis" / "spi-descriptions"
SP = REPO / ".venv" / "lib" / "python3.12" / "site-packages"

# supplement section -> family list (section numbers from supplement.txt headings)
SECTION_FAMILIES = {
    "1.1.1": ["cov_prec_simple", "cov_prec_robust", "cov_prec_glasso"],
    "1.1.2": ["cov_prec_simple", "cov_prec_robust", "cov_prec_glasso"],
    "1.1.3": ["rank_corr"], "1.1.4": ["rank_corr"], "1.1.5": ["xcorr"],
    "1.2": ["pdist"], "1.2.1": ["dcorr_hsic"], "1.2.2": ["lagged_dcorr_mgc"], "1.2.3": ["hhg_mgc"],
    "1.2.4": ["lagged_dcorr_mgc"], "1.2.5": ["dcorr_hsic"], "1.2.6": ["hhg_mgc"],
    "1.2.7": ["dtw_family"], "1.2.8": ["dtw_family"], "1.2.9": ["dtw_family"], "1.2.10": ["barycenters"],
    "1.3.1": ["cdt_pairwise"], "1.3.2": ["cdt_pairwise"], "1.3.3": ["cdt_pairwise"], "1.3.4": ["cdt_pairwise"],
    "1.3.5": ["ccm"],
    "1.4": ["it_gaussian", "it_kozachenko", "it_kernel", "it_kraskov"],
    "1.4.1": ["it_gaussian", "it_kozachenko", "it_kernel"], "1.4.2": ["it_gaussian", "it_kozachenko", "it_kernel"],
    "1.4.3": ["it_gaussian", "it_kozachenko", "it_kernel", "it_kraskov"],
    "1.4.4": ["it_gaussian", "it_kozachenko", "it_kernel", "it_kraskov"],
    "1.4.5": ["it_kraskov", "it_kernel", "it_gaussian"], "1.4.6": ["it_gaussian"],
    "1.4.7": ["it_gaussian", "it_kozachenko", "it_kernel"], "1.4.8": ["it_gaussian", "it_kozachenko", "it_kernel"],
    "1.4.9": ["it_gaussian", "it_kozachenko", "it_kernel"], "1.4.10": ["phi"],
    "1.5": ["spectral_csm", "spectral_coherence", "spectral_phase_sync", "spectral_directed", "spectral_psi_gd", "sgc", "wavelet_measures", "wavelet_psi"],
    "1.5.1": ["spectral_coherence", "wavelet_measures"], "1.5.2": ["spectral_coherence", "wavelet_measures"],
    "1.5.3": ["spectral_psi_gd"], "1.5.4": ["spectral_psi_gd", "wavelet_psi"],
    "1.5.5": ["spectral_coherence", "wavelet_measures"], "1.5.6": ["spectral_phase_sync", "wavelet_measures"],
    "1.5.7": ["spectral_phase_sync", "wavelet_measures"], "1.5.8": ["spectral_phase_sync", "wavelet_measures"],
    "1.5.9": ["spectral_directed"], "1.5.10": ["spectral_directed"], "1.5.11": ["spectral_directed"], "1.5.12": ["sgc"],
    "1.6.1": ["lmfit"], "1.6.2": ["gpfit"], "1.6.3": ["coint"], "1.6.4": ["pec"],
}
GITBOOK_PAGES = {
    "basic-statistics.md": ["cov_prec_simple", "cov_prec_robust", "cov_prec_glasso", "rank_corr", "xcorr"],
    "distance-similarity.md": ["pdist", "dcorr_hsic", "hhg_mgc", "lagged_dcorr_mgc", "dtw_family", "barycenters", "gwtau"],
    "causal-inference.md": ["cdt_pairwise", "ccm"],
    "information-theory.md": ["it_gaussian", "it_kozachenko", "it_kernel", "it_kraskov", "phi"],
    "spectral.md": ["spectral_csm", "spectral_coherence", "spectral_phase_sync", "spectral_directed", "spectral_psi_gd", "sgc", "wavelet_measures", "wavelet_psi"],
    "miscellaneous.md": ["lmfit", "gpfit", "coint", "pec", "ids"],
}
THIRD_PARTY = {
    "cov_prec_simple": ["sklearn/covariance/_empirical_covariance.py", "sklearn/covariance/_shrunk_covariance.py"],
    "cov_prec_robust": ["sklearn/covariance/_robust_covariance.py", "sklearn/covariance/_elliptic_envelope.py"],
    "cov_prec_glasso": ["sklearn/covariance/_graph_lasso.py"],
    "rank_corr": ["scipy/stats/_stats_py.py (spearmanr, kendalltau)"],
    "xcorr": ["scipy/signal/_signaltools.py (correlate, choose_conv_method)"],
    "pdist": ["sklearn/metrics/pairwise.py (pairwise_distances, euclidean_distances)", "scipy/spatial/distance.py"],
    "dcorr_hsic": ["hyppo/independence/dcorr.py", "hyppo/independence/hsic.py", "hyppo/tools/common.py"],
    "hhg_mgc": ["hyppo/independence/hhg.py", "hyppo/independence/mgc.py", "scipy/stats/_mgc.py"],
    "lagged_dcorr_mgc": ["hyppo/time_series/_utils.py", "hyppo/time_series/dcorrx.py", "hyppo/time_series/mgcx.py"],
    "dtw_family": ["tslearn/metrics/_dtw.py", "tslearn/metrics/dtw_variants.py", "tslearn/metrics/softdtw_variants.py", "tslearn/metrics/soft_dtw_fast.py", "tslearn/metrics/_masks.py"],
    "barycenters": ["tslearn/barycenters/euclidean.py", "tslearn/barycenters/dba.py", "tslearn/barycenters/softdtw.py"],
    "gwtau": ["(pure pyspi: pyspi/statistics/distance.py GromovWasserstainTau)"],
    "cdt_pairwise": ["cdt/causality/pairwise/ANM.py", "cdt/causality/pairwise/CDS.py", "cdt/causality/pairwise/RECI.py", "cdt/causality/pairwise/IGCI.py", "sklearn/gaussian_process/_gpr.py"],
    "ccm": ["pyEDM/CoreEDM.py", "pyEDM/AuxFunc.py", "pyEDM/pyBindEDM*.so (cppEDM; use the cppEDM GitHub source for the algorithm)"],
    "it_gaussian": ["JIDT 1.6.1 jar (pyspi/lib/jidt/infodynamics.jar); source: github.com/jlizier/jidt v1.6.1 infodynamics/measures/continuous/gaussian/*.java"],
    "it_kozachenko": ["JIDT: infodynamics/measures/continuous/kozachenko/EntropyCalculatorMultiVariateKozachenko.java"],
    "it_kernel": ["JIDT: infodynamics/measures/continuous/kernel/*.java (box kernel, KERNEL_WIDTH)"],
    "it_kraskov": ["JIDT: infodynamics/measures/continuous/kraskov/{MutualInfoCalculatorMultiVariateKraskov1,TransferEntropyCalculatorKraskov,ConditionalMutualInfoCalculatorMultiVariateKraskov1}.java, TransferEntropyCalculatorViaCondMutualInfo.java (auto-embedding)"],
    "phi": ["pyspi/lib/phi_native.py; MATLAB original: git show 76ab2bb^:pyspi/lib/PhiToolbox/Phi/phi_star_Gauss.m (also phi_G_Gauss_AL.m, utility/I_s_I_s_d.m, utility/Gauss/logdet.m, Cov_comp.m)"],
    "spectral_csm": ["spectral_connectivity/transforms.py (Multitaper, _make_tapers, fft)", "spectral_connectivity/connectivity.py (Connectivity, _cross_spectral_matrix, _expectation, _non_negative_frequencies)"],
    "spectral_coherence": ["spectral_connectivity/connectivity.py (coherency, coherence_magnitude, coherence_phase, imaginary_coherence)"],
    "spectral_phase_sync": ["spectral_connectivity/connectivity.py (phase_locking_value, phase_lag_index, weighted_phase_lag_index, debiased_squared_*, pairwise_phase_consistency)"],
    "spectral_directed": ["spectral_connectivity/connectivity.py (directed_transfer_function, directed_coherence, partial_directed_coherence, generalized_partial_directed_coherence, direct_directed_transfer_function, _transfer_function, _noise_covariance)", "spectral_connectivity/minimum_phase_decomposition.py"],
    "spectral_psi_gd": ["spectral_connectivity/connectivity.py (phase_slope_index, _inner_combination, _bandpass, group_delay, _find_significant_frequencies)", "spectral_connectivity/statistics.py"],
    "sgc": ["spectral_connectivity/connectivity.py (pairwise_spectral_granger_prediction, _estimate_predictive_power, _remove_instantaneous_causality)", "nitime/analysis/granger.py", "nitime/algorithms/autoregressive.py", "nitime/algorithms/spectral.py", "nitime/utils.py"],
    "wavelet_measures": ["mne_connectivity/spectral/epochs.py", "mne_connectivity/spectral/epochs_bivariate.py", "mne/time_frequency/tfr.py (morlet, cwt, _cwt_gen)"],
    "wavelet_psi": ["mne_connectivity/effective.py (phase_slope_index, _compute_psi)", "mne_connectivity/spectral/epochs.py"],
    "lmfit": ["sklearn/linear_model/_ridge.py", "sklearn/linear_model/_coordinate_descent.py", "sklearn/linear_model/_stochastic_gradient.py", "sklearn/linear_model/_bayes.py"],
    "gpfit": ["sklearn/gaussian_process/_gpr.py", "sklearn/gaussian_process/kernels.py"],
    "coint": ["statsmodels/tsa/stattools/_stattools.py (coint, adfuller)", "statsmodels/tsa/vector_ar/vecm.py (coint_johansen)"],
    "pec": ["mne_connectivity/envelope.py (envelope_correlation)"],
    "ids": ["pyspi/lib/ids/dependence.py", "pyspi/lib/ids/numpy_dependence.py"],
    "preprocessing_base": ["pyspi/data.py", "pyspi/base.py", "pyspi/calculator.py"],
}
PYSPI_FILES = {
    "basic": "pyspi/statistics/basic.py", "distance": "pyspi/statistics/distance.py",
    "causal": "pyspi/statistics/causal.py", "infotheory": "pyspi/statistics/infotheory.py",
    "spectral": "pyspi/statistics/spectral.py", "wavelet": "pyspi/statistics/wavelet.py",
    "misc": "pyspi/statistics/misc.py", "data": "pyspi/data.py", "base": "pyspi/base.py",
}


def supplement_sections():
    lines = SUPP_TXT.read_text().splitlines()
    heads = []
    pat = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+([A-Z].{3,80}?)\s*$")
    for i, l in enumerate(lines, 1):
        m = pat.match(l)
        if m and m.group(1).startswith("1"):
            heads.append((i, m.group(1), m.group(2).strip()))
    out = {}
    for n, (line, num, title) in enumerate(heads):
        end = heads[n + 1][0] - 1 if n + 1 < len(heads) else len(lines)
        out[num] = dict(title=title, lines=[line, end])
    return out


def main():
    units = yaml.safe_load((AUDIT / "units.yaml").read_text())["units"]
    secs = supplement_sections()
    fams = {}
    for u in units:
        fams.setdefault(u["family"], []).append(u)
    out_dir = AUDIT / "briefing"
    hyp_file = out_dir / "hypotheses.json"
    hypotheses = json.loads(hyp_file.read_text()) if hyp_file.is_file() else {}
    for fam, us in fams.items():
        sections = [dict(section=k, **v) for k, v in secs.items() if fam in SECTION_FAMILIES.get(k, [])]
        pages = [p for p, f in GITBOOK_PAGES.items() if fam in f]
        pack = dict(
            family=fam,
            units=[dict(unit_id=u["unit_id"], pyspi_class=u["pyspi_class"], module=u["module"],
                        method_or_estimator=u["method_or_estimator"], hyperparameters=u["hyperparameters"],
                        source=u["source"], identifiers=u["identifiers"], identifier_params=u["identifier_params"],
                        representative=u["representative"], pair_basis=u["pair_basis"],
                        in_eeg_selection=u["in_eeg_selection"], eeg_identifiers=u["eeg_identifiers"],
                        labels=u["labels"]) for u in us],
            pyspi_files=sorted({PYSPI_FILES[u["module"]] for u in us if u["module"] in PYSPI_FILES}),
            third_party=[(p if p.startswith("(") or p.startswith("JIDT") or p.startswith("pyspi") else str(SP / p)) for p in THIRD_PARTY.get(fam, [])],
            supplement_txt=str(SUPP_TXT),
            supplement_sections=sections,
            gitbook_pages=[str(GITBOOK / p) for p in pages],
        )
        pack["hypotheses"] = hypotheses.get(fam, [])
        pack["known_issues_file"] = str(out_dir / "KNOWN_ISSUES.md")
        (out_dir / f"{fam}.json").write_text(json.dumps(pack, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(fams)} briefing packs; supplement sections found: {len(secs)}")


if __name__ == "__main__":
    main()
