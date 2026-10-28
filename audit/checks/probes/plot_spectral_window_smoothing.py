"""Plot the saved experiment receipt; does not recalculate connectivity."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
report = json.loads((ROOT/'audit/briefing/spectral_window_smoothing.json').read_text())
keys = [s['name'] for s in report['settings']]
labels = [f"{s['duration_s']:g} s\n{s['actual_tapers']} tapers" for s in report['settings']]
colors = ['#335c81', '#2a9d8f', '#e9a23b', '#b75c84', '#6d597a']
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
ax = axes[0, 0]
null = [report['synthetic']['summary'][k]['null_msc_alpha'] for k in keys]
ax.bar(labels, [x['mean'] for x in null], yerr=[x['sd'] for x in null],
       color=colors, capsize=4)
ax.axhline(0, color='black', linewidth=.8)
ax.set(title='Independent signals: recording-mean coherence',
       ylabel='Magnitude-squared coherence (true value = 0)', ylim=(0, .6))
ax.text(.02, .97, 'Error bars: SD across 16 synthetic recording means',
        transform=ax.transAxes, va='top', fontsize=9)

patients = report['eeg']['patients']
ax = axes[0, 1]
for patient in patients:
    correlations = []
    for key in keys:
        rows = patient['settings'][key]['comparison_to_baseline']
        row = next(r for r in rows if r['metric']=='msc' and r['band']==[8, 12])
        correlations.append(row['edge_rank_correlation'])
    ax.plot(labels, correlations, marker='o', alpha=.8,
            label=patient['metadata']['patient_id'])
ax.set(title='EEG: preservation of alpha connection rankings',
       ylabel='Spearman correlation with 0.5 s / five tapers', ylim=(0, 1.03))
ax.legend(title='Patient', ncol=3, fontsize=8, loc='lower left')

ax = axes[1, 0]
for i, key in enumerate(['one_nw2', 'two_nw3']):
    for j, patient in enumerate(patients):
        rows = patient['settings'][key]['comparison_to_baseline']
        changes = [r['overlap_vs_nonoverlap_mean_absolute_difference']
                   for r in rows if r['metric']=='msc']
        ax.scatter(i+(j-2)*.035, np.median(changes), s=45, color=f'C{j}')
ax.set_xticks([0, 1], ['1 s / three tapers', '2 s / five tapers'])
ax.set(title='EEG: effect of overlap on recording means',
       ylabel='Absolute coherence difference\n(median across five bands)',
       xlim=(-.4, 1.4))
ax.set_ylim(0, .0053)
ax.text(.02, .97, 'Each point is one patient; same duration and tapers.\n'
        'Compare 0.5 s steps with non-overlapping windows.',
        transform=ax.transAxes, va='top', fontsize=9)

ax = axes[1, 1]
patient_ratios = np.array([[p['settings'][k]['seconds']/p['settings'][keys[0]]['seconds']
                           for k in keys] for p in patients])
ratios = np.median(patient_ratios, axis=0)
errors = [ratios-patient_ratios.min(axis=0), patient_ratios.max(axis=0)-ratios]
ax.bar(labels, ratios, color=colors, yerr=errors, capsize=4)
ax.axhline(1, color='black', linewidth=.8, linestyle='--')
for i, ratio in enumerate(ratios):
    ax.text(i, patient_ratios[:, i].max()+.15, f'{ratio:.2f}×', ha='center')
ax.set(title='EEG spectral kernel runtime at 0.5 s steps',
       ylabel='Relative runtime: 12 contacts, four estimators',
       ylim=(0, patient_ratios.max()*1.2))
ax.text(.02, .97, 'Median and range across five records; reusable tapers.\nExcludes EEG preparation.',
        transform=ax.transAxes, va='top', fontsize=9)
for ax in axes.flat:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18)
fig.suptitle('Window and smoothing pilot: mean across windows', fontsize=16)
fig.savefig(ROOT/'audit/briefing/spectral_window_smoothing.png', dpi=150, bbox_inches='tight')
