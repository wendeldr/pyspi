"""Seeded ordinal-TE controls, including independent autocorrelated channels.

This is an estimator feasibility check, not a clinical/null-significance study.
The longer synthetic records characterize bias; run windows remain 1024 samples.
"""

import contextlib
import datetime
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

import numpy as np
from scipy.signal import lfilter
from pyspi.data import Data
from pyspi.statistics.infotheory import TransferEntropy


def controls(repeats=8):
    rows=[]
    for spacing in (4,16):
        spi=TransferEntropy(estimator='symbolic',pattern_length=3,
                            pattern_spacing=spacing,prediction_shift=spacing)
        for samples in (1024,8192):
            values=[]
            for repeat in range(repeats):
                rng=np.random.default_rng(14100+repeat)
                ar=lfilter([1.],[1.,-.9],rng.normal(size=(3,samples+512)),axis=1)[:,512:]
                # Source and independent control have the same AR law. The
                # target adds a delayed source, with no target->source edge.
                x=ar.copy()
                x[1,spacing:]=.9*ar[0,:-spacing]+.1*ar[1,spacing:]
                with contextlib.redirect_stdout(io.StringIO()):
                    data=Data(x,normalise=False,detrend=False)
                matrix=spi.multivariate(data)
                values.append(dict(coupled=float(matrix[0,1]),reverse=float(matrix[1,0]),
                                   independent=float(matrix[0,2])))
            means={key:float(np.mean([row[key] for row in values])) for key in values[0]}
            rows.append(dict(spacing=spacing,prediction_shift=spacing,samples=samples,
                             repeats=repeats,seed_start=14100,values=values,mean=means))
    return rows


def main():
    rows=controls()
    for row in rows:
        assert row['mean']['coupled']>row['mean']['independent']+.2
        assert row['mean']['coupled']>row['mean']['reverse']+.2
    for spacing in (4,16):
        short,long=[row for row in rows if row['spacing']==spacing]
        assert long['mean']['independent']<short['mean']['independent']/3
    sources=['pyspi/_symbolic.py','audit/checks/probes/symbolic_null_review.py']
    report=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model='Independent AR(1), coefficient 0.9, Gaussian innovations; target = 0.9 delayed source + 0.1 independent AR noise.',
        units='nats',results=rows,
        limits='Positive finite-sample null bias is expected. These controls do not establish patient-specific significance, optimal lags or clinical discrimination.',
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    (ROOT/'audit/briefing/symbolic_null_review.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in rows:
        print(row['spacing'],row['samples'],row['mean'],flush=True)


if __name__=='__main__':
    main()
