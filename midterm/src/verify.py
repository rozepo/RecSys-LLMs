"""Independent brute-force checks of the join, ALS equations, ranking and CSVs."""
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse

from prepare import BEGIN, DATA, END, OUT, ROOT, SPLIT, covered_integral, seconds
from models import als, ap5, rank


def verify():
    started = time.perf_counter()
    experiments = json.loads((OUT/'experiments.json').read_text())
    ids = [e['user_id'] for e in experiments['examples']]
    selected = []
    for chunk in pd.read_csv(DATA/'dataset11-30.csv', dtype={'user_id': str}, chunksize=300000):
        selected.append(chunk[chunk.user_id.isin(ids)])
    views = pd.concat(selected)
    views['s'] = seconds(views.start_time, '%Y-%m-%d %H:%M:%S')
    views['e'] = seconds(views.stop_time, '%Y-%m-%d %H:%M:%S')
    frames = [pd.read_csv(DATA/f'export_arh_{p}-final.csv', usecols=['channel_id', 'start_time', 'duration', 'tv_show_id'])
              for p in ['11-20', '21-30']]
    schedule = pd.concat(frames)
    schedule['s'] = seconds(schedule.start_time, '%d.%m.%Y %H:%M:%S')
    schedule['e'] = schedule.s+schedule.duration
    schedule = schedule.drop_duplicates(['channel_id', 's', 'e', 'tv_show_id'])
    schedule = schedule[(schedule.tv_show_id != 0) & (schedule.e > schedule.s)]
    grouped = {c: (g.s.to_numpy(), g.e.to_numpy(), g.tv_show_id.to_numpy()) for c, g in schedule.groupby('channel_id')}
    checked = {}
    for name, lo, hi in [('train', BEGIN, SPLIT), ('validation', SPLIT, END)]:
        expected = {}
        for row in views.itertuples():
            s, e = max(row.s, lo), min(row.e, hi)
            if s >= e or row.vsetv_id not in grouped:
                continue
            ss, ee, ii = grouped[row.vsetv_id]
            overlaps = np.maximum(0, np.minimum(e, ee)-np.maximum(s, ss))
            for item, value in zip(ii[overlaps > 0], overlaps[overlaps > 0]):
                key = (row.user_id, int(item))
                expected[key] = expected.get(key, 0)+int(value)
        actual = pd.read_csv(OUT/f'{name}_interactions.csv.gz', dtype={'user_id': str})
        actual = actual[actual.user_id.isin(ids)]
        actual_map = {(r.user_id, int(r.tv_show_id)): int(r.seconds) for r in actual.itertuples()}
        assert actual_map == expected, f'Independent join mismatch: {name}'
        checked[name] = len(expected)
    # Half-open boundary and union with overlapping schedule rows.
    fn = covered_integral(np.array([0, 5, 20]), np.array([10, 15, 30]))
    assert np.array_equal(fn(np.array([0, 10, 15, 20, 25, 30])), [0, 10, 15, 15, 20, 25])
    assert max(0, min(10, 20)-max(0, 10)) == 0
    assert ap5([1, 2, 3, 4, 5], {1, 3}) == (1+2/3)/2
    assert ap5([1, 2, 3, 4, 5], set()) == 0
    # Independent dense normal equations for the last ALS item update.
    tiny = sparse.csr_matrix([[60., 0., 600.], [0., 120., 60.]])
    uf, itf = als(tiny, factors=3, scale=2., regularization=4., iterations=2)
    confidence = 1+2*np.log1p(tiny.toarray()/60)
    preference = tiny.toarray() > 0
    residuals = []
    for i in range(3):
        a = np.einsum('uf,u,ug->fg', uf, confidence[:, i], uf)+4*np.eye(3)
        b = np.einsum('uf,u->f', uf, confidence[:, i]*preference[:, i])
        residual = np.max(np.abs(np.einsum('fg,g->f', a, itf[i])-b))
        residuals.append(float(residual))
        assert residual < 1e-10
    candidates = np.array([True]*7)
    pop = np.arange(7, dtype=float)
    rec = rank(np.array([[1., 0, 0, 0, 0, 0, 0], [0.]*7]), candidates, pop, np.array([False, True]))
    assert rec.tolist() == [[0, 6, 5, 4, 3], [6, 5, 4, 3, 2]]
    template = pd.read_csv(DATA/'submission.csv', dtype=str)
    allowed = set(np.load(OUT/'final_candidates.npy').tolist())
    files = []
    for info in experiments['submissions']:
        path = ROOT/info['file']
        frame = pd.read_csv(path, dtype=str)
        assert frame.shape == (1260, 2)
        assert frame.columns.tolist() == template.columns.tolist()
        assert frame.user_id.tolist() == template.user_id.tolist() and frame.user_id.is_unique
        assert len(path.read_text().splitlines()) == 1261
        for value in frame.tv_show_id:
            tokens = value.split(' ')
            assert len(tokens) == len(set(tokens)) == 5 and all(t.isdigit() for t in tokens)
            assert set(map(int, tokens)) <= allowed and '0' not in tokens
        files.append(info['file'])
    result = {'status': 'passed', 'independent_join_view_rows': len(views), 'users': ids,
              'independent_join_pairs': checked, 'als_max_equation_residual': max(residuals),
              'verified_submissions': files, 'seconds': time.perf_counter()-started}
    (OUT/'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
