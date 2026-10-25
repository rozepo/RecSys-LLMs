"""Run validation, selection, sensitivity and final retraining. Never submits online."""
import argparse
import importlib.metadata
import itertools
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

from prepare import DATA, OUT, ROOT, prepare
from models import SEED, als, ap5, evaluate, factor_scores, fusion_features, matrix, ppr, ppr_graph, rank, truths


def main(rebuild=False):
    started = time.perf_counter()
    if rebuild or not (OUT/'data_audit.json').exists():
        prepare()
    template = pd.read_csv(DATA/'submission.csv', dtype=str)
    assert list(template.columns) == ['user_id', 'tv_show_id']
    assert len(template) == 1260 and template.user_id.is_unique
    targets = template.user_id.tolist()
    train = pd.read_csv(OUT/'train_interactions.csv.gz', dtype={'user_id': str})
    validation = pd.read_csv(OUT/'validation_interactions.csv.gz', dtype={'user_id': str})
    users = sorted(set(train.user_id) | set(validation.user_id) | set(targets))
    items = np.unique(np.concatenate([np.load(OUT/'item_ids.npy'), np.load(OUT/'final_candidates.npy')]))
    target_indices = np.array([users.index(u) for u in targets])
    candidates = np.isin(items, np.load(OUT/'validation_candidates.npy'))
    truth = truths(validation, targets, items)
    report = {'seed': SEED, 'threshold_seconds': 60, 'environment': {'python': sys.version,
        'platform': platform.platform(), 'machine': platform.machine(), 'cpu_count': os.cpu_count(),
        'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'pandas']},
        'threads': {p: os.environ.get(p) for p in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']}},
        'truth_nonempty_users': {k: sum(bool(t) for t in v) for k, v in truth.items()},
        'ppr_grid': [], 'als_grid': [], 'fusion_grid': [], 'sensitivity': [], 'timings': {}}
    x = matrix(train, users, items, 60)
    history = x[target_indices].toarray()
    popularity = np.asarray(x.sum(axis=0)).ravel()
    popscore = np.broadcast_to(popularity, history.shape)
    cold = history.sum(axis=1) == 0
    report['validation_cold_users'] = int(cold.sum())
    scores = {'popularity': popscore, 'history': history}
    predictions = {}
    results = {}
    for name in scores:
        t = time.perf_counter()
        predictions[name] = rank(scores[name], candidates, popularity, cold)
        results[name] = evaluate(predictions[name], truth)
        report['timings'][name+'_ranking_seconds'] = time.perf_counter()-t
        print(name, results[name], flush=True)
    t = time.perf_counter()
    graph = ppr_graph(x)
    report['timings']['ppr_graph_seconds'] = time.perf_counter()-t
    report['ppr_graph'] = {'nodes': graph.shape[0], 'edges': graph.nnz, 'neighbors': 50}
    best = -1
    for alpha in [.3, .5, .7, .85, .95]:
        t = time.perf_counter()
        score, convergence = ppr(graph, history, alpha)
        pred = rank(score, candidates, popularity, cold)
        metrics = evaluate(pred, truth)
        row = {'alpha': alpha, **metrics, **convergence, 'seconds': time.perf_counter()-t}
        report['ppr_grid'].append(row)
        print('PPR', row, flush=True)
        if metrics['top10'] > best:
            best = metrics['top10']
            scores['ppr'], predictions['ppr'], results['ppr'] = score, pred, metrics
            report['selected_ppr_alpha'] = alpha
    del graph
    best = -1
    grid = itertools.product([16, 32], [1., 10.], ['log', 'seconds'])
    for factors, scale, transform in grid:
        params = {'factors': factors, 'scale': scale, 'regularization': 10., 'transform': transform, 'iterations': 15}
        t = time.perf_counter()
        uf, itf = als(x, **params)
        score = factor_scores(uf[target_indices], itf)
        pred = rank(score, candidates, popularity, cold)
        metrics = evaluate(pred, truth)
        row = {**params, **metrics, 'seconds': time.perf_counter()-t}
        report['als_grid'].append(row)
        print('ALS', row, flush=True)
        if metrics['top10'] > best:
            best = metrics['top10']
            scores['als'], predictions['als'], results['als'] = score, pred, metrics
            report['selected_als'] = params
    t = time.perf_counter()
    features = fusion_features([scores[n] for n in ['history', 'ppr', 'als', 'popularity']], candidates, popularity, cold)
    best = -1
    # Simplex weights on history/PPR/ALS, plus a small popularity component.
    for h in range(5):
        for p in range(5-h):
            m = 4-h-p
            for pop in [0., .02, .05, .1]:
                weights = [(1-pop)*h/4, (1-pop)*p/4, (1-pop)*m/4, pop]
                fused = sum(w*f for w, f in zip(weights, features))
                pred = rank(fused, candidates, popularity, cold)
                metrics = evaluate(pred, truth)
                report['fusion_grid'].append({'weights': weights, **metrics})
                if metrics['top10'] > best:
                    best = metrics['top10']
                    predictions['hybrid'], results['hybrid'] = pred, metrics
                    report['selected_fusion_weights'] = weights
    report['timings']['fusion_grid_seconds'] = time.perf_counter()-t
    report['metrics'] = results
    print('Selected', report['selected_fusion_weights'], results, flush=True)
    # Paired bootstrap uncertainty for selection outcome (descriptive, same hold-out).
    rng = np.random.default_rng(SEED)
    best_single = max(['popularity', 'history', 'ppr', 'als'], key=lambda n: results[n]['top10'])
    report['best_single'] = best_single
    diffs = np.array([ap5(h, tr)-ap5(b, tr) for h, b, tr in
                      zip(predictions['hybrid'], predictions[best_single], truth['top10'])])
    bootstrap = np.array([rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(2000)])
    report['hybrid_minus_best_single_bootstrap95'] = np.quantile(bootstrap, [.025, .975]).tolist()
    # Freeze selected hyperparameters; vary only the watched threshold; truth unchanged.
    for threshold in [1, 300, 1800]:
        t = time.perf_counter()
        xs = matrix(train, users, items, threshold)
        hs = xs[target_indices].toarray()
        ps = np.asarray(xs.sum(axis=0)).ravel()
        cs = hs.sum(axis=1) == 0
        gs = ppr_graph(xs)
        prs, _ = ppr(gs, hs, report['selected_ppr_alpha'])
        del gs
        uf, itf = als(xs, **report['selected_als'])
        ss = {'history': hs, 'ppr': prs, 'als': factor_scores(uf[target_indices], itf),
              'popularity': np.broadcast_to(ps, hs.shape)}
        fs = fusion_features([ss[n] for n in ['history', 'ppr', 'als', 'popularity']], candidates, ps, cs)
        ss['hybrid'] = sum(w*f for w, f in zip(report['selected_fusion_weights'], fs))
        metrics = {n: evaluate(rank(s, candidates, ps, cs), truth) for n, s in ss.items()}
        report['sensitivity'].append({'threshold': threshold, 'cold_users': int(cs.sum()),
            'train_pairs': xs.nnz, 'metrics': metrics, 'seconds': time.perf_counter()-t})
        print('Sensitivity', report['sensitivity'][-1], flush=True)
    # Explain one target at the 90th and 10th percentile of raw training seconds,
    # among targets with usable train and validation truth; no cherry-picking on success.
    raw_totals = train.groupby('user_id').seconds.sum().reindex(targets, fill_value=0)
    usable = [i for i in range(len(targets)) if not cold[i] and truth['top10'][i]]
    ordered = sorted(usable, key=lambda i: (raw_totals.iloc[i], targets[i]))
    examples = []
    for label, quantile in [('heavy', .9), ('light', .1)]:
        idx = ordered[int((len(ordered)-1)*quantile)]
        uid = targets[idx]
        hist = train[train.user_id == uid].sort_values(['seconds', 'tv_show_id'], ascending=[False, True])
        actual = validation[validation.user_id == uid].sort_values(['seconds', 'tv_show_id'], ascending=[False, True])
        hist.to_csv(OUT/f'{label}_history.csv', index=False)
        actual.to_csv(OUT/f'{label}_validation.csv', index=False)
        examples.append({'label': label, 'user_id': uid, 'quantile': quantile,
            'training_seconds': int(raw_totals.iloc[idx]), 'training_items': len(hist),
            'validation_seconds': int(actual.seconds.sum()),
            'history_top10': hist.head(10).to_dict('records'), 'actual_top10': actual.head(10).to_dict('records'),
            'methods': {n: {'ids': items[pr[idx]].tolist(), 'ap5': ap5(pr[idx], truth['top10'][idx]),
                'familiar_ids': [int(items[i]) for i in pr[idx] if history[idx, i] > 0],
                'hits_top10': [int(items[i]) for i in pr[idx] if i in truth['top10'][idx]]}
                for n, pr in predictions.items()}})
    report['examples'] = examples
    np.savez_compressed(OUT/'validation_predictions.npz', user_ids=np.array(targets),
                        **{n: items[p] for n, p in predictions.items()})
    # Save all original model scores for reproducibility/independent inspection.
    np.savez_compressed(OUT/'validation_scores.npz', items=items, user_ids=np.array(targets),
                        **{n: np.asarray(s, dtype=np.float32) for n, s in scores.items()})
    # Final fitting uses weeks 11-30 views only, identical threshold/parameters.
    t = time.perf_counter()
    full = pd.concat([train, validation]).groupby(['user_id', 'tv_show_id'], sort=False).seconds.sum().reset_index()
    xf = matrix(full, users, items, 60)
    hf = xf[target_indices].toarray()
    pf = np.asarray(xf.sum(axis=0)).ravel()
    cf = hf.sum(axis=1) == 0
    gf = ppr_graph(xf)
    prf, conv = ppr(gf, hf, report['selected_ppr_alpha'])
    del gf
    uf, itf = als(xf, **report['selected_als'])
    final_scores = {'history': hf, 'ppr': prf, 'als': factor_scores(uf[target_indices], itf),
                    'popularity': np.broadcast_to(pf, hf.shape)}
    final_mask = np.isin(items, np.load(OUT/'final_candidates.npy'))
    ff = fusion_features([final_scores[n] for n in ['history', 'ppr', 'als', 'popularity']], final_mask, pf, cf)
    final_scores['hybrid'] = sum(w*f for w, f in zip(report['selected_fusion_weights'], ff))
    submissions = ROOT/'submissions'
    submissions.mkdir(exist_ok=True)
    report['submissions'] = []
    for name in dict.fromkeys(['hybrid', best_single, 'popularity', 'history']):
        pred = items[rank(final_scores[name], final_mask, pf, cf)]
        path = submissions/f'{name}.csv'
        pd.DataFrame({'user_id': targets, 'tv_show_id': [' '.join(map(str, row)) for row in pred]}).to_csv(path, index=False)
        saved = pd.read_csv(path, dtype=str)
        assert saved.shape == (1260, 2) and saved.columns.tolist() == template.columns.tolist()
        assert saved.user_id.tolist() == targets and saved.user_id.is_unique
        allowed = set(items[final_mask])
        for cell in saved.tv_show_id:
            ids = list(map(int, cell.split()))
            assert len(ids) == len(set(ids)) == 5 and 0 not in ids and set(ids) <= allowed
        report['submissions'].append({'file': str(path.relative_to(ROOT)), 'rows': len(saved),
                                      'unique_recommended_items': len(set(pred.ravel().tolist())), 'format_verified': True})
    report['final'] = {'cold_users': int(cf.sum()), 'train_pairs': xf.nnz, 'ppr_convergence': conv,
                       'seconds': time.perf_counter()-t}
    report['experiment_seconds'] = time.perf_counter()-started
    (OUT/'experiments.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(results).T.to_csv(OUT/'metrics.csv', index_label='method')
    pd.DataFrame(report['ppr_grid']).to_csv(OUT/'ppr_grid.csv', index=False)
    pd.DataFrame(report['als_grid']).to_csv(OUT/'als_grid.csv', index=False)
    print('Complete:', report['experiment_seconds'], 'seconds', report['submissions'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true')
    main(parser.parse_args().rebuild)
