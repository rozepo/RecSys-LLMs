"""Second experiment: 12-week hold-out and schedule-aware popularity fusion.

No network operations. Original artifacts and submissions are left intact.
"""
import argparse
import hashlib
import itertools
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse, stats

from iteration2_data import I2, WEEK, load_schedule, prepare_weekly
from models import SEED, als, ap5, evaluate, factor_scores, fusion_features, matrix, ppr, ppr_graph, rank, truths
from prepare import BEGIN, DATA, END, OUT, ROOT


def aggregate(frame):
    return frame.groupby(['user_id','tv_show_id'], sort=False).seconds.sum().reset_index()


def schedule_window(schedule, lo, hi, items):
    frame = schedule[(schedule.s < hi) & (schedule.e > lo)].copy()
    frame['seconds'] = np.minimum(frame.e, hi)-np.maximum(frame.s, lo)
    grouped = frame.groupby('tv_show_id')
    counts = grouped.size().reindex(items, fill_value=0).to_numpy(dtype=float)
    duration = grouped.seconds.sum().reindex(items, fill_value=0).to_numpy(dtype=float)
    return frame, counts, duration


def context_scores(train_weekly, channel_weekly, train_schedule, target_schedule, targets, items, train):
    """Within-channel/category conditional popularity, user profiles from train only."""
    uim = {u:i for i,u in enumerate(targets)}
    iim = {int(v):i for i,v in enumerate(items)}
    eligible = train.loc[train.seconds >= 60, ['user_id','tv_show_id']]
    labelled = train_weekly.merge(eligible, on=['user_id','tv_show_id'], how='inner')
    channels = sorted(set(channel_weekly.channel_id) | set(target_schedule.channel_id))
    cim = {c:i for i,c in enumerate(channels)}
    user_channel = channel_weekly[channel_weekly.user_id.isin(targets)].groupby(['user_id','channel_id']).seconds.sum().reset_index()
    profile = sparse.csr_matrix((user_channel.seconds.to_numpy(dtype=float),
        (user_channel.user_id.map(uim), user_channel.channel_id.map(cim))), shape=(len(targets),len(channels)))
    sums = np.asarray(profile.sum(axis=1)).ravel()
    profile = sparse.diags(1/np.maximum(sums,1)) @ profile
    available = target_schedule[['channel_id','tv_show_id']].drop_duplicates()
    pop = labelled.groupby(['channel_id','tv_show_id']).seconds.sum().reset_index()
    denom = pop.groupby('channel_id').seconds.sum()
    pop['conditional'] = pop.seconds/pop.channel_id.map(denom)
    pop = pop.merge(available, on=['channel_id','tv_show_id'])
    conditional = sparse.csr_matrix((pop.conditional,
        (pop.channel_id.map(cim), pop.tv_show_id.map(iim))), shape=(len(channels),len(items)))
    channel_score = (profile @ conditional).toarray()
    # Category lookup is derived only from pre-cut schedule rows. Pick largest
    # scheduled duration for id, deterministic alphabetical ties.
    cats = train_schedule.groupby(['tv_show_id','tv_show_category']).seconds.sum().reset_index()
    cats = cats.sort_values(['tv_show_id','seconds','tv_show_category'], ascending=[True,False,True]).drop_duplicates('tv_show_id')
    itemcat = dict(zip(cats.tv_show_id,cats.tv_show_category))
    cat_train = train[train.seconds >= 60].copy()
    cat_train['category'] = cat_train.tv_show_id.map(itemcat)
    assert cat_train.category.notna().all()
    cats = sorted(cat_train.category.unique())
    catim = {c:i for i,c in enumerate(cats)}
    profile = cat_train[cat_train.user_id.isin(targets)].groupby(['user_id','category']).seconds.sum().reset_index()
    ucat = sparse.csr_matrix((profile.seconds.to_numpy(dtype=float),
        (profile.user_id.map(uim),profile.category.map(catim))), shape=(len(targets),len(cats)))
    denom = np.asarray(ucat.sum(axis=1)).ravel()
    ucat = sparse.diags(1/np.maximum(denom,1)) @ ucat
    pop = cat_train.groupby(['category','tv_show_id']).seconds.sum().reset_index()
    denom = pop.groupby('category').seconds.sum()
    conditional = sparse.csr_matrix((pop.seconds/pop.category.map(denom),
        (pop.category.map(catim),pop.tv_show_id.map(iim))), shape=(len(cats),len(items)))
    category_score = (ucat @ conditional).toarray()
    return channel_score, category_score


def fit_window(weekly, channel_weekly, schedule, cutoff_week, items, users, targets, original, final=False):
    start = time.perf_counter()
    train_weekly = weekly[weekly.week <= cutoff_week]
    train = aggregate(train_weekly)
    cutoff = BEGIN+(cutoff_week-10)*WEEK
    hi = END+12*WEEK if final else END
    target_source = load_schedule(test=True) if final else schedule
    target, air_count, air_duration = schedule_window(target_source, cutoff, hi, items)
    training_schedule, _, training_duration = schedule_window(schedule, BEGIN, cutoff, items)
    candidates = air_count > 0
    x = matrix(train, users, items, 60)
    ui = {u:i for i,u in enumerate(users)}
    ti = np.array([ui[u] for u in targets])
    history = x[ti].toarray()
    popularity = np.asarray(x.sum(axis=0)).ravel()
    cold = history.sum(axis=1) == 0
    raw = {'popularity':np.broadcast_to(popularity, history.shape), 'history':history}
    graph = ppr_graph(x)
    raw['ppr'], convergence = ppr(graph, history, original['selected_ppr_alpha'])
    del graph
    uf,itf = als(x, **original['selected_als'])
    raw['als'] = factor_scores(uf[ti], itf)
    legacy = fusion_features([raw[n] for n in ['history','ppr','als','popularity']], candidates, popularity, cold)
    raw['legacy_hybrid'] = sum(w*f for w,f in zip(original['selected_fusion_weights'],legacy))
    del legacy
    for weeks in [2,4,6]:
        recent = aggregate(train_weekly[train_weekly.week > cutoff_week-weeks])
        pop = recent[recent.seconds >= 60].groupby('tv_show_id').seconds.sum().reindex(items,fill_value=0).to_numpy(dtype=float)
        raw[f'pop_recent_{weeks}w'] = np.broadcast_to(pop, history.shape)
    raw['air_count'] = np.broadcast_to(air_count, history.shape)
    raw['air_duration'] = np.broadcast_to(air_duration, history.shape)
    # Opportunity adjustment: empirical demand per training scheduled second,
    # extrapolated to target broadcast seconds. Zero-demand items stay zero.
    # Clip denominator at one hour to avoid explosive ratios for boundary slivers.
    for power in [.25,.5,1.]:
        for basis,exposure in [('duration',air_duration),('count',air_count)]:
            raw[f'pop_x_{basis}_{power:g}'] = np.broadcast_to(popularity*np.power(exposure, power),history.shape)
        forecast = popularity*np.power(air_duration/np.maximum(training_duration,3600), power)
        raw[f'pop_air_rate_{power:g}'] = np.broadcast_to(forecast, history.shape)
    channel_score,cat_score = context_scores(train_weekly, channel_weekly[channel_weekly.week <= cutoff_week],
                                            training_schedule,target,targets,items,train)
    raw['channel_pop'],raw['category_pop'] = channel_score,cat_score
    contextual = fusion_features([channel_score,cat_score],candidates,popularity,cold)
    raw['channel_category'] = .5*contextual[0]+.5*contextual[1]
    predictions = {n:rank(s,candidates,popularity,cold) for n,s in raw.items()}
    truth = None if final else truths(aggregate(weekly[weekly.week > cutoff_week]),targets,items)
    metrics = {} if final else {n:evaluate(p,truth) for n,p in predictions.items()}
    # Pre-compute small rank-fusion banks once, no blending raw score scales.
    feature_names = ['history','ppr','als','popularity','legacy_hybrid','pop_recent_2w','pop_recent_4w',
                     'pop_recent_6w','channel_pop','category_pop','channel_category','air_count','air_duration']
    features = dict(zip(feature_names,fusion_features([raw[n] for n in feature_names],candidates,popularity,cold)))
    for name,score in raw.items():
        assert np.isfinite(score).all(), name
    summary = {'cutoff_week':cutoff_week,'cutoff':pd.Timestamp(cutoff,unit='s').isoformat(),
               'target_end':pd.Timestamp(hi,unit='s').isoformat(),'candidates':int(candidates.sum()),
               'train_pairs':x.nnz,'cold_users':int(cold.sum()),'ppr_convergence':convergence,
               'target_broadcasts':len(target),'target_broadcast_seconds':int(air_duration.sum()),
               'truth_nonempty':None if final else {k:sum(bool(t) for t in v) for k,v in truth.items()},
               'fit_seconds':time.perf_counter()-start}
    summary['target_items_seen_in_train'] = int(np.sum(candidates & (popularity > 0)))
    summary['target_items_unseen_in_train'] = int(np.sum(candidates & (popularity == 0)))
    summary['target_user_history_seconds_retained'] = float(history[:,candidates].sum()/history.sum())
    summary['training_view_seconds_all_channels'] = int(channel_weekly.loc[channel_weekly.week <= cutoff_week,'seconds'].sum())
    if not final:
        summary['truth_top10_slots'] = sum(len(t) for t in truth['top10'])
        summary['truth_top10_slots_in_personal_history'] = sum(history[u,i] > 0 for u,tr in enumerate(truth['top10']) for i in tr).item()
    print('Fit',summary,flush=True)
    for name,m in metrics.items():
        print(cutoff_week,name,m,flush=True)
    return {'features':features,'predictions':predictions,'metrics':metrics,'truth':truth,
            'candidates':candidates,'popularity':popularity,'cold':cold,'summary':summary}


def configurations(recent):
    configs = []
    seen = set()
    for rec_fraction,personal,pw,exposure,ew in itertools.product(
            [0.,.5,1.], ['history','ppr','legacy_hybrid','channel_pop','category_pop','channel_category'],
            [0.,.25,.5,.75], ['air_count','air_duration'], [0.,.1,.25,.5]):
        if pw+ew > 1:
            continue
        popw = 1-pw-ew
        weights = {'popularity':popw*(1-rec_fraction),recent:popw*rec_fraction,
                   personal:pw,exposure:ew}
        weights = {k:round(v,8) for k,v in weights.items() if v > 0}
        key = tuple(sorted(weights.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append({'id':f'fusion_{len(configs):03d}','weights':weights,'pop_weight':popw,
                        'personal_weight':pw,'exposure_weight':ew})
    return configs


def fuse(state, config):
    return sum(w*state['features'][n] for n,w in config['weights'].items())


def predict_fusion(state, config):
    return rank(fuse(state,config),state['candidates'],state['popularity'],state['cold'])


def run(rebuild=False):
    started = time.perf_counter()
    I2.mkdir(exist_ok=True)
    if rebuild or not (I2/'data_audit.json').exists():
        prepare_weekly()
    original = json.loads((OUT/'experiments.json').read_text())
    weekly = pd.read_csv(I2/'weekly_interactions.csv.gz',dtype={'user_id':str})
    channel_weekly = pd.read_csv(I2/'weekly_channels.csv.gz',dtype={'user_id':str})
    template = pd.read_csv(DATA/'submission.csv',dtype=str)
    targets = template.user_id.tolist()
    users = sorted(set(weekly.user_id)|set(targets))
    items = np.unique(np.r_[np.load(OUT/'item_ids.npy'),np.load(OUT/'final_candidates.npy')])
    schedule = load_schedule()
    newer = fit_window(weekly,channel_weekly,schedule,18,items,users,targets,original)
    older = fit_window(weekly,channel_weekly,schedule,20,items,users,targets,original)
    # Original split should reproduce the previous experiment with frozen params.
    saved = np.load(OUT/'validation_predictions.npz')
    for name in ['popularity','history','ppr','als','hybrid']:
        key = 'legacy_hybrid' if name == 'hybrid' else name
        assert np.array_equal(items[older['predictions'][key]],saved[name]), f'Original reproduction mismatch: {name}'
    recent = max(['pop_recent_2w','pop_recent_4w','pop_recent_6w'],key=lambda n:newer['metrics'][n]['top10'])
    configs = configurations(recent)
    grid = []
    t = time.perf_counter()
    for c in configs:
        row = {**c}
        for key,state in [('new',newer),('old',older)]:
            row[key] = evaluate(predict_fusion(state,c),state['truth'])
        grid.append(row)
    grid_seconds = time.perf_counter()-t
    best = max(grid,key=lambda r:r['new']['top10'])
    highpop = max([r for r in grid if r['pop_weight'] >= .5],key=lambda r:r['new']['top10'])
    highpop_schedule = max([r for r in grid if r['pop_weight'] >= .5 and r['exposure_weight'] > 0],key=lambda r:r['new']['top10'])
    exposure = max([r for r in grid if r['personal_weight'] == 0 and r['exposure_weight'] > 0],key=lambda r:r['new']['top10'])
    contextual = max(['channel_pop','category_pop','channel_category'],key=lambda n:newer['metrics'][n]['top10'])
    exposure_names = [n for n in newer['metrics'] if n.startswith('pop_x_') or n.startswith('pop_air_rate_')]
    best_rate = max(exposure_names,key=lambda n:newer['metrics'][n]['top10'])
    proposed = [('i2_best_hybrid',best),('i2_highpop_hybrid',highpop),('i2_pop_schedule',exposure),
                ('i2_recent_pop',{'method':recent}),('i2_context_pop',{'method':contextual}),
                ('i2_highpop_schedule_hybrid',highpop_schedule)]
    for name,c in proposed:
        for key,state in [('new',newer),('old',older)]:
            pr = state['predictions'][c['method']] if 'method' in c else predict_fusion(state,c)
            state['predictions'][name] = pr
            state['metrics'][name] = evaluate(pr,state['truth'])
    rows = []
    for name in older['metrics']:
        rows.append({'method':name,**{f'{split}_{k}':v for split,state in [('old',older),('new',newer)]
                                      for k,v in state['metrics'][name].items()}})
    pd.DataFrame(rows).to_csv(I2/'comparison.csv',index=False)
    pd.DataFrame([{**{k:v for k,v in r.items() if k not in ['new','old','weights']},
                   'weights':json.dumps(r['weights']),**{f'{s}_{k}':v for s in ['new','old'] for k,v in r[s].items()}}
                  for r in grid]).to_csv(I2/'fusion_grid.csv',index=False)
    for key,state in [('old',older),('new',newer)]:
        np.savez_compressed(I2/f'{key}_predictions.npz',user_ids=np.array(targets),items=items,
                            **{n:items[p] for n,p in state['predictions'].items()})
    kaggle = {'source':'User-reported late submission results; not independently fetched or verified',
              'hybrid':{'private':.27692,'public':.30632},'popularity':{'private':.27447,'public':.28361},
              'ppr':{'private':.25737,'public':.29060},'history':{'private':.24869,'public':.27882}}
    common = ['legacy_hybrid','popularity','ppr','history']
    correlations = {}
    for key,state in [('old',older),('new',newer)]:
        vals = [state['metrics'][n]['top10'] for n in common]
        correlations[key] = {'ordering':sorted(common,key=lambda n:-state['metrics'][n]['top10'])}
        for board in ['public','private']:
            kv = [kaggle['hybrid' if n=='legacy_hybrid' else n][board] for n in common]
            correlations[key][board+'_spearman'] = float(stats.spearmanr(vals,kv).statistic)
            correlations[key][board+'_kendall'] = float(stats.kendalltau(vals,kv).statistic)
    # Final refit. No parameters are selected from either Kaggle score column.
    final = fit_window(weekly,channel_weekly,schedule,30,items,users,targets,original,final=True)
    # Reproduce the four originally submitted final outputs too.
    for name in ['hybrid','popularity','ppr','history']:
        expected = pd.read_csv(ROOT/'submissions'/f'{name}.csv',dtype=str)
        key = 'legacy_hybrid' if name == 'hybrid' else name
        actual = [' '.join(map(str,r)) for r in items[final['predictions'][key]]]
        assert actual == expected.tv_show_id.tolist(), f'Final original mismatch: {name}'
    selected = []
    # Do not spend a submission slot on an already-scored original prediction.
    seen = set()
    for name in ['hybrid','popularity','ppr','history']:
        old = pd.read_csv(ROOT/'submissions'/f'{name}.csv',dtype=str)
        values = np.array([list(map(int,v.split())) for v in old.tv_show_id],dtype=items.dtype)
        seen.add(hashlib.sha256(values.tobytes()).hexdigest())
    for name,c in proposed:
        pred = final['predictions'][c['method']] if 'method' in c else predict_fusion(final,c)
        digest = hashlib.sha256(items[pred].tobytes()).hexdigest()
        if digest in seen:
            print('Omit duplicate final list',name,flush=True)
            continue
        seen.add(digest)
        path = ROOT/'submissions'/f'{name}.csv'
        result = pd.DataFrame({'user_id':targets,'tv_show_id':[' '.join(map(str,r)) for r in items[pred]]})
        result.to_csv(path,index=False)
        # Independent read-back checks on every newly generated file.
        readback = pd.read_csv(path,dtype=str)
        assert readback.shape == template.shape == (1260,2)
        assert readback.columns.tolist() == template.columns.tolist()
        assert readback.user_id.tolist() == targets and readback.user_id.is_unique
        assert len(path.read_text().splitlines()) == 1261
        allowed = set(items[final['candidates']])
        for cell in readback.tv_show_id:
            tokens = cell.split(' ')
            assert len(tokens) == len(set(tokens)) == 5 and all(t.isdigit() for t in tokens)
            assert set(map(int,tokens)) <= allowed and '0' not in tokens
        for state in [older,newer,final]:
            pr = state['predictions'][c['method']] if 'method' in c else predict_fusion(state,c)
            assert np.array_equal(pr[state['cold']],state['predictions']['popularity'][state['cold']])
        oldcells = pd.read_csv(ROOT/'submissions'/'hybrid.csv',dtype=str).tv_show_id.tolist()
        changed = sum(a!=b for a,b in zip(result.tv_show_id,oldcells))
        selected.append({'file':str(path.relative_to(ROOT)),'name':name,'config':c,
                         'old':older['metrics'][name],'new':newer['metrics'][name],
                         'changed_users_vs_original_hybrid':changed,'unique_recommended_ids':len(np.unique(items[pred])),
                         'rows':len(readback),'verified':True,'prediction_sha256':digest})
    diffs = np.array([ap5(a,t)-ap5(b,t) for a,b,t in zip(newer['predictions']['i2_best_hybrid'],
        newer['predictions']['legacy_hybrid'],newer['truth']['top10'])])
    rng = np.random.default_rng(SEED)
    bootstrap = [rng.choice(diffs,len(diffs),replace=True).mean() for _ in range(2000)]
    summary = {'primary_selection':'MAP@5 top10 on train11-18/validation19-30 ONLY',
               'original_parameters_frozen':{'ppr_alpha':original['selected_ppr_alpha'],'als':original['selected_als'],
                                              'hybrid_weights':original['selected_fusion_weights']},
               'recent_selected':recent,'grid_configurations':len(grid),'grid_seconds':grid_seconds,
               'best_exposure_adjustment':best_rate,
               'best_vs_legacy_new_paired_bootstrap95':np.quantile(bootstrap,[.025,.975]).tolist(),
               'kaggle_reported':kaggle,'ranking_comparison':correlations,
               'windows':{key:state['summary'] for key,state in [('old',older),('new',newer),('final',final)]},
               'selected':selected,'old_metrics':older['metrics'],'new_metrics':newer['metrics'],
               'verification':{'weekly_vs_original_join':'exact all-pair equality',
                               'old_validation_predictions':'exact all-user equality, five methods',
                               'original_final_submissions':'exact all-user equality, four files',
                               'new_csv_format_and_cold_fallback':'passed'},
               'seconds':time.perf_counter()-started}
    (I2/'experiments.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Iteration 2 complete',json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild',action='store_true')
    run(parser.parse_args().rebuild)
