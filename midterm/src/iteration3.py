"""Last, public-score-directed experiment; weights selected on weeks 19-30 only."""
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse

from iteration2 import aggregate, schedule_window
from iteration2_data import I2, WEEK, load_schedule
from models import als, evaluate, factor_scores, fusion_features, matrix, ppr, ppr_graph, rank, truths
from prepare import BEGIN, DATA, END, OUT, ROOT, seconds

I3 = OUT/'iteration3'
SMOOTHING = [0., .05, .1, .2, .4, .6]
BASES = ['time_rrf', 'count_rrf', 'time_channel', 'count_channel', 'category',
         'time_product', 'count_product', 'time_joint', 'count_joint']


def prepare_counts(targets):
    """Count ORIGINAL view-log records, not programmes or weekly fragments."""
    start = time.perf_counter()
    parts = []
    total = 0
    independent = {18:0, 20:0, 30:0}
    for frame in pd.read_csv(DATA/'dataset11-30.csv', dtype={'user_id':str}, chunksize=300000):
        frame = frame[frame.user_id.isin(targets)].copy()
        s = seconds(frame.start_time, '%Y-%m-%d %H:%M:%S')
        e = seconds(frame.stop_time, '%Y-%m-%d %H:%M:%S')
        valid = (s < END) & (e > BEGIN) & (e > s)
        frame = frame[valid].rename(columns={'vsetv_id':'channel_id'})
        s = np.maximum(s[valid], BEGIN)
        frame['week'] = (s-BEGIN)//WEEK+11
        total += len(frame)
        for cutoff in independent:
            independent[cutoff] += int(np.sum(s < BEGIN+(cutoff-10)*WEEK))
        parts.append(frame.groupby(['user_id','channel_id','week']).size().rename('views').reset_index())
    counts = pd.concat(parts).groupby(['user_id','channel_id','week'],sort=False).views.sum().reset_index()
    assert int(counts.views.sum()) == total
    for cutoff,expected in independent.items():
        assert int(counts.loc[counts.week <= cutoff,'views'].sum()) == expected
    counts.to_csv(I3/'channel_view_counts.csv.gz',index=False)
    audit = {'target_view_records':total,'weekly_count_rows':len(counts),
             'records_before_cutoffs':independent,'seconds':time.perf_counter()-start}
    (I3/'count_audit.json').write_text(json.dumps(audit,indent=2))
    return counts


def normalize_rows(score, candidates):
    score = np.asarray(score,dtype=np.float64).copy()
    score[:,~candidates] = 0
    sums = score.sum(axis=1,keepdims=True)
    score /= np.maximum(sums,1e-30)
    return score


def context_variants(train_weekly, channel_weekly, counts, train_schedule, target_schedule, train, targets, items,
                     candidates, popularity, cold):
    uim = {u:i for i,u in enumerate(targets)}
    iim = {int(i):k for k,i in enumerate(items)}
    eligible = train[train.seconds >= 60]
    labelled = train_weekly.merge(eligible[['user_id','tv_show_id']],on=['user_id','tv_show_id'])
    cats = train_schedule.groupby(['tv_show_id','tv_show_category']).seconds.sum().reset_index()
    cats = cats.sort_values(['tv_show_id','seconds','tv_show_category'],ascending=[True,False,True]).drop_duplicates('tv_show_id')
    lookup = dict(zip(cats.tv_show_id,cats.tv_show_category))
    eligible = eligible.assign(category=eligible.tv_show_id.map(lookup))
    labelled = labelled.assign(category=labelled.tv_show_id.map(lookup))
    assert eligible.category.notna().all() and labelled.category.notna().all()
    categories = sorted(eligible.category.unique())
    catim = {c:i for i,c in enumerate(categories)}
    channels = sorted(set(channel_weekly.channel_id)|set(target_schedule.channel_id))
    cim = {c:i for i,c in enumerate(channels)}
    available = target_schedule[['channel_id','tv_show_id']].drop_duplicates()

    def profile(frame,column,values,mapping):
        g = frame[frame.user_id.isin(targets)].groupby(['user_id',column])[values].sum().reset_index()
        m = sparse.csr_matrix((g[values].to_numpy(dtype=float),(g.user_id.map(uim),g[column].map(mapping))),
                              shape=(len(targets),len(mapping)))
        total = np.asarray(m.sum(axis=1)).ravel()
        return (sparse.diags(1/np.maximum(total,1)) @ m).tocsr()

    profiles = {'time':profile(channel_weekly,'channel_id','seconds',cim),
                'count':profile(counts,'channel_id','views',cim)}
    category_profile = profile(eligible,'category','seconds',catim)
    cat_pop = eligible.groupby(['category','tv_show_id']).seconds.sum().reset_index()
    denom = cat_pop.groupby('category').seconds.sum()
    cat_conditional = sparse.csr_matrix((cat_pop.seconds/cat_pop.category.map(denom),
        (cat_pop.category.map(catim),cat_pop.tv_show_id.map(iim))),shape=(len(categories),len(items)))
    category_score = (category_profile @ cat_conditional).toarray()

    ch_pop = labelled.groupby(['channel_id','tv_show_id']).seconds.sum().reset_index()
    denom = ch_pop.groupby('channel_id').seconds.sum()
    ch_pop['weight'] = ch_pop.seconds/ch_pop.channel_id.map(denom)
    ch_pop = ch_pop.merge(available,on=['channel_id','tv_show_id'])
    ch_conditional = sparse.csr_matrix((ch_pop.weight,(ch_pop.channel_id.map(cim),ch_pop.tv_show_id.map(iim))),
                                     shape=(len(channels),len(items)))
    raw = {'category':category_score}
    for mode in ['time','count']:
        channel_score = (profiles[mode] @ ch_conditional).toarray()
        raw[mode+'_channel'] = channel_score
        features = fusion_features([channel_score,category_score],candidates,popularity,cold)
        raw[mode+'_rrf'] = .5*features[0]+.5*features[1]
        # Product-of-experts variant: zero unless both contexts have evidence.
        raw[mode+'_product'] = normalize_rows(channel_score,candidates)*normalize_rows(category_score,candidates)
        raw[mode+'_joint'] = np.zeros_like(category_score)
    # Genuine joint conditional popularity, distinct from multiplying two scores:
    # sum_c q_u(c)*q_u(g_i)*P_train(i | c,g_i)*availability(i,c).
    joint = labelled.groupby(['channel_id','category','tv_show_id']).seconds.sum().reset_index()
    denom = joint.groupby(['channel_id','category']).seconds.transform('sum')
    joint['weight'] = joint.seconds/denom
    joint = joint.merge(available,on=['channel_id','tv_show_id'])
    for category,g in joint.groupby('category'):
        conditional = sparse.csr_matrix((g.weight,(g.channel_id.map(cim),g.tv_show_id.map(iim))),
                                       shape=(len(channels),len(items)))
        q = category_profile[:,catim[category]].toarray().ravel()
        for mode in ['time','count']:
            raw[mode+'_joint'] += (sparse.diags(q) @ profiles[mode] @ conditional).toarray()
    # Invariant checks catch channel-profile counting or category wiring mistakes.
    for mode,p in profiles.items():
        total = np.asarray(p.sum(axis=1)).ravel()
        assert np.all(np.isclose(total,0)|np.isclose(total,1)), mode
    assert all(np.isfinite(s).all() and (s >= 0).all() for s in raw.values())
    return raw


def configurations():
    mixtures = [{}]
    for component in ['history','ppr','als']:
        for weight in [.025,.05,.1,.15,.2]:
            mixtures.append({component:weight})
    mixtures.extend([{'history':.05,'ppr':.05},{'history':.1,'ppr':.1},
                     {'history':.05,'als':.05},{'ppr':.05,'als':.05},
                     {'history':.05,'ppr':.05,'als':.05}])
    configs = []
    for base in BASES:
        for smoothing in SMOOTHING:
            for mix in mixtures:
                configs.append({'id':f'cfg_{len(configs):04d}','base':base,'global_smoothing':smoothing,
                                'cf_weights':mix,'cf_total':round(sum(mix.values()),8)})
    return configs


def fit_components(weekly,channels,counts,schedule,cutoff_week,items,users,targets,original,final=False):
    start = time.perf_counter()
    cutoff = BEGIN+(cutoff_week-10)*WEEK
    target_source = load_schedule(test=True) if final else schedule
    target,air_count,_ = schedule_window(target_source,cutoff,END+12*WEEK if final else END,items)
    train_schedule,_,_ = schedule_window(schedule,BEGIN,cutoff,items)
    tw = weekly[weekly.week <= cutoff_week]
    train = aggregate(tw)
    x = matrix(train,users,items,60)
    u = {uid:i for i,uid in enumerate(users)}
    ti = np.array([u[uid] for uid in targets])
    history = x[ti].toarray()
    pop = np.asarray(x.sum(axis=0)).ravel()
    candidates = air_count > 0
    cold = history.sum(axis=1) == 0
    graph = ppr_graph(x)
    pagerank,convergence = ppr(graph,history,original['selected_ppr_alpha'])
    del graph
    uf,itf = als(x,**original['selected_als'])
    mf = factor_scores(uf[ti],itf)
    cf_features = dict(zip(['history','ppr','als'],fusion_features([history,pagerank,mf],candidates,pop,cold)))
    raw = context_variants(tw,channels[channels.week <= cutoff_week],counts[counts.week <= cutoff_week],
                           train_schedule,target,train,targets,items,candidates,pop,cold)
    legacy = rank(raw['time_rrf'],candidates,pop,cold)
    if not final:
        name = 'new' if cutoff_week == 18 else 'old'
        with np.load(I2/f'{name}_predictions.npz') as saved:
            assert saved['user_ids'].tolist() == targets
            assert np.array_equal(items[legacy],saved['channel_category'])
            for n,score in [('history',history),('ppr',pagerank),('als',mf)]:
                assert np.array_equal(items[rank(score,candidates,pop,cold)],saved[n]), n
    else:
        old = pd.read_csv(ROOT/'submissions'/'i2_context_pop.csv',dtype=str)
        assert old.tv_show_id.tolist() == [' '.join(map(str,row)) for row in items[legacy]]
    truth = None if final else truths(aggregate(weekly[weekly.week > cutoff_week]),targets,items)
    state = {'raw':raw,'features':cf_features,'popularity':pop,'candidates':candidates,'cold':cold,
             'legacy':legacy,'truth':truth,
             'summary':{'cutoff_week':cutoff_week,'train_pairs':x.nnz,'cold_users':int(cold.sum()),
                        'candidates':int(candidates.sum()),'ppr_convergence':convergence,
                        'fit_seconds':time.perf_counter()-start}}
    print('Iteration 3 fit:',state['summary'],flush=True)
    return state


def base_score(state,config):
    base = normalize_rows(state['raw'][config['base']],state['candidates'])
    p = state['popularity']*state['candidates']
    p /= max(p.sum(),1)
    lam = config['global_smoothing']
    return (1-lam)*base+lam*p[None,:]


def predict(state,config,feature=None):
    if not config['cf_weights']:
        return rank(base_score(state,config),state['candidates'],state['popularity'],state['cold'])
    if feature is None:
        feature = fusion_features([base_score(state,config)],state['candidates'],state['popularity'],state['cold'])[0]
    score = (1-config['cf_total'])*feature
    for name,w in config['cf_weights'].items():
        score += w*state['features'][name]
    return rank(score,state['candidates'],state['popularity'],state['cold'])


def evaluate_window(state,configs,name,items,targets):
    start = time.perf_counter()
    records,preds = [],{}
    last = None
    feature = None
    fallback = rank(np.broadcast_to(state['popularity'],(len(targets),len(items))),state['candidates'],state['popularity'],state['cold'])
    for c in configs:
        key = (c['base'],c['global_smoothing'])
        if key != last:
            feature = fusion_features([base_score(state,c)],state['candidates'],state['popularity'],state['cold'])[0]
            last = key
        pr = predict(state,c,feature)
        assert np.array_equal(pr[state['cold']],fallback[state['cold']])
        metrics = evaluate(pr,state['truth'])
        records.append({**c,**metrics})
        preds[c['id']] = pr
    np.savez_compressed(I3/f'{name}_predictions.npz',user_ids=np.array(targets),
                        **{cid:items[p] for cid,p in preds.items()})
    result = {'records':records,'predictions':preds,'summary':state['summary'],
              'evaluation_seconds':time.perf_counter()-start,
              'legacy_metrics':evaluate(state['legacy'],state['truth'])}
    print(name,'best:',max(records,key=lambda r:r['top10']),'seconds:',result['evaluation_seconds'],flush=True)
    return result


def digest(prediction):
    return hashlib.sha256(np.asarray(prediction,dtype=np.int64).tobytes()).hexdigest()


def run(rebuild_counts=False):
    start = time.perf_counter()
    I3.mkdir(exist_ok=True)
    template = pd.read_csv(DATA/'submission.csv',dtype=str)
    assert template.shape == (1260,2) and template.user_id.is_unique
    targets = template.user_id.tolist()
    counts = prepare_counts(targets) if rebuild_counts or not (I3/'count_audit.json').exists() else pd.read_csv(I3/'channel_view_counts.csv.gz',dtype={'user_id':str})
    weekly = pd.read_csv(I2/'weekly_interactions.csv.gz',dtype={'user_id':str})
    channels = pd.read_csv(I2/'weekly_channels.csv.gz',dtype={'user_id':str})
    users = sorted(set(weekly.user_id)|set(targets))
    items = np.unique(np.r_[np.load(OUT/'item_ids.npy'),np.load(OUT/'final_candidates.npy')])
    original = json.loads((OUT/'experiments.json').read_text())
    schedule = load_schedule()
    configs = configurations()
    outcomes = {}
    for name,cutoff in [('new',18),('old',20)]:
        state = fit_components(weekly,channels,counts,schedule,cutoff,items,users,targets,original)
        outcomes[name] = evaluate_window(state,configs,name,items,targets)
        del state
    rows = []
    for new,old in zip(outcomes['new']['records'],outcomes['old']['records']):
        assert new['id'] == old['id']
        rows.append({**{k:v for k,v in new.items() if k not in ['top10','top5','ge1800']},
                     **{f'{split}_{metric}':row[metric] for split,row in [('new',new),('old',old)]
                        for metric in ['top10','top5','ge1800']}})
    pd.DataFrame([{**r,'cf_weights':json.dumps(r['cf_weights'])} for r in rows]).to_csv(I3/'comparison.csv',index=False)
    pd.DataFrame([{**r,'cf_weights':json.dumps(r['cf_weights'])} for r in rows if r['cf_total']==0]).to_csv(I3/'base_grid.csv',index=False)
    pure = max([r for r in rows if r['cf_total']==0 and r['global_smoothing']==0],key=lambda r:r['new_top10'])
    # Pure control plus the top THREE validation-ranked distinct alternatives.
    # Test predictions only enforce diversity; no final test behaviour is known.
    alternatives = sorted([r for r in rows if r['cf_total']>0 or r['global_smoothing']>0],key=lambda r:-r['new_top10'])
    final = fit_components(weekly,channels,counts,schedule,30,items,users,targets,original,final=True)
    old_files = {p.name:digest(np.array([list(map(int,v.split())) for v in pd.read_csv(p,dtype=str).tv_show_id]))
                 for p in (ROOT/'submissions').glob('*.csv') if not p.name.startswith('i3_')}
    selected = []
    seen_final = set()
    seen_validation = set()
    for c in [pure]+alternatives:
        final_pred = items[predict(final,c)]
        checksum = digest(final_pred)
        val_checksum = digest(items[outcomes['new']['predictions'][c['id']]])
        is_pure = len(selected)==0
        duplicates = [n for n,h in old_files.items() if h==checksum]
        if not is_pure and (checksum in seen_final or checksum in old_files.values() or val_checksum in seen_validation):
            continue
        name = 'i3_context_pure' if is_pure else f'i3_context_top{len(selected)}'
        path = ROOT/'submissions'/f'{name}.csv'
        pd.DataFrame({'user_id':targets,'tv_show_id':[' '.join(map(str,r)) for r in final_pred]}).to_csv(path,index=False)
        saved = pd.read_csv(path,dtype=str)
        assert saved.shape == template.shape and saved.columns.tolist() == template.columns.tolist()
        assert saved.user_id.tolist() == targets and saved.user_id.is_unique
        assert len(path.read_text().splitlines())==1261
        allowed = set(items[final['candidates']])
        for value in saved.tv_show_id:
            tokens = value.split(' ')
            assert len(tokens)==len(set(tokens))==5 and all(t.isdigit() for t in tokens)
            assert '0' not in tokens and set(map(int,tokens)) <= allowed
        assert np.array_equal(final_pred[final['cold']],items[final['legacy']][final['cold']])
        previous = pd.read_csv(ROOT/'submissions'/'i2_context_pop.csv',dtype=str)
        changed = int(np.sum(saved.tv_show_id.to_numpy()!=previous.tv_show_id.to_numpy()))
        selected.append({'file':str(path.relative_to(ROOT)),'config':c,'is_pure_control':is_pure,
                         'changed_users_vs_i2_context':changed,'unique_items':len(np.unique(final_pred)),
                         'rows':len(saved),'format_verified':True,'sha256':checksum,'duplicates_previous_files':duplicates})
        seen_final.add(checksum)
        seen_validation.add(val_checksum)
        if len(selected)==4:
            break
    assert 1 <= len(selected) <= 4
    exp = {'direction_source':'Iteration 2 USER-REPORTED PUBLIC score: i2_context_pop=0.34906; adaptive family selection',
           'weight_selection':'MAP@5 top10 train11-18/validation19-30 only; old split diagnostic',
           'base_names':BASES,'global_smoothing_grid':SMOOTHING,'configurations':len(configs),
           'pure_configurations':len(BASES),'smoothing_configurations':len(BASES)*len(SMOOTHING),
           'cf_weight_cap':.2,'selected':selected,
           'windows':{n:{'summary':v['summary'],'evaluation_seconds':v['evaluation_seconds'],
                         'legacy_i2_context_metrics':v['legacy_metrics']} for n,v in outcomes.items()},
           'final':final['summary'],
           'verification':{'i2_context_reproduction':'exact predictions on both validation windows and final',
                           'history_ppr_als_reproduction':'exact predictions on both validation windows',
                           'cold_fallback':'checked for every grid prediction on both windows and all exported final files',
                           'csv_format':'passed', 'old_submissions':'not modified'},
           'seconds':time.perf_counter()-start}
    (I3/'experiments.json').write_text(json.dumps(exp,ensure_ascii=False,indent=2))
    print('Iteration 3 complete:',json.dumps(exp,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild-counts',action='store_true')
    run(parser.parse_args().rebuild_counts)
