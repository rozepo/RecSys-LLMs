"""Display saved predictions; no fitting or score selection."""
import json

import numpy as np
import pandas as pd

from prepare import DATA, END, OUT, ROOT, seconds


def load_examples():
    folder = OUT/'iteration3'
    if not (folder/'kaggle_reported.json').exists():
        return {}
    experiments = json.loads((folder/'experiments.json').read_text())
    chosen = {s['file'].split('/')[-1].removesuffix('.csv'):s['config']['id'] for s in experiments['selected']}
    original = json.loads((OUT/'experiments.json').read_text())
    users = [e['user_id'] for e in original['examples']]
    result = {u:{} for u in users}
    with np.load(folder/'old_predictions.npz') as saved:
        order = saved['user_ids'].tolist()
        for method in ['i3_context_top3','i3_context_pure']:
            final = pd.read_csv(ROOT/'submissions'/f'{method}.csv',dtype=str).set_index('user_id')
            for u in users:
                validation = saved[chosen[method]][order.index(u)].tolist()
                predicted = list(map(int,final.loc[u,'tv_show_id'].split()))
                assert len(set(validation)) == len(set(predicted)) == 5
                result[u][method] = {'validation':validation,'final':predicted}
    # Test-schedule metadata is read exclusively for displaying final examples.
    schedule = pd.read_csv(DATA/'export_arh_31-42-final.csv',usecols=[
        'tv_show_id','tv_show_title','tv_show_category','start_time','duration'])
    starts = seconds(schedule.start_time,'%d.%m.%Y %H:%M:%S')
    schedule = schedule[(schedule.tv_show_id != 0) & (schedule.duration > 0) &
                        (starts < END+12*7*86400) & (starts+schedule.duration > END)].copy()
    metadata = {int(r.tv_show_id):{'title':r.tv_show_title,'category':r.tv_show_category}
                for r in schedule.drop_duplicates('tv_show_id').itertuples()}
    for methods in result.values():
        for record in methods.values():
            record['final_metadata'] = {i:metadata[i] for i in record['final']}
    return result
