"""Exact interval join; no future viewing or test metadata enters features."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'results'
SPLIT = pd.Timestamp('2020-05-18').value // 10**9
BEGIN = pd.Timestamp('2020-03-09').value // 10**9
END = pd.Timestamp('2020-07-27').value // 10**9


def seconds(series, fmt):
    return pd.to_datetime(series, format=fmt).astype('int64').to_numpy() // 10**9


def covered_integral(starts, ends):
    """Integral of the union indicator, avoiding double-counted airtime."""
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    if not len(starts):
        return lambda x: np.zeros(len(x), dtype=np.int64)
    previous = np.maximum.accumulate(ends)
    new = np.r_[True, starts[1:] > previous[:-1]]
    indices = np.flatnonzero(new)
    left = starts[indices]
    right = np.maximum.reduceat(ends, indices)
    cumulative = np.r_[0, np.cumsum(right-left)]

    def integral(x):
        idx = np.searchsorted(left, x, side='right') - 1
        safe = np.maximum(idx, 0)
        return np.where(idx >= 0, cumulative[safe] + np.clip(x-left[safe], 0, right[safe]-left[safe]), 0)
    return integral


def prepare():
    OUT.mkdir(exist_ok=True)
    started = time.perf_counter()
    audit = {'schedule': {}, 'windows': {k: {'view_rows_clipped': 0, 'view_seconds': 0,
             'covered_seconds': 0, 'labelled_seconds': 0, 'zero_seconds': 0,
             'intersection_pairs': 0, 'labelled_pairs': 0, 'interaction_seconds_raw': 0}
             for k in ['train', 'validation']}}
    schedules = []
    for period in ['11-20', '21-30']:
        name = f'export_arh_{period}-final.csv'
        df = pd.read_csv(DATA/name, keep_default_na=False)
        audit['schedule'][name] = {'rows': len(df), 'labelled_rows': int((df.tv_show_id != 0).sum()),
            'programmes': int(df.loc[df.tv_show_id != 0, 'tv_show_id'].nunique()),
            'channels': int(df.channel_id.nunique())}
        df['s'] = seconds(df.start_time, '%d.%m.%Y %H:%M:%S')
        df['e'] = df.s + df.duration
        schedules.append(df)
    schedule = pd.concat(schedules, ignore_index=True)
    # One broadcast identity; differing descriptions must not duplicate interactions.
    identity = ['channel_id', 's', 'e', 'tv_show_id']
    before = len(schedule)
    schedule = schedule.drop_duplicates(identity).sort_values(['channel_id', 's', 'e', 'tv_show_id'])
    audit['schedule_duplicates_removed'] = before-len(schedule)
    audit['schedule_nonpositive_duration'] = int((schedule.e <= schedule.s).sum())
    schedule = schedule[schedule.e > schedule.s]
    audit['airtime'] = {}
    for key, lo, hi in [('train', BEGIN, SPLIT), ('validation', SPLIT, END)]:
        window = schedule[(schedule.s < hi) & (schedule.e > lo)].copy()
        window['s'] = np.maximum(window.s, lo)
        window['e'] = np.minimum(window.e, hi)
        total = labelled = 0
        for _, g in window.groupby('channel_id'):
            s, e = g.s.to_numpy(), g.e.to_numpy()
            total += int(covered_integral(s, e)(np.array([hi]))[0])
            mask = g.tv_show_id.to_numpy() != 0
            labelled += int(covered_integral(s[mask], e[mask])(np.array([hi]))[0])
        audit['airtime'][key] = {'union_channel_seconds': total, 'labelled_union_channel_seconds': labelled,
                                 'labelled_fraction': labelled/total}
    metadata = schedule[schedule.tv_show_id != 0].drop_duplicates('tv_show_id')
    metadata[['tv_show_id', 'tv_show_title', 'tv_show_category']].to_json(OUT/'metadata.json', orient='records', force_ascii=False)
    item_ids = np.sort(metadata.tv_show_id.unique())
    np.save(OUT/'item_ids.npy', item_ids)
    # Test schedule is read ONLY for the eligible id set and schedule accounting.
    test = pd.read_csv(DATA/'export_arh_31-42-final.csv', usecols=['channel_id', 'start_time', 'duration', 'tv_show_id'])
    ts = seconds(test.start_time, '%d.%m.%Y %H:%M:%S')
    test_end = pd.Timestamp('2020-10-19').value // 10**9
    eligible = test.loc[(test.tv_show_id != 0) & (test.duration > 0) & (ts < test_end) & (ts+test.duration > END), 'tv_show_id'].unique()
    np.save(OUT/'final_candidates.npy', np.sort(eligible))
    audit['schedule']['export_arh_31-42-final.csv'] = {'rows': len(test),
        'labelled_rows': int((test.tv_show_id != 0).sum()), 'programmes': int(test.loc[test.tv_show_id != 0, 'tv_show_id'].nunique()),
        'channels': int(test.channel_id.nunique())}
    audit['final_candidates'] = len(eligible)
    del test
    val = schedule[(schedule.s < END) & (schedule.e > SPLIT) & (schedule.tv_show_id != 0)]
    np.save(OUT/'validation_candidates.npy', np.sort(val.tv_show_id.unique()))
    audit['validation_candidates'] = int(val.tv_show_id.nunique())
    groups = {}
    for channel, g in schedule.groupby('channel_id'):
        s, e, ids = g.s.to_numpy(), g.e.to_numpy(), g.tv_show_id.to_numpy()
        groups[channel] = (s, e, ids, np.maximum.accumulate(e),
                           [covered_integral(s[mask], e[mask]) for mask in
                            [np.ones(len(s), dtype=bool), ids != 0, ids == 0]])
    del schedule, schedules
    users = set()
    duration_samples = []
    parts = {'train': [], 'validation': []}
    counts = {'rows': 0, 'nonpositive_intervals': 0, 'duration_mismatch': 0, 'outside_window_rows': 0}
    for chunk in pd.read_csv(DATA/'dataset11-30.csv', dtype={'user_id': str}, chunksize=250000):
        counts['rows'] += len(chunk)
        users.update(chunk.user_id)
        s = seconds(chunk.start_time, '%Y-%m-%d %H:%M:%S')
        e = seconds(chunk.stop_time, '%Y-%m-%d %H:%M:%S')
        counts['duration_mismatch'] += int(np.sum(e-s != chunk.duraton.to_numpy()))
        counts['nonpositive_intervals'] += int(np.sum(e <= s))
        counts['outside_window_rows'] += int(np.sum((e <= BEGIN) | (s >= END)))
        duration_samples.append(e-s)
        chunk['s'], chunk['e'] = s, e
        for key, lo, hi in [('train', BEGIN, SPLIT), ('validation', SPLIT, END)]:
            selected = chunk[(chunk.s < hi) & (chunk.e > lo) & (chunk.e > chunk.s)].copy()
            selected['s'] = np.maximum(selected.s, lo)
            selected['e'] = np.minimum(selected.e, hi)
            stats = audit['windows'][key]
            stats['view_rows_clipped'] += len(selected)
            stats['view_seconds'] += int((selected.e-selected.s).sum())
            for channel, views in selected.groupby('vsetv_id', sort=False):
                if channel not in groups:
                    continue
                ss, ee, ids, prefix_end, integrals = groups[channel]
                vs, ve = views.s.to_numpy(), views.e.to_numpy()
                for name, fn in zip(['covered_seconds', 'labelled_seconds', 'zero_seconds'], integrals):
                    stats[name] += int(np.sum(fn(ve)-fn(vs)))
                left = np.searchsorted(prefix_end, vs, side='right')
                right = np.searchsorted(ss, ve, side='left')
                lengths = np.maximum(right-left, 0)
                vi = np.repeat(np.arange(len(views)), lengths)
                if not len(vi):
                    continue
                offsets = np.repeat(np.cumsum(lengths)-lengths, lengths)
                si = np.repeat(left, lengths) + np.arange(len(vi)) - offsets
                overlap = np.minimum(ve[vi], ee[si])-np.maximum(vs[vi], ss[si])
                positive = overlap > 0
                stats['intersection_pairs'] += int(positive.sum())
                valid = positive & (ids[si] != 0)
                stats['labelled_pairs'] += int(valid.sum())
                stats['interaction_seconds_raw'] += int(overlap[valid].sum())
                if valid.any():
                    records = pd.DataFrame({'user_id': views.user_id.to_numpy()[vi[valid]],
                        'tv_show_id': ids[si[valid]], 'seconds': overlap[valid]})
                    parts[key].append(records.groupby(['user_id', 'tv_show_id'], sort=False).seconds.sum().reset_index())
        if counts['rows'] % 1000000 == 0:
            print('Joined', counts['rows'], 'view rows', flush=True)
    counts['users'] = len(users)
    durations = np.concatenate(duration_samples)
    counts['duration_quantiles_seconds'] = {str(q): float(np.quantile(durations, q)) for q in [0, .25, .5, .75, .9, .99, 1]}
    audit['views'] = counts
    for key in parts:
        interactions = pd.concat(parts[key], ignore_index=True).groupby(['user_id', 'tv_show_id'], sort=False).seconds.sum().reset_index()
        interactions.to_csv(OUT/f'{key}_interactions.csv.gz', index=False)
        audit['windows'][key]['user_item_pairs'] = len(interactions)
        audit['windows'][key]['users_with_labelled_history'] = int(interactions.user_id.nunique())
        audit['windows'][key]['positive_pair_seconds_quantiles'] = {str(q): float(interactions.seconds.quantile(q)) for q in [.25, .5, .75, .9, .99]}
    audit['preparation_seconds'] = time.perf_counter()-started
    (OUT/'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    prepare()
