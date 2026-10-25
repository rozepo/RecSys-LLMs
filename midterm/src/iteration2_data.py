"""Weekly interval aggregates, preserving channels and exact split boundaries."""
import json
import time

import numpy as np
import pandas as pd

from prepare import BEGIN, DATA, END, OUT, seconds

WEEK = 7*86400
I2 = OUT/'iteration2'


def load_schedule(test=False):
    periods = ['31-42'] if test else ['11-20', '21-30']
    frames = []
    for period in periods:
        frame = pd.read_csv(DATA/f'export_arh_{period}-final.csv',
                            usecols=['channel_id', 'start_time', 'duration', 'tv_show_id', 'tv_show_category'])
        frame['s'] = seconds(frame.start_time, '%d.%m.%Y %H:%M:%S')
        frame['e'] = frame.s+frame.duration
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(['channel_id', 's', 'e', 'tv_show_id'])
    return frame[(frame.e > frame.s) & (frame.tv_show_id != 0)].sort_values(['channel_id', 's', 'e', 'tv_show_id'])


def split_weeks(frame, keys):
    """Split positive intervals at Mondays; aggregate seconds, no rounding loss."""
    s, e = frame.s.to_numpy(), frame.e.to_numpy()
    first, last = (s-BEGIN)//WEEK, (e-1-BEGIN)//WEEK
    counts = last-first+1
    ii = np.repeat(np.arange(len(frame)), counts)
    ww = np.repeat(first, counts)+np.arange(len(ii))-np.repeat(np.cumsum(counts)-counts, counts)
    values = {k: frame[k].to_numpy()[ii] for k in keys}
    values['week'] = ww+11
    values['seconds'] = np.minimum(e[ii], BEGIN+(ww+1)*WEEK)-np.maximum(s[ii], BEGIN+ww*WEEK)
    return pd.DataFrame(values).groupby(keys+['week'], sort=False).seconds.sum().reset_index()


def prepare_weekly():
    I2.mkdir(exist_ok=True)
    started = time.perf_counter()
    schedule = load_schedule()
    groups = {c: (g.s.to_numpy(), g.e.to_numpy(), g.tv_show_id.to_numpy(), np.maximum.accumulate(g.e.to_numpy()))
              for c, g in schedule.groupby('channel_id')}
    parts, channels = [], []
    rows = 0
    for frame in pd.read_csv(DATA/'dataset11-30.csv', dtype={'user_id': str}, chunksize=250000):
        rows += len(frame)
        frame['s'] = np.maximum(seconds(frame.start_time, '%Y-%m-%d %H:%M:%S'), BEGIN)
        frame['e'] = np.minimum(seconds(frame.stop_time, '%Y-%m-%d %H:%M:%S'), END)
        frame = frame[frame.e > frame.s].rename(columns={'vsetv_id':'channel_id'})
        channels.append(split_weeks(frame, ['user_id', 'channel_id']))
        joined = []
        for channel, views in frame.groupby('channel_id', sort=False):
            if channel not in groups:
                continue
            ss, ee, ids, prefix = groups[channel]
            vs, ve = views.s.to_numpy(), views.e.to_numpy()
            left = np.searchsorted(prefix, vs, side='right')
            right = np.searchsorted(ss, ve, side='left')
            lengths = np.maximum(right-left, 0)
            vi = np.repeat(np.arange(len(views)), lengths)
            if not len(vi):
                continue
            si = np.repeat(left, lengths)+np.arange(len(vi))-np.repeat(np.cumsum(lengths)-lengths, lengths)
            s, e = np.maximum(vs[vi], ss[si]), np.minimum(ve[vi], ee[si])
            valid = e > s
            joined.append(pd.DataFrame({'user_id': views.user_id.to_numpy()[vi[valid]],
                'tv_show_id': ids[si[valid]], 'channel_id': channel, 's': s[valid], 'e': e[valid]}))
        parts.append(split_weeks(pd.concat(joined, ignore_index=True), ['user_id', 'tv_show_id', 'channel_id']))
        if rows % 1000000 == 0:
            print('Iteration 2 weekly join:', rows, flush=True)
    weekly = pd.concat(parts).groupby(['user_id','tv_show_id','channel_id','week'], sort=False).seconds.sum().reset_index()
    channel = pd.concat(channels).groupby(['user_id','channel_id','week'], sort=False).seconds.sum().reset_index()
    weekly.to_csv(I2/'weekly_interactions.csv.gz', index=False)
    channel.to_csv(I2/'weekly_channels.csv.gz', index=False)
    # Full-data comparison to the independent first-iteration cache, not a sample.
    verification = {}
    for name, mask in [('train',weekly.week <= 20),('validation',weekly.week >= 21)]:
        expected = pd.read_csv(OUT/f'{name}_interactions.csv.gz', dtype={'user_id':str})
        actual = weekly[mask].groupby(['user_id','tv_show_id']).seconds.sum().sort_index()
        expected = expected.set_index(['user_id','tv_show_id']).seconds.sort_index()
        pd.testing.assert_series_equal(actual, expected, check_names=False)
        verification[name] = {'pairs': len(actual), 'seconds': int(actual.sum()), 'exact_match': True}
    audit = {'view_rows': rows, 'weekly_interaction_rows': len(weekly), 'weekly_channel_rows':len(channel),
             'seconds':time.perf_counter()-started, 'cache_comparison':verification}
    (I2/'data_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    prepare_weekly()
