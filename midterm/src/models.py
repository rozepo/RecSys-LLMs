"""Reproducible implicit models, MAP@5 and common recommendation policy."""
import time
import numpy as np
from scipy import sparse

SEED = 20261024


def matrix(frame, users, items, threshold):
    selected = frame[frame.seconds >= threshold]
    u = {v: i for i, v in enumerate(users)}
    it = {v: i for i, v in enumerate(items)}
    rows = selected.user_id.map(u).to_numpy()
    cols = selected.tv_show_id.map(it).to_numpy()
    return sparse.csr_matrix((selected.seconds.to_numpy(dtype=float), (rows, cols)), shape=(len(users), len(items)))


def rank(scores, candidates, popularity, cold=None, k=5):
    """Positive model scores first, then popularity; stable ties by ascending id index."""
    scores = np.asarray(scores)
    result = np.empty((len(scores), k), dtype=np.int64)
    eligible = np.flatnonzero(candidates)
    pop_order = eligible[np.lexsort((eligible, -popularity[eligible]))]
    for u, values in enumerate(scores):
        if cold is not None and cold[u]:
            result[u] = pop_order[:k]
            continue
        good = eligible[np.isfinite(values[eligible]) & (values[eligible] > 0)]
        order = good[np.lexsort((good, -values[good]))][:k].tolist()
        for item in pop_order:
            if len(order) == k:
                break
            if item not in order:
                order.append(int(item))
        result[u] = order
    return result


def truths(frame, users, items):
    it = {v: i for i, v in enumerate(items)}
    grouped = {str(u): g.sort_values(['seconds', 'tv_show_id'], ascending=[False, True])
               for u, g in frame.groupby('user_id')}
    out = {'top10': [], 'top5': [], 'ge1800': []}
    for user in users:
        g = grouped.get(user)
        for name, n in [('top10', 10), ('top5', 5), ('ge1800', None)]:
            values = [] if g is None else (g[g.seconds >= 1800] if n is None else g.iloc[:n]).tv_show_id
            out[name].append(set(it[int(v)] for v in values))
    return out


def ap5(prediction, truth):
    if not truth:
        return 0.0
    hits = 0
    score = 0.0
    for r, item in enumerate(prediction, 1):
        if int(item) in truth:
            hits += 1
            score += hits/r
    return score/min(5, len(truth))


def evaluate(prediction, truth):
    return {name: float(np.mean([ap5(p, t) for p, t in zip(prediction, rows)])) for name, rows in truth.items()}


def ppr_graph(x, neighbors=50):
    """Top-k weighted cosine co-view graph; zero diagonal, row stochastic."""
    weighted = x.copy()
    weighted.data = np.log1p(weighted.data/60)
    norms = np.sqrt(np.asarray(weighted.power(2).sum(axis=1)).ravel())
    weighted = sparse.diags(1/np.maximum(norms, 1e-12)) @ weighted
    graph = (weighted.T @ weighted).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    norms = np.sqrt(np.asarray(weighted.power(2).sum(axis=0)).ravel())
    graph = (sparse.diags(1/np.maximum(norms, 1e-12)) @ graph @ sparse.diags(1/np.maximum(norms, 1e-12))).tocsr()
    rows, cols, vals = [], [], []
    for i in range(graph.shape[0]):
        a, b = graph.indptr[i:i+2]
        ix, v = graph.indices[a:b], graph.data[a:b]
        order = np.lexsort((ix, -v))[:neighbors]
        rows.extend([i]*len(order))
        cols.extend(ix[order])
        vals.extend(v[order])
    graph = sparse.csr_matrix((vals, (rows, cols)), shape=graph.shape, dtype=np.float32)
    sums = np.asarray(graph.sum(axis=1)).ravel()
    # Isolated items retain their mass via a self-loop.
    graph += sparse.diags((sums == 0).astype(np.float32))
    sums = np.asarray(graph.sum(axis=1)).ravel()
    return (sparse.diags(1/sums) @ graph).tocsr()


def ppr(graph, history, alpha, tol=1e-6, max_iter=200):
    initial = history.T.astype(np.float32).copy()
    sums = initial.sum(axis=0)
    initial /= np.maximum(sums, 1)
    score = initial.copy()
    transition = graph.T.tocsr()
    for iteration in range(max_iter):
        updated = (1-alpha)*initial + alpha*(transition @ score)
        delta = float(np.max(np.sum(np.abs(updated-score), axis=0)))
        score = updated
        if delta < tol:
            break
    return score.T, {'iterations': iteration+1, 'max_user_l1_residual': delta}


def als(x, factors=32, scale=10., regularization=10., transform='log', iterations=15, seed=SEED):
    """Hu-style implicit ALS, NumPy solves; missing confidence=1, preference=0."""
    confidence = x.copy()
    confidence.data = scale*(np.log1p(confidence.data/60) if transform == 'log' else confidence.data/3600)
    transposed = confidence.T.tocsr()
    rng = np.random.default_rng(seed)
    user = rng.normal(0, .01, (x.shape[0], factors))
    item = rng.normal(0, .01, (x.shape[1], factors))
    eye = np.eye(factors)*regularization

    def update(rows, fixed, target):
        # Explicit einsum avoids spurious floating-point flags from the macOS
        # Accelerate/NumPy 2.0 matmul path; no warning suppression is used.
        base = np.einsum('ni,nj->ij', fixed, fixed) + eye
        for i in range(rows.shape[0]):
            a, b = rows.indptr[i:i+2]
            indices, excess = rows.indices[a:b], rows.data[a:b]
            if len(indices) == 0:
                target[i] = 0
                continue
            y = fixed[indices]
            lhs = base + np.einsum('ni,nj->ij', y*excess[:, None], y)
            rhs = np.einsum('ni,n->i', y, 1+excess)
            target[i] = np.linalg.solve(lhs, rhs)
    for _ in range(iterations):
        update(confidence, item, user)
        update(transposed, user, item)
        assert np.isfinite(user).all() and np.isfinite(item).all(), 'Non-finite ALS factors'
    return user, item


def factor_scores(user, item):
    score = np.einsum('uf,if->ui', user, item)
    assert np.isfinite(score).all(), 'Non-finite ALS scores'
    return score


def fusion_features(scores, candidates, popularity, cold):
    """Reciprocal-rank scores with k=20; raw scales cannot dominate fusion."""
    features = []
    for score in scores:
        order = rank(score, candidates, popularity, cold, k=100)
        feature = np.zeros_like(score, dtype=np.float32)
        np.put_along_axis(feature, order, np.broadcast_to(1/(20+np.arange(1, 101)), order.shape), axis=1)
        features.append(feature)
    return features
