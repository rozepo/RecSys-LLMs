/**
 * week5/script.js — Matrix Factorization with TensorFlow.js (HW5).
 *
 * This is a plain (classic) script, NOT an ES module. `index.html` loads the
 * TensorFlow.js runtime from a CDN before this file, exposing the global `tf`:
 *
 *   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
 *   <script src="data.js"></script>
 *   <script src="script.js"></script>
 *
 * `week5/data.js` (loaded before this file) exposes the MovieLens 100k data
 * layer as globals: `movies`, `ratings`, `numUsers`, `numMovies`,
 * `ratingMatrix`, `SPLIT_EXCLUSIONS`, `loadData()` and `splitByTimestamp()`.
 * This file reads those globals, renders a dataset / split summary, and wires
 * the page controls. The factorization itself is built on `tf.Variable`s and
 * tensor ops so it can run on the WebGL (GPU) backend.
 *
 * BACKEND: `initTf()` prefers WebGL, falls back to CPU, and writes the chosen
 * backend into `#tf-status`. If the CDN did not load (`typeof tf ===
 * 'undefined'`) the page shows a clear message and stops instead of throwing.
 *
 * Implemented HW5 functions:
 *   1. `predictRating`        — scalar gather + dot + globalMean, clamp [1, 5].
 *   2. `initializeFactors`    — two `tf.Variable`s + the global mean.
 *   3. `trainMF`              — batched tensor training loop with L2 / Adam.
 *   4. `selectBestCheckpoint` — earliest entry with the smallest val RMSE.
 *   5. `recommendTopK`        — `matMul` + `tf.topk`, exclude train/val items.
 *   6. `recallAtK`            — Recall@K against relevant (>= 4.0) test items.
 *
 * PROVIDED for you (scaffolding): `initTf`, the shared metric helpers
 * (`computeRMSEFor`, `computeBaselineRMSE`), formatting helpers, every DOM
 * renderer, the hyper-parameter reader, the training pipeline wrapper, and the
 * automated test harness. Scaffolding also handles cold-start and tensor cleanup.
 *
 * The harness reports unimplemented stubs as PENDING rather than FAIL, so the
 * page stays usable before and after the assignment is done.
 *
 * @module week5/script
 */

// ---------------------------------------------------------------------------
// Typedefs
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} Movie
 * @property {number} id
 * @property {string} title
 * @property {string[]} genres
 */

/**
 * @typedef {Object} Rating
 * @property {number} userId
 * @property {number} movieId
 * @property {number} rating
 * @property {number} timestamp
 */

/**
 * @typedef {Object} Model
 * @property {tf.Variable} userFactors Dense factor matrix `[numUsers + 1, F]`,
 *   row index = raw `userId`.
 * @property {tf.Variable} itemFactors Dense factor matrix `[maxMovieId + 1, F]`,
 *   row index = raw `movieId`.
 * @property {number} globalMean       Mean rating of the training set.
 */

/**
 * @typedef {Object} HistoryRow
 * @property {number} epoch
 * @property {number} trainRMSE
 * @property {number} valRMSE
 */

/**
 * @typedef {Object} TrainOptions
 * @property {number} epochs    Number of full passes over the training set.
 * @property {number} lr        Learning rate.
 * @property {number} reg       L2 regularization strength.
 * @property {number} batchSize Mini-batch size.
 * @property {function(HistoryRow): void} [onEpoch] Synchronous progress callback.
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/** Number of latent factors used by the default pipeline. */
const NUM_FACTORS = 16;
const RANDOM_SEED = 20261016;
// Do not inherit backend.epsilon(): WebGL may choose 1e-4 while Node uses 1e-7.
const ADAM_EPSILON = 1e-7;

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function hasTrainingPair(userId, movieId) {
  return Number.isInteger(userId) && Number.isInteger(movieId) &&
    userId > 0 && userId < model.userFactors.shape[0] &&
    movieId > 0 && movieId < model.itemFactors.shape[0] &&
    (!model.trainUsers || model.trainUsers.has(userId)) &&
    (!model.trainMovies || model.trainMovies.has(movieId));
}

function disposeModel(value) {
  if (value) tf.dispose([value.userFactors, value.itemFactors]);
}

/** Training split (80% of each user's ratings, oldest first). */
let trainSet = [];

/** Validation split (10% of each user's ratings). */
let valSet = [];

/** Test split (10% of each user's ratings, newest). */
let testSet = [];

/**
 * The fitted factorization, or `null` before `initializeFactors` runs.
 *
 * @type {Model|null}
 */
let model = null;

/**
 * Metric history produced by `trainMF`.
 *
 * @type {HistoryRow[]}
 */
let trainingHistory = [];

/**
 * The history entry returned by `selectBestCheckpoint`, or `null`.
 *
 * @type {HistoryRow|null}
 */
let bestCheckpoint = null;

/** Name of the TensorFlow.js backend selected by `initTf` (`"webgl"` / `"cpu"`). */
let tfBackend = null;
let uiBusy = false;

// Both training and async fixtures own the shared model until they finish.
function setUiBusy(busy) {
  uiBusy = busy;
  for (const id of ['train', 'run-tests', 'recommend-btn', 'recall-btn']) {
    const button = document.getElementById(id);
    if (button) button.disabled = busy;
  }
}

// ---------------------------------------------------------------------------
// TensorFlow.js bootstrap (provided)
// ---------------------------------------------------------------------------

/**
 * Select the fastest available TensorFlow.js backend and report it.
 *
 * Tries WebGL (GPU) first and falls back to the pure-JS CPU backend. Writes
 * `backend: <name>` into `#tf-status` when that element exists. If the CDN
 * failed, `tf` is undefined: this renders a readable message and returns
 * `null` instead of letting the page throw an uncaught `ReferenceError`.
 *
 * @returns {Promise<string|null>} the selected backend, or `null` when TF.js is missing.
 */
async function initTf() {
  const statusEl = document.getElementById("tf-status");
  if (typeof tf === "undefined") {
    const message =
      "TensorFlow.js failed to load. Check your network connection (the CDN script) and reload the page.";
    if (statusEl) {
      statusEl.textContent = message;
      statusEl.className = "error";
    } else if (document.body) {
      const banner = document.createElement("p");
      banner.className = "error";
      banner.textContent = message;
      document.body.insertBefore(banner, document.body.firstChild);
    }
    return null;
  }

  try {
    await tf.setBackend("webgl");
  } catch (error) {
    await tf.setBackend("cpu");
  }
  await tf.ready();

  tfBackend = tf.getBackend();
  if (statusEl) statusEl.textContent = `TensorFlow.js backend: ${tfBackend}`;
  return tfBackend;
}

/**
 * True when `candidate` is a TF.js factor model (dense `tf.Variable` rows).
 *
 * Used to route `computeRMSEFor` through the batched tensor path and to keep
 * the fixture-based tests independent of how the model was produced.
 *
 * @param {*} candidate
 * @returns {boolean}
 */
function isTfModel(candidate) {
  return (
    !!candidate &&
    !!candidate.userFactors &&
    typeof candidate.userFactors.dataSync === "function" &&
    !!candidate.itemFactors &&
    typeof candidate.itemFactors.dataSync === "function"
  );
}

// ---------------------------------------------------------------------------
// Formatting helpers (provided)
// ---------------------------------------------------------------------------

/**
 * Format a number to a fixed number of decimals. Non-finite and missing values
 * render as an em dash so the tables never show `NaN` or `undefined`.
 *
 * @param {number|null|undefined} x
 * @param {number} [d=4] decimal places
 * @returns {string}
 */
function fmt(x, d = 4) {
  if (x === null || x === undefined) return "\u2014";
  const n = Number(x);
  if (!Number.isFinite(n)) return "\u2014";
  return n.toFixed(d);
}

/**
 * Look up a movie title by id, with a readable fallback for unknown ids.
 *
 * @param {number|string} id
 * @returns {string}
 */
function movieTitle(id) {
  const numericId = Number(id);
  const movie = movies.find((m) => m.id === numericId);
  return movie ? movie.title : `Movie #${id}`;
}

/**
 * Escape text before inserting it into HTML.
 *
 * @param {string} text
 * @returns {string}
 */
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Clamp a number into the inclusive range `[lo, hi]`.
 *
 * @param {number} value
 * @param {number} lo
 * @param {number} hi
 * @returns {number}
 */
function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

// ---------------------------------------------------------------------------
// Shared metric helpers (provided)
// ---------------------------------------------------------------------------

/**
 * Root-mean-square error of the current model over a split.
 *
 * Provided so that `trainMF` and the test harness share one implementation.
 *
 * When `model` holds TF.js variables this builds the index tensors ONCE, uses
 * `tf.gather` + element-wise multiply to score every rating in a single batch,
 * and clamps to `[1, 5]` — the same contract as `predictRating`, but fast
 * enough to call every epoch. Out-of-range ids fall back to `globalMean`, and
 * the whole computation runs inside `tf.tidy` so the intermediate tensors are
 * released automatically.
 *
 * When no TF.js model is available it falls back to the scalar
 * `predictRating` loop. That path returns `NaN` instead of throwing while
 * `predictRating` is still a `TODO(hw5)` stub, so the helper never crashes the
 * page or the harness.
 *
 * @param {Rating[]} split
 * @returns {number} `0` for an empty split; `NaN` if the scalar path throws.
 */
function computeRMSEFor(split) {
  if (!split || split.length === 0) return 0;

  if (isTfModel(model)) {
    return tf.tidy(() => {
      const n = split.length;
      const uRows = model.userFactors.shape[0];
      const iRows = model.itemFactors.shape[0];
      const userIds = new Int32Array(n);
      const movieIds = new Int32Array(n);
      const actual = new Float32Array(n);
      const valid = new Float32Array(n);

      for (let k = 0; k < n; k++) {
        const r = split[k];
        userIds[k] = r.userId;
        movieIds[k] = r.movieId;
        actual[k] = r.rating;
        valid[k] = hasTrainingPair(r.userId, r.movieId) ? 1 : 0;
      }

      const uIdx = tf.clipByValue(tf.tensor1d(userIds, "int32"), 0, Math.max(0, uRows - 1));
      const iIdx = tf.clipByValue(tf.tensor1d(movieIds, "int32"), 0, Math.max(0, iRows - 1));
      const u = tf.gather(model.userFactors, uIdx);
      const i = tf.gather(model.itemFactors, iIdx);
      const dot = tf.sum(tf.mul(u, i), 1);
      const mask = tf.tensor1d(valid, "float32");
      const pred = tf.clipByValue(dot.mul(mask).add(model.globalMean), 1, 5);
      const diff = pred.sub(tf.tensor1d(actual, "float32"));
      return Math.sqrt(tf.mean(diff.square()).dataSync()[0]);
    });
  }

  let sum = 0;
  try {
    for (const r of split) {
      const diff = predictRating(r.userId, r.movieId) - r.rating;
      sum += diff * diff;
    }
  } catch (error) {
    return Number.NaN;
  }
  return Math.sqrt(sum / split.length);
}

/**
 * Baseline RMSE: predict the training-set mean rating for every pair and
 * measure the error on `split`. This is the number your factorization must beat.
 *
 * @param {Rating[]} split
 * @returns {number} `0` for an empty split (never divides by zero).
 */
function computeBaselineRMSE(split) {
  if (!split || split.length === 0) return 0;
  const mean = trainSet.length
    ? trainSet.reduce((acc, r) => acc + r.rating, 0) / trainSet.length
    : 0;
  let sum = 0;
  for (const r of split) {
    const diff = mean - r.rating;
    sum += diff * diff;
  }
  return Math.sqrt(sum / split.length);
}

// ---------------------------------------------------------------------------
// HW5 implementation. The task notes below describe the original six exercises.
// ---------------------------------------------------------------------------

/**
 * Predict the rating a user would give a movie (scalar).
 *
 * TODO(hw5): implement the prediction with TensorFlow.js. The model stores the
 * factors as two dense `tf.Variable`s:
 *   - `model.userFactors` shape `[numUsers + 1, F]`, row index = `userId`
 *   - `model.itemFactors` shape `[maxMovieId + 1, F]`, row index = `movieId`
 *
 * Suggested TF.js path (wrap it in `tf.tidy` and pull out a JS number):
 * ```js
 * const f = model.userFactors.shape[1];
 * return tf.tidy(() => {
 *   const u = model.userFactors.slice([userId, 0], [1, f]); // [1, F]
 *   const i = model.itemFactors.slice([movieId, 0], [1, f]); // [1, F]
 *   const dot = u.mul(i).sum().dataSync()[0];                // scalar
 *   return clamp(model.globalMean + dot, 1, 5);
 * });
 * ```
 * `dataSync()` copies the value to the CPU as a plain number, which survives
 * `tf.tidy` (the tensors do not). Keep this function scalar: `computeRMSEFor`
 * uses its own batched tensor path, and the harness calls it once per pair.
 *
 * Contract:
 *  - Return `model.globalMean` for any user or movie absent from train,
 *    including allocated but untrained rows. Never throw for an unknown id.
 *  - Return `0` when the model is not initialized (`model === null`) so callers
 *    can guard cheaply.
 *  - The result is clamped to `[1, 5]`.
 *
 * @param {number} userId
 * @param {number} movieId
 * @returns {number} predicted rating in `[1, 5]`
 */
function predictRating(userId, movieId) {
  if (!model) return 0;
  if (!hasTrainingPair(userId, movieId)) return model.globalMean;
  return tf.tidy(() => {
    const u = tf.gather(model.userFactors, userId);
    const i = tf.gather(model.itemFactors, movieId);
    return clamp(u.mul(i).sum().dataSync()[0] + model.globalMean, 1, 5);
  });
}

/**
 * Allocate the latent factor variables and the global mean.
 *
 * TODO(hw5): compute `globalMean` as the mean rating over `train`
 * (`0` for an empty training set), then create the two dense variables:
 * ```js
 * const maxMovieId = movies.reduce((m, x) => Math.max(m, x.id), 0);
 * model = {
 *   // Wrap variable creation in tf.tidy to release randomNormal temporaries.
 *   userFactors: tf.tidy(() => tf.variable(tf.randomNormal([numUsers + 1, numFactors], 0, 0.05))),
 *   itemFactors: tf.tidy(() => tf.variable(tf.randomNormal([maxMovieId + 1, numFactors], 0, 0.05))),
 *   globalMean,
 * };
 * ```
 * `tf.randomNormal(shape, mean, std)` draws small values — use a small std
 * (e.g. `0.05`) so training starts near the global mean. `tf.variable` wraps a
 * tensor so the optimizer can update it in place. Store the result on the
 * module-level `model` object and return it.
 *
 * Contract:
 *  - `numFactors` defaults to the module-level `NUM_FACTORS`.
 *  - If a previous `model` exists, dispose its variables (`tf.dispose`) before
 *    replacing it so repeated "Train" clicks do not leak GPU memory.
 *  - Return the `model` object.
 *
 * @param {Rating[]} train
 * @param {number} [numFactors=NUM_FACTORS]
 * @returns {Model}
 */
function initializeFactors(train, numFactors = NUM_FACTORS) {
  disposeModel(model);
  const maxUser = train.reduce((m, r) => Math.max(m, r.userId), numUsers);
  const maxMovie = movies.reduce((m, x) => Math.max(m, x.id), 0);
  model = tf.tidy(() => ({
    userFactors: tf.variable(tf.randomNormal([maxUser + 1, numFactors], 0, 0.05, 'float32', RANDOM_SEED)),
    itemFactors: tf.variable(tf.randomNormal([maxMovie + 1, numFactors], 0, 0.05, 'float32', RANDOM_SEED + 1)),
  }));
  model.globalMean = train.length ? train.reduce((s, r) => s + r.rating, 0) / train.length : 0;
  model.trainUsers = new Set(train.map(r => r.userId));
  model.trainMovies = new Set(train.map(r => r.movieId));
  trainingHistory = [];
  bestCheckpoint = null;
  return model;
}

/**
 * Train the factorization with batched gradient descent.
 *
 * TODO(hw5): run a mini-batch loop over `train` using the TF.js optimizer.
 * Score each batch with tensor ops (never one `tf` call per rating) and let the
 * optimizer update both variables in place:
 * ```js
 * const optimizer = tf.train.adam(lr, 0.9, 0.999, ADAM_EPSILON);
 * for (let epoch = 1; epoch <= epochs; epoch++) {
 *   for (const batch of batches) {
 *     optimizer.minimize(() => tf.tidy(() => {
 *       const u = tf.gather(model.userFactors, batch.userIdx);
 *       const i = tf.gather(model.itemFactors, batch.itemIdx);
 *       const pred = tf.sum(u.mul(i), 1).add(model.globalMean);
 *       const mse = tf.losses.meanSquaredError(batch.targets, pred);
 *       const l2 = u.square().sum(1).add(i.square().sum(1)).mean().mul(reg);
 *       return mse.add(l2);
 *     }), false, [model.userFactors, model.itemFactors]);
 *   }
 *   trainingHistory.push({
 *     epoch,
 *     trainRMSE: computeRMSEFor(train),
 *     valRMSE: computeRMSEFor(val),
 *   });
 * }
 * return trainingHistory;
 * ```
 * Key TF.js APIs to use: `tf.train.adam(lr)` (or `tf.train.sgd`),
 * `optimizer.minimize(fn, returnCost, varList)`, `tf.gather`,
 * `tf.losses.meanSquaredError`, and `tf.tidy` for per-batch intermediates.
 * `minimize` returns null with returnCost=false; with true, dispose the loss.
 * The implementation uses mean per-observation squared factor norms for L2.
 *
 * Contract:
 *  - `opts = { epochs, lr, reg, batchSize }`. Use mini-batches of `batchSize`
 *    (or full-batch when `batchSize >= train.length`).
 *  - Update `model` in place. Do NOT reallocate it.
 *  - Append one `HistoryRow` per epoch to the module-level `trainingHistory`
 *    and return it: `{ epoch, trainRMSE: computeRMSEFor(train),
 *    valRMSE: computeRMSEFor(val) }`. `epoch` is 1-based.
 *  - Shuffling the batch order each epoch is allowed and usually helps.
 *  - Guard `train.length === 0` by returning an empty history.
 *
 * @param {Rating[]} train
 * @param {Rating[]} val
 * @param {TrainOptions} opts
 * @returns {Promise<HistoryRow[]>}
 */
async function trainMF(train, val, opts) {
  trainingHistory = [];
  if (!train.length) return trainingHistory;
  const { epochs, lr, reg, batchSize } = opts;
  if (!model || !Number.isInteger(epochs) || epochs < 1 ||
      !Number.isInteger(batchSize) || batchSize < 1 ||
      !Number.isFinite(lr) || lr <= 0 || !Number.isFinite(reg) || reg < 0) {
    throw new Error('Invalid model or training options');
  }
  if (!val.length) throw new Error('Validation ratings required for checkpoint selection');
  const optimizer = tf.train.adam(lr, 0.9, 0.999, ADAM_EPSILON);
  const random = seededRandom(RANDOM_SEED + 2);
  const order = train.map((_, i) => i);
  let saved = null;
  let best = null;
  try {
    for (let epoch = 1; epoch <= epochs; epoch++) {
      const tensorsBefore = tf.memory().numTensors;
      let batches = 0;
      for (let j = order.length - 1; j > 0; j--) {
        const k = Math.floor(random() * (j + 1));
        [order[j], order[k]] = [order[k], order[j]];
      }
      for (let start = 0; start < order.length; start += batchSize) {
        const batch = order.slice(start, start + batchSize).map(i => train[i]);
        tf.tidy(() => {
          const users = tf.tensor1d(batch.map(r => r.userId), 'int32');
          const items = tf.tensor1d(batch.map(r => r.movieId), 'int32');
          const targets = tf.tensor1d(batch.map(r => r.rating));
          optimizer.minimize(() => {
            const u = tf.gather(model.userFactors, users);
            const i = tf.gather(model.itemFactors, items);
            const pred = u.mul(i).sum(1).add(model.globalMean);
            const penalty = u.square().sum(1).add(i.square().sum(1)).mean().mul(reg);
            return tf.losses.meanSquaredError(targets, pred).add(penalty);
          }, false, [model.userFactors, model.itemFactors]);
        });
        // Never await inside tidy: its scope must finish synchronously.
        if (++batches % 20 === 0) await tf.nextFrame();
      }
      const row = { epoch, trainRMSE: computeRMSEFor(train), valRMSE: computeRMSEFor(val) };
      if (!Number.isFinite(row.valRMSE)) throw new Error('Non-finite validation RMSE');
      trainingHistory.push(row);
      if (!best || row.valRMSE < best.valRMSE) {
        tf.dispose(saved);
        saved = [model.userFactors.clone(), model.itemFactors.clone()];
        best = row;
      }
      const tensorsAfter = tf.memory().numTensors;
      // Adam lazily allocates 2 slots per variable; the first checkpoint owns 2 tensors.
      const expectedGrowth = epoch === 1 ? 6 : 0;
      row.memory = { before: tensorsBefore, after: tensorsAfter, expectedGrowth };
      if (tensorsAfter - tensorsBefore !== expectedGrowth) {
        throw new Error(`Tensor count changed unexpectedly in epoch ${epoch}: ${tensorsBefore} -> ${tensorsAfter}`);
      }
      if (opts.onEpoch) opts.onEpoch(row);
      await tf.nextFrame();
    }
    model.userFactors.assign(saved[0]);
    model.itemFactors.assign(saved[1]);
    bestCheckpoint = selectBestCheckpoint(trainingHistory);
    return trainingHistory;
  } finally {
    tf.dispose(saved);
    optimizer.dispose();
  }
}

/**
 * Pick the best epoch from a training history.
 *
 * TODO(hw5): return the entry with the smallest `valRMSE`. On a tie return the
 * entry with the EARLIEST `epoch` (do not simply take the last minimum). This
 * one is plain JavaScript — no TF.js needed.
 *
 * Contract:
 *  - Return `null` for an empty or missing history.
 *  - Do not mutate the input array.
 *
 * @param {HistoryRow[]} history
 * @returns {HistoryRow|null}
 */
function selectBestCheckpoint(history) {
  return (history || []).reduce((best, row) => {
    if (!Number.isFinite(row.valRMSE)) return best;
    return !best || row.valRMSE < best.valRMSE ||
      (row.valRMSE === best.valRMSE && row.epoch < best.epoch) ? row : best;
  }, null);
}

/**
 * Recommend the top-K movies for a user.
 *
 * TODO(hw5): score every candidate movie and return the K highest-scoring
 * movies the user has NOT already rated in `trainSet` or `valSet`. Test-set
 * ratings are allowed as candidates. **Prefer the tensor path**:
 * ```js
 * const userRow = model.userFactors.slice([userId, 0], [1, F]); // [1, F]
 * const scores = model.itemFactors.matMul(userRow, false, true)  // [rows, 1]
 *   .squeeze().add(model.globalMean);                            // [rows]
 * // mask excluded ids to a large negative value, then:
 * const { values, indices } = tf.topk(maskedScores, Math.min(K, rows));
 * ```
 * Because `itemFactors` is indexed by raw `movieId`, the `tf.topk` indices ARE
 * the movie ids — no reverse mapping needed. Excluded rows must be masked (for
 * example multiply by a 0/1 mask and add `(1 - mask) * -1e9`); re-apply the
 * same mask when mapping results back so masked ids are dropped.
 *
 * Contract:
 *  - Return `[{ movieId, title, score }, ...]` sorted by descending `score`,
 *    truncated to `K` (fewer if not enough candidates).
 *  - Candidates are catalog movies present in train; exclude the user's
 *    train+val movies. Scores are unclipped; equal scores use ascending movie ID.
 *  - Return `[]` when the model is not initialized, when `K <= 0`, or when the
 *    user has no eligible candidates — never throw.
 *  - `title` comes from `movieTitle(movieId)`.
 *
 * @param {number} userId
 * @param {number} [K=10]
 * @returns {Array<{movieId: number, title: string, score: number}>}
 */
function recommendTopK(userId, K = 10) {
  if (!model || !Number.isInteger(K) || K <= 0 ||
      !trainSet.some(r => r.userId === userId)) return [];
  const seen = new Set([...trainSet, ...valSet].filter(r => r.userId === userId).map(r => r.movieId));
  const pool = new Set(trainSet.map(r => r.movieId));
  const candidates = movies.filter(m => pool.has(m.id) && !seen.has(m.id) &&
    hasTrainingPair(userId, m.id)).sort((a, b) => a.id - b.id);
  if (!candidates.length) return [];
  return tf.tidy(() => {
    const items = tf.gather(model.itemFactors, tf.tensor1d(candidates.map(m => m.id), 'int32'));
    const user = model.userFactors.slice([userId, 0], [1, model.userFactors.shape[1]]);
    // Rank by raw dot-product scores: clipping would introduce artificial ties.
    const scores = items.matMul(user, false, true).reshape([candidates.length]).add(model.globalMean);
    const { values, indices } = tf.topk(scores, Math.min(K, candidates.length), true);
    const data = values.dataSync();
    return Array.from(indices.dataSync(), (index, rank) => ({
      movieId: candidates[index].id, title: candidates[index].title,
      genres: candidates[index].genres, score: data[rank],
    }));
  });
}

/**
 * Compute Recall@K for the current model.
 *
 * Batched retrieval evaluation with the same candidate/ranking rules as recommendTopK.
 *   - relevant = test ratings with `rating >= 4.0`.
 *   - Candidate items = the set of movie ids that appear anywhere in `trainSet`
 *     (a global candidate pool, NOT just the user's own training items).
 *   - For every user with at least one relevant test movie among the
 *     candidates, score all train movies with one matMul per user batch,
 *     mask train+val movies, and count relevant movies in each row's Top-K.
 *
 * Contract:
 *  - Return
 *    `{ overallRecall, perUser, excluded }` where:
 *      `overallRecall` = mean over reported users of `hit / relevant`
 *        (`0` when there are no reported users — never divide by zero);
 *      `perUser` = `[{ userId, relevant, hit }, ...]` (reported users only);
 *      `excluded` = `{ unseenIds, usersWithoutRelevant }`.
 *  - `unseenIds` = distinct relevant test movie ids that are NOT present in
 *    `trainSet` (they can never be recommended from a training-only pool).
 *  - `usersWithoutRelevant` = distinct users that appear in `testSet` but have
 *    no relevant test movie among the candidates.
 *  - Report exclusions explicitly so the recall number is interpretable.
 *
 * @param {number} [K=10]
 * @param {{batchSize?: number, scoring?: string, onProgress?: function(Object): void}} [opts]
 * @returns {Promise<Object>} Recall and explicit exclusions; no tensors escape.
 */
async function recallAtK(K = 10, { batchSize = 64, scoring = 'mf', onProgress } = {}) {
  if (!Number.isInteger(batchSize) || batchSize < 1) throw new Error('Invalid recall batch size');
  if (!['mf', 'popularity'].includes(scoring)) throw new Error('Invalid recall scoring mode');
  const pool = new Set(trainSet.map(r => r.movieId));
  const users = new Set(trainSet.map(r => r.userId));
  const seen = new Set([...trainSet, ...valSet].map(r => `${r.userId}:${r.movieId}`));
  const relevant = new Map();
  const unseenIds = new Set();
  const unseenUsers = new Set();
  const seenRelevantPairs = new Set();
  for (const r of testSet) {
    if (!relevant.has(r.userId)) relevant.set(r.userId, new Set());
    if (!users.has(r.userId)) unseenUsers.add(r.userId);
    if (r.rating < 4) continue;
    if (!pool.has(r.movieId)) unseenIds.add(r.movieId);
    if (seen.has(`${r.userId}:${r.movieId}`)) seenRelevantPairs.add(`${r.userId}:${r.movieId}`);
    if (users.has(r.userId) && pool.has(r.movieId) && !seen.has(`${r.userId}:${r.movieId}`)) {
      relevant.get(r.userId).add(r.movieId);
    }
  }
  const perUser = [];
  const usersWithoutRelevant = [];
  const eligible = [];
  for (const [userId, ids] of relevant) {
    if (!ids.size) { usersWithoutRelevant.push(userId); continue; }
    eligible.push({ userId, ids });
  }
  // Build the pool and per-user exclusions once, not once per recommendation list.
  const candidates = movies.filter(m => pool.has(m.id) && (scoring === 'popularity' || (model &&
    m.id > 0 && m.id < model.itemFactors.shape[0] &&
    (!model.trainMovies || model.trainMovies.has(m.id))))).sort((a, b) => a.id - b.id);
  const columns = new Map(candidates.map((m, i) => [m.id, i]));
  const popularity = new Float32Array(candidates.length);
  if (scoring === 'popularity') {
    for (const r of trainSet) if (columns.has(r.movieId)) popularity[columns.get(r.movieId)]++;
  }
  const seenColumns = new Map();
  for (const r of [...trainSet, ...valSet]) {
    if (!columns.has(r.movieId)) continue;
    if (!seenColumns.has(r.userId)) seenColumns.set(r.userId, new Set());
    seenColumns.get(r.userId).add(columns.get(r.movieId));
  }
  const k = Number.isInteger(K) && K > 0 ? Math.min(K, candidates.length) : 0;
  for (let start = 0; start < eligible.length; start += batchSize) {
    const batch = eligible.slice(start, start + batchSize);
    const tensorsBefore = tf.memory().numTensors;
    const ranked = k && (model || scoring === 'popularity') ? tf.tidy(() => {
      const width = candidates.length;
      const mask = new Uint8Array(batch.length * width).fill(1);
      const userIds = batch.map(({ userId }, row) => {
        if (scoring === 'mf' && !hasTrainingPair(userId, candidates[0].id)) {
          mask.fill(0, row * width, (row + 1) * width);
          return 0; // Safe gather for an untrained/out-of-range user; masked out below.
        }
        for (const column of seenColumns.get(userId) || []) mask[row * width + column] = 0;
        return userId;
      });
      let scores;
      if (scoring === 'popularity') {
        scores = tf.tensor2d(popularity, [1, width]).tile([batch.length, 1]);
      } else {
        const u = tf.gather(model.userFactors, tf.tensor1d(userIds, 'int32'));
        const i = tf.gather(model.itemFactors, tf.tensor1d(candidates.map(m => m.id), 'int32'));
        scores = u.matMul(i, false, true).add(model.globalMean);
      }
      const masked = tf.where(tf.tensor2d(mask, [batch.length, width], 'bool'), scores,
        tf.fill([batch.length, width], -Infinity));
      const { indices } = tf.topk(masked, k, true);
      const data = indices.dataSync();
      return batch.map((_, row) => Array.from(data.slice(row * k, (row + 1) * k))
        .filter(column => mask[row * width + column])
        .map(column => candidates[column].id));
    }) : batch.map(() => []);
    batch.forEach(({ userId, ids }, row) => {
      perUser.push({ userId, relevant: ids.size, hit: ranked[row].filter(id => ids.has(id)).length });
    });
    const tensorsAfter = tf.memory().numTensors;
    if (tensorsAfter !== tensorsBefore) throw new Error('Recall batch leaked tensors');
    if (onProgress) onProgress({ completed: perUser.length, total: eligible.length, tensorsBefore, tensorsAfter });
    await tf.nextFrame();
  }
  return {
    overallRecall: perUser.length ? perUser.reduce((s, u) => s + u.hit / u.relevant, 0) / perUser.length : 0,
    perUser,
    excluded: { unseenIds: [...unseenIds], usersWithoutRelevant,
      unseenUsers: [...unseenUsers], seenRelevantPairs: [...seenRelevantPairs] },
  };
}

// ---------------------------------------------------------------------------
// UI rendering (provided)
// ---------------------------------------------------------------------------

/**
 * Render the dataset shape summary.
 *
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderDatasetSummary(container) {
  const target = container || document.getElementById("dataset-summary-body");
  if (!target) return;
  const maxMovieId = movies.reduce((max, m) => Math.max(max, m.id), 0);
  target.innerHTML = `
    <dl class="summary-list">
      <dt>Movies</dt><dd>${movies.length.toLocaleString("en-US")}</dd>
      <dt>Ratings</dt><dd>${ratings.length.toLocaleString("en-US")}</dd>
      <dt>Users</dt><dd>${new Set(ratings.map(r => r.userId)).size.toLocaleString("en-US")}</dd>
      <dt>Sparsity</dt><dd>${fmt(100 * (1 - ratings.length / (new Set(ratings.map(r => r.userId)).size * movies.length)), 4)}%</dd>
      <dt>Rating matrix</dt><dd>${(numUsers + 1).toLocaleString("en-US")} &times; ${(maxMovieId + 1).toLocaleString("en-US")}</dd>
    </dl>`;
}

/**
 * Render the chronological 80/10/10 split summary.
 *
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderSplitSummary(container) {
  const target = container || document.getElementById("split-summary-body");
  if (!target) return;
  const total = trainSet.length + valSet.length + testSet.length;
  const pct = (n) => (total ? ((n / total) * 100).toFixed(1) : "0.0");
  const exclusions = typeof SPLIT_EXCLUSIONS === "undefined" ? [] : SPLIT_EXCLUSIONS;
  target.innerHTML = `
    <table class="data-table">
      <thead><tr><th scope="col">Split</th><th scope="col">Ratings</th><th scope="col">Share</th></tr></thead>
      <tbody>
        <tr><td>Train</td><td class="num">${trainSet.length}</td><td class="num">${pct(trainSet.length)}%</td></tr>
        <tr><td>Validation</td><td class="num">${valSet.length}</td><td class="num">${pct(valSet.length)}%</td></tr>
        <tr><td>Test</td><td class="num">${testSet.length}</td><td class="num">${pct(testSet.length)}%</td></tr>
      </tbody>
    </table>
    <p class="hint">${exclusions.length} user(s) skipped while splitting; baseline test RMSE ${fmt(computeBaselineRMSE(testSet))}.</p>`;
}

/**
 * Render the training history as a plain table (no SVG, no chart library).
 *
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderTrainingCurve(container) {
  const target = container || document.getElementById("training-curve");
  if (!target) return;
  if (!trainingHistory || trainingHistory.length === 0) {
    target.innerHTML =
      '<p class="empty-state">No training history yet. Press &ldquo;Train&rdquo;.</p>';
    return;
  }
  let bestIndex = 0;
  for (let i = 1; i < trainingHistory.length; i++) {
    if (trainingHistory[i].valRMSE < trainingHistory[bestIndex].valRMSE) bestIndex = i;
  }
  const rows = trainingHistory
    .map(
      (row, i) => `
      <tr${i === bestIndex ? ' class="best-epoch"' : ""}>
        <td class="num">${row.epoch}</td>
        <td class="num">${fmt(row.trainRMSE)}</td>
        <td class="num">${fmt(row.valRMSE)}</td>
      </tr>`,
    )
    .join("");
  target.innerHTML = `
    <table class="data-table">
      <thead><tr><th scope="col">Epoch</th><th scope="col">Train RMSE</th><th scope="col">Val RMSE</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

/**
 * Count favorite genres in positive past ratings, never test ratings.
 *
 * @param {number} userId
 * @returns {Array<[string, number]>}
 */
function favoriteGenres(userId) {
  const catalog = new Map(movies.map(m => [m.id, m]));
  const counts = new Map();
  for (const r of [...trainSet, ...valSet]) {
    if (r.userId !== Number(userId) || r.rating < 4) continue;
    for (const genre of catalog.get(r.movieId)?.genres || []) {
      counts.set(genre, (counts.get(genre) || 0) + 1);
    }
  }
  return [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

/** Render the top-K recommendations and past genre counts for a user. */
function renderTopK(userId, container) {
  const target = container || document.getElementById("recommendations");
  if (!target) return;
  if (!model) {
    target.innerHTML = '<p class="empty-state">Train the model before requesting recommendations.</p>';
    return;
  }
  try {
    const recs = recommendTopK(Number(userId), 10);
    if (!recs || recs.length === 0) {
      target.innerHTML = '<p class="empty-state">No eligible recommendations for this user.</p>';
      return;
    }
    target.innerHTML =
      `<p>Favorite past genres (train+val ratings ≥ 4): ${escapeHtml(favoriteGenres(userId).slice(0, 5).map(([g, n]) => `${g}: ${n}`).join(', '))}</p>` +
      '<ol class="recommendation-list">' +
      recs
        .map(
          (r) =>
            `<li><span class="rec-title">${escapeHtml(r.title)}</span> <span class="rec-score">${fmt(r.score)}</span> — ${escapeHtml(r.genres.join(', '))}</li>`,
        )
        .join("") +
      "</ol>";
  } catch (error) {
    target.innerHTML = `<p class="empty-state">Recommendations unavailable — implement the <code>TODO(hw5)</code> functions. <code>${escapeHtml(error.message)}</code></p>`;
  }
}

/**
 * Render the Recall@K report, including explicit exclusions.
 *
 * @param {number} [K=10]
 * @param {HTMLElement|null} [container]
 * @returns {Promise<void>}
 */
async function renderRecall(K = 10, container) {
  const target = container || document.getElementById("recall-results");
  if (!target || uiBusy) return;
  if (!model) {
    target.textContent = 'Train the model before evaluating Recall.';
    return;
  }
  setUiBusy(true);
  try {
    target.textContent = `Evaluating Recall@${K}…`;
    await tf.nextFrame();
    const result = await recallAtK(K, {
      onProgress: ({ completed, total }) => {
        target.textContent = `Evaluating Recall@${K}: ${completed}/${total} users`;
      },
    });
    const popularity = await recallAtK(K, {
      scoring: 'popularity',
      onProgress: ({ completed, total }) => {
        target.textContent = `Evaluating popularity Recall@${K}: ${completed}/${total} users`;
      },
    });
    const rows = result.perUser
      .map(
        (u) => `
        <tr>
          <td>${u.userId}</td>
          <td class="num">${u.relevant}</td>
          <td class="num">${u.hit}</td>
          <td class="num">${u.relevant ? fmt(u.hit / u.relevant) : "\u2014"}</td>
        </tr>`,
      )
      .join("");
    target.innerHTML = `
      <p><strong>Recall@${K}:</strong> <span class="num">${fmt(result.overallRecall, 6)}</span></p>
      <p><strong>Popularity Recall@${K}:</strong> <span class="num">${fmt(popularity.overallRecall, 6)}</span> (${popularity.perUser.length} users; train rating counts, movie ID breaks ties).</p>
      <table class="data-table">
        <thead><tr><th scope="col">User</th><th scope="col">Relevant</th><th scope="col">Hits</th><th scope="col">Recall</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="4">No evaluable users.</td></tr>'}</tbody>
      </table>
      <p class="hint">Evaluated: ${result.perUser.length} users. Excluded: ${result.excluded.unseenIds.length} unseen relevant test item(s); ${result.excluded.usersWithoutRelevant.length} user(s) without a relevant candidate (including ${result.excluded.unseenUsers.length} unseen users); ${result.excluded.seenRelevantPairs.length} already-seen relevant pairs.</p>`;
  } catch (error) {
    target.innerHTML = `<p class="empty-state">Recall failed: <code>${escapeHtml(error.message)}</code></p>`;
  } finally {
    setUiBusy(false);
  }
}

/**
 * Read the training hyper-parameters from the DOM, falling back to defaults when
 * the inputs are absent or invalid.
 *
 * @returns {TrainOptions}
 */
function readHyperparameters() {
  const readNumber = (id, fallback) => {
    const el = document.getElementById(id);
    if (!el) return fallback;
    const value = Number(el.value);
    return Number.isFinite(value) ? value : fallback;
  };
  return {
    epochs: Math.max(1, Math.round(readNumber("epochs", 20))),
    lr: Math.max(0.0001, readNumber("lr", 0.01)),
    reg: Math.max(0, readNumber("reg", 0.05)),
    batchSize: Math.max(1, Math.round(readNumber("batch-size", 512))),
  };
}

/**
 * Populate the user `<select>` from the training split.
 *
 * @returns {void}
 */
function populateUserSelect() {
  const select = document.getElementById("user-select");
  if (!select) return;
  const ids = Array.from(new Set(trainSet.map((r) => r.userId))).sort((a, b) => a - b);
  select.innerHTML = ids.length
    ? ids.map((id) => `<option value="${id}">User ${id}</option>`).join("")
    : '<option value="">No users</option>';
}

/**
 * Run the full training pipeline: initialize, train, select the best checkpoint,
 * and refresh the training curve.
 *
 * @returns {Promise<void>}
 */
async function runPipeline() {
  if (uiBusy) return;
  const status = document.getElementById("status");
  if (!movies.length || trainSet.length === 0) {
    if (status) status.textContent = "Data not loaded yet — reload the page.";
    return;
  }
  const opts = readHyperparameters();
  setUiBusy(true);
  try {
    if (status) status.textContent = "Training\u2026";
    await tf.nextFrame();
    initializeFactors(trainSet, NUM_FACTORS);
    renderTrainingCurve();
    trainingHistory = await trainMF(trainSet, valSet, {
      ...opts,
      onEpoch: row => {
        renderTrainingCurve();
        if (status) status.textContent = `Epoch ${row.epoch}/${opts.epochs}: val RMSE ${fmt(row.valRMSE)}; tensors ${row.memory.before} → ${row.memory.after}.`;
      },
    });
    bestCheckpoint = selectBestCheckpoint(trainingHistory);
    renderTrainingCurve();
    if (status) {
      status.textContent = bestCheckpoint
        ? `Trained ${opts.epochs} epoch(s) on ${tfBackend || "tf"}, seed ${RANDOM_SEED} — best val RMSE ${fmt(bestCheckpoint.valRMSE)} at epoch ${bestCheckpoint.epoch}. Restored checkpoint: test RMSE ${fmt(computeRMSEFor(testSet))}; baseline ${fmt(computeBaselineRMSE(testSet))}.`
        : `Trained ${opts.epochs} epoch(s) — no history recorded.`;
    }
  } catch (error) {
    const message = String(error && error.message ? error.message : error);
    if (status) status.textContent = message;
    const target = document.getElementById("training-curve");
    if (target) {
      target.innerHTML = `<p class="empty-state">Training failed: <code>${escapeHtml(message)}</code></p>`;
    }
  } finally {
    setUiBusy(false);
  }
}

/**
 * Wire the controls once the data has been loaded. Rendering is defensive so an
 * unimplemented stub can never take the whole page down at load time.
 *
 * @returns {void}
 */
function init() {
  populateUserSelect();
  renderDatasetSummary();
  renderSplitSummary();
  renderTrainingCurve();

  const testButton = document.getElementById("run-tests");
  if (testButton) testButton.addEventListener("click", async () => { await runTests(); });

  const trainButton = document.getElementById("train");
  if (trainButton) trainButton.addEventListener("click", async () => { await runPipeline(); });

  const recommendButton = document.getElementById("recommend-btn");
  if (recommendButton) {
    recommendButton.addEventListener("click", () => {
      const select = document.getElementById("user-select");
      renderTopK(select ? select.value : 1);
    });
  }

  const recallButton = document.getElementById("recall-btn");
  if (recallButton) recallButton.addEventListener("click", async () => { await renderRecall(10); });

  const status = document.getElementById("status");
  if (status) {
    status.textContent = `Dataset ready: ${movies.length.toLocaleString("en-US")} movies, ${trainSet.length.toLocaleString("en-US")} train ratings on ${tfBackend || "tf"}. Seed ${RANDOM_SEED}. Press \u201cTrain\u201d.`;
  }
}

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

/** Marker carried by every unimplemented stub error message. */
const TODO_MARKER = "TODO(hw5)";

/** A five-rating toy training set used by several checks. */
const TOY_RATINGS = [
  { userId: 1, movieId: 1, rating: 5, timestamp: 1 },
  { userId: 1, movieId: 2, rating: 4, timestamp: 2 },
  { userId: 1, movieId: 3, rating: 3, timestamp: 3 },
  { userId: 2, movieId: 1, rating: 2, timestamp: 1 },
  { userId: 2, movieId: 2, rating: 5, timestamp: 2 },
];

/** Movies matching `TOY_RATINGS`. */
const TOY_MOVIES = [
  { id: 1, title: "Toy One", genres: [] },
  { id: 2, title: "Toy Two", genres: [] },
  { id: 3, title: "Toy Three", genres: [] },
];

/** A tiny train/val/test fixture for the recommendation checks. */
const REC_FIXTURE = {
  movies: [
    { id: 1, title: "Alpha", genres: [] },
    { id: 2, title: "Bravo", genres: [] },
    { id: 3, title: "Charlie", genres: [] },
    { id: 4, title: "Delta", genres: [] },
    { id: 5, title: "Echo", genres: [] },
  ],
  train: [
    { userId: 1, movieId: 1, rating: 5, timestamp: 1 },
    ...[3, 4, 5].map(movieId => ({ userId: 2, movieId, rating: 4, timestamp: 1 })),
  ],
  val: [{ userId: 1, movieId: 2, rating: 4, timestamp: 2 }],
  test: [{ userId: 1, movieId: 3, rating: 5, timestamp: 3 }],
};

/**
 * Build a TF.js factor model from sparse score maps.
 *
 * `userScores[id]` / `itemScores[id]` may be a number (written into factor 0,
 * so the dot product is just the product) or an array (written across factors).
 * Rows are dense and indexed by raw id, exactly like the real model, so the
 * recommendation / recall fixtures exercise the same tensor path as training.
 *
 * @param {Object<number, number|number[]>} userScores
 * @param {Object<number, number|number[]>} itemScores
 * @param {number} [globalMean=0]
 * @returns {Model}
 */
function fixtureModel(userScores, itemScores, globalMean = 0) {
  const F = NUM_FACTORS;
  const userIds = Object.keys(userScores).map(Number);
  const itemIds = Object.keys(itemScores).map(Number);

  // A number goes into factor 0 (so the dot product is the product); an array
  // is copied across the first factors.
  const rowsFor = (value) => {
    if (Array.isArray(value) || ArrayBuffer.isView(value)) return Array.from(value);
    const row = new Array(F).fill(0);
    row[0] = Number(value);
    return row;
  };

  const maxUser = Math.max(numUsers, ...userIds);
  const maxMovieFromMovies = movies.reduce((m, x) => Math.max(m, x.id), 0);
  const maxItem = Math.max(maxMovieFromMovies, ...itemIds);

  const uData = new Float32Array((maxUser + 1) * F);
  for (const id of userIds) {
    const row = rowsFor(userScores[id]);
    for (let f = 0; f < Math.min(F, row.length); f++) uData[id * F + f] = row[f];
  }
  const iData = new Float32Array((maxItem + 1) * F);
  for (const id of itemIds) {
    const row = rowsFor(itemScores[id]);
    for (let f = 0; f < Math.min(F, row.length); f++) iData[id * F + f] = row[f];
  }

  return tf.tidy(() => ({
    userFactors: tf.variable(tf.tensor2d(uData, [maxUser + 1, F])),
    itemFactors: tf.variable(tf.tensor2d(iData, [maxItem + 1, F])),
    globalMean,
  }));
}

/**
 * A recall fixture with 12 movies, one user whose single relevant test movie
 * (movie 2) is placed at a chosen rank by the hand-made model.
 *
 * @param {number} targetRank 1 to put movie 2 first, 11 to put it last.
 * @returns {{movies: Movie[], train: Rating[], val: Rating[], test: Rating[], model: Model}}
 */
function recallFixture(targetRank) {
  const movies = [];
  for (let id = 1; id <= 12; id++) {
    movies.push({ id, title: `Movie ${id}`, genres: [] });
  }
  const train = [
    { userId: 1, movieId: 1, rating: 5, timestamp: 1 },
    { userId: 2, movieId: 2, rating: 5, timestamp: 1 },
    { userId: 2, movieId: 3, rating: 4, timestamp: 2 },
  ];
  const val = [];
  for (let movieId = 4; movieId <= 12; movieId++) {
    train.push({ userId: 2, movieId, rating: 3, timestamp: movieId });
  }
  const test = [{ userId: 1, movieId: 2, rating: 5, timestamp: 2 }];
  const itemScores = { 1: 3.0 };
  if (targetRank === 1) {
    itemScores[2] = 5.0;
    for (let id = 3; id <= 12; id++) itemScores[id] = 1.0 + 0.01 * id;
  } else {
    itemScores[2] = 1.0;
    for (let id = 3; id <= 12; id++) itemScores[id] = 1.1 + 0.01 * id;
  }
  return { movies, train, val, test, model: fixtureModel({ 1: 1 }, itemScores) };
}

/**
 * Run `fn` with the module-level data globals temporarily replaced by a fixture,
 * then restore them. Keeps the checks independent of the loaded dataset.
 *
 * @param {{movies?: Movie[], train?: Rating[], val?: Rating[], test?: Rating[], model?: Model|null, history?: HistoryRow[]}} fixture
 * @param {() => any} fn
 * @returns {Promise<any>}
 */
async function withFixture(fixture, fn) {
  const saved = { movies, trainSet, valSet, testSet, model, trainingHistory, bestCheckpoint };
  try {
    movies = fixture.movies !== undefined ? fixture.movies : movies;
    trainSet = fixture.train || [];
    valSet = fixture.val || [];
    testSet = fixture.test || [];
    model = fixture.model !== undefined ? fixture.model : null;
    trainingHistory = fixture.history || [];
    return await fn();
  } finally {
    if (model !== saved.model) disposeModel(model);
    movies = saved.movies;
    trainSet = saved.trainSet;
    valSet = saved.valSet;
    testSet = saved.testSet;
    model = saved.model;
    trainingHistory = saved.trainingHistory;
    bestCheckpoint = saved.bestCheckpoint;
  }
}

/**
 * Run the automated self-checks. Checks that call an unimplemented
 * `TODO(hw5)` stub are reported as PENDING rather than FAIL, so the harness is
 * useful before and after the assignment is implemented.
 *
 * @param {HTMLElement|null} [logElement] element that receives the text log
 * @returns {Promise<{passed: number, failed: number, pending: number, checks: Array<Object>}|null>}
 */
async function runTests(logElement) {
  if (uiBusy) return null;
  setUiBusy(true);
  try {
    return await runSelfChecks(logElement);
  } finally {
    setUiBusy(false);
  }
}

async function runSelfChecks(logElement) {
  const checks = [];
  const close = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

  const pass = (name, detail) => checks.push({ name, status: "PASS", detail });
  const fail = (name, detail) => checks.push({ name, status: "FAIL", detail });
  const pending = (name, detail) => checks.push({ name, status: "PENDING", detail });

  const check = async (name, fn) => {
    const target = logElement || document.getElementById('testLog');
    if (target) target.textContent = `Running self-check ${checks.length + 1}/11: ${name}`;
    await tf.nextFrame();
    try {
      const outcome = await fn();
      if (outcome && outcome.pending) pending(name, outcome.pending);
      else pass(name, outcome === undefined ? "" : String(outcome));
    } catch (error) {
      if (String(error && error.message).includes(TODO_MARKER)) {
        pending(name, "TODO(hw5): not implemented yet.");
      } else {
        fail(name, String(error && error.message ? error.message : error));
      }
    }
  };

  // 1. Shared helper: baseline RMSE on a toy split.
  await check("computeBaselineRMSE: toy split matches the hand-computed value", () =>
    withFixture(
      {
        train: [
          { userId: 1, movieId: 1, rating: 2, timestamp: 1 },
          { userId: 1, movieId: 2, rating: 4, timestamp: 2 },
        ],
      },
      () => {
        const rmse = computeBaselineRMSE(trainSet);
        if (!close(rmse, 1)) throw new Error(`rmse = ${rmse}, expected 1`);
        return `rmse=${fmt(rmse)}`;
      },
    ),
  );

  // 2. Shared helper: zero-denominator guard.
  await check("computeBaselineRMSE: empty split returns 0 (no division by zero)", () =>
    withFixture({ train: [{ userId: 1, movieId: 1, rating: 3, timestamp: 1 }] }, () => {
      if (computeBaselineRMSE([]) !== 0) throw new Error("expected 0");
      return "guarded";
    }),
  );

  // 3. Formatting helper.
  await check("fmt: formats numbers and guards non-finite input", () => {
    if (fmt(1.23456) !== "1.2346") throw new Error(`fmt(1.23456) = ${fmt(1.23456)}`);
    if (fmt(2, 2) !== "2.00") throw new Error(`fmt(2, 2) = ${fmt(2, 2)}`);
    if (fmt(Number.NaN) !== "\u2014") throw new Error("NaN must render as an em dash");
    return "ok";
  });

  // 4. Formatting helper.
  await check("movieTitle: resolves ids and falls back for unknown ids", () =>
    withFixture({ movies: [{ id: 7, title: "Seven Samurai", genres: [] }] }, () => {
      if (movieTitle(7) !== "Seven Samurai") throw new Error("known id failed");
      if (movieTitle(999) !== "Movie #999") throw new Error(`fallback = ${movieTitle(999)}`);
      return "ok";
    }),
  );

  // 5. Student stub: prediction and clamp (TF.js gather + dot).
  await check("predictRating: clamps to [1, 5] and falls back to the global mean", () =>
    withFixture(
      {
        movies: TOY_MOVIES,
        train: TOY_RATINGS,
        model: fixtureModel({ 1: [1, 2] }, { 3: [3, 4] }, 3),
      },
      () => {
        const pred = predictRating(1, 3);
        if (pred !== 5) throw new Error(`predictRating = ${pred}, expected 5 (3 + 11 clamped)`);
        const fallback = predictRating(100000, 3);
        if (fallback !== 3) throw new Error(`out-of-range user must return globalMean, got ${fallback}`);
        return `clamped=${pred}, fallback=${fallback}`;
      },
    ),
  );

  // 6. Student stub: TF.js variable allocation on the toy 5-rating set.
  await check("initializeFactors: allocates tf.Variable factors and the global mean", () =>
    withFixture({ movies: TOY_MOVIES, train: TOY_RATINGS, model: null }, () => {
      const created = initializeFactors(TOY_RATINGS, 4);
      if (!created || !model) throw new Error("model was not assigned");
      if (!close(model.globalMean, 3.8)) {
        throw new Error(`globalMean = ${model.globalMean}, expected 3.8`);
      }
      if (!isTfModel(model)) throw new Error("userFactors/itemFactors are not tf tensors");
      if (model.userFactors.shape[1] !== 4 || model.itemFactors.shape[1] !== 4) {
        throw new Error(
          `factor width = ${model.userFactors.shape[1]} / ${model.itemFactors.shape[1]}, expected 4`,
        );
      }
      if (model.itemFactors.shape[0] !== 4) {
        throw new Error(`item rows = ${model.itemFactors.shape[0]}, expected 4 (max movie id 3 + 1)`);
      }
      if (model.userFactors.shape[0] < 3) {
        throw new Error(`user rows = ${model.userFactors.shape[0]}, expected at least 3`);
      }
      const row = tf.tidy(() => Array.from(model.userFactors.slice([1, 0], [1, 4]).dataSync()));
      if (!Array.from(row).every(Number.isFinite)) {
        throw new Error("user factor row is not finite");
      }
      return `user [${model.userFactors.shape}], item [${model.itemFactors.shape}]`;
    }),
  );

  // 7. Student stub: batched training loop.
  await check("trainMF: returns one finite history row per epoch", () =>
    withFixture({ movies: TOY_MOVIES, train: TOY_RATINGS, val: TOY_RATINGS, model: null }, async () => {
      initializeFactors(TOY_RATINGS, 4);
      const tensorsBefore = tf.memory().numTensors;
      let progressCount = 0;
      const history = await trainMF(TOY_RATINGS, TOY_RATINGS, {
        epochs: 3,
        lr: 0.02,
        reg: 0.05,
        batchSize: 2,
        onEpoch: () => { progressCount++; },
      });
      if (progressCount !== 3 || tf.memory().numTensors !== tensorsBefore) {
        throw new Error('Missing epoch progress or leaked training tensors');
      }
      if (!Array.isArray(history) || history.length !== 3) {
        throw new Error(`history length = ${history && history.length}, expected 3`);
      }
      history.forEach((row, i) => {
        if (row.epoch !== i + 1 && row.epoch !== i) {
          throw new Error(`row ${i} epoch = ${row.epoch}`);
        }
        if (!Number.isFinite(row.trainRMSE) || !Number.isFinite(row.valRMSE)) {
          throw new Error(`row ${i} RMSE is not finite`);
        }
      });
      return history.map((r) => `e${r.epoch}:${fmt(r.trainRMSE)}`).join(" ");
    }),
  );

  // 8. Student stub: checkpoint selection and tie-break.
  await check("selectBestCheckpoint: smallest valRMSE wins, ties keep the earliest epoch", () => {
    const history = [
      { epoch: 2, trainRMSE: 0.9, valRMSE: 0.4 },
      { epoch: 1, trainRMSE: 1.0, valRMSE: 0.3 },
      { epoch: 3, trainRMSE: 0.8, valRMSE: 0.3 },
    ];
    const best = selectBestCheckpoint(history);
    if (!best || best.epoch !== 1) {
      throw new Error(`best epoch = ${best && best.epoch}, expected 1 (tie -> earliest)`);
    }
    return `best epoch ${best.epoch}`;
  });

  // 9. Student stub: recommendation filtering (tensor matMul + topk).
  await check("recommendTopK: excludes train+val items but allows test-rated items", () =>
    withFixture(
      {
        movies: REC_FIXTURE.movies,
        train: REC_FIXTURE.train,
        val: REC_FIXTURE.val,
        test: REC_FIXTURE.test,
        model: fixtureModel({ 1: 1 }, { 1: 5, 2: 4, 3: 3, 4: 2, 5: 1 }),
      },
      () => {
        const recs = recommendTopK(1, 10);
        const ids = recs.map((r) => r.movieId);
        if (ids.includes(1)) throw new Error("movie 1 is in train and must be excluded");
        if (ids.includes(2)) throw new Error("movie 2 is in val and must be excluded");
        if (!ids.includes(3)) throw new Error("movie 3 is test-rated and must be eligible");
        for (let i = 1; i < recs.length; i++) {
          if (recs[i - 1].score < recs[i].score) {
            throw new Error("results are not sorted by descending score");
          }
        }
        return ids.join(", ");
      },
    ),
  );

  // 10. Student stub: recall at rank 1 vs rank 11.
  await check("recallAtK: relevant movie at rank 1 scores 1.0; at rank 11 scores 0.0", async () => {
    const before = tf.memory().numTensors;
    let progress = 0;
    const rankOne = await withFixture({ ...recallFixture(1), history: [] }, () => recallAtK(10, {
      batchSize: 1, onProgress: ({ completed, total }) => {
        if (completed !== 1 || total !== 1) throw new Error('Incorrect recall progress');
        progress++;
      },
    }));
    if (!close(rankOne.overallRecall, 1)) {
      throw new Error(`rank-1 recall = ${rankOne.overallRecall}, expected 1`);
    }
    const rankEleven = await withFixture({ ...recallFixture(11), history: [] }, () => recallAtK(10));
    if (!close(rankEleven.overallRecall, 0)) {
      throw new Error(`rank-11 recall = ${rankEleven.overallRecall}, expected 0`);
    }
    if (progress !== 1 || tf.memory().numTensors !== before) throw new Error('Missing progress or leaked recall tensors');
    return `rank1=${fmt(rankOne.overallRecall)} rank11=${fmt(rankEleven.overallRecall)}`;
  });

  // 11. Student stub: empty-result guard.
  await check("recallAtK: empty test set returns 0 without dividing by zero", () =>
    withFixture(
      {
        movies: REC_FIXTURE.movies,
        train: REC_FIXTURE.train,
        val: [],
        test: [],
        model: fixtureModel({ 1: 1 }, { 1: 1, 2: 1, 3: 1, 4: 1, 5: 1 }),
      },
      async () => {
        const result = await recallAtK(10);
        if (result.overallRecall !== 0) {
          throw new Error(`expected 0, got ${result.overallRecall}`);
        }
        if (!Array.isArray(result.perUser) || result.perUser.length !== 0) {
          throw new Error("perUser must be empty");
        }
        if (!result.excluded) throw new Error("excluded report missing");
        return "guarded";
      },
    ),
  );

  const passed = checks.filter((c) => c.status === "PASS").length;
  const failed = checks.filter((c) => c.status === "FAIL").length;
  const pendingCount = checks.filter((c) => c.status === "PENDING").length;

  const lines = [
    `HW5 self-checks — pass ${passed}, fail ${failed}, pending ${pendingCount} (of ${checks.length})`,
    "",
    ...checks.map((c) => `${c.status.padEnd(7)} ${c.name}${c.detail ? ` \u2014 ${c.detail}` : ""}`),
  ];
  const text = lines.join("\n");

  const target = logElement || document.getElementById("testLog");
  if (target) target.textContent = text;

  return { passed, failed, pending: pendingCount, checks };
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

/**
 * Select the TF.js backend, load the MovieLens data, split it chronologically,
 * and wire the page. `data.js` must be loaded before this script.
 *
 * If TensorFlow.js is missing, `initTf` renders a clear message and this
 * function stops without throwing.
 *
 * @returns {Promise<void>}
 */
window.onload = async function () {
  const status = document.getElementById("status");
  const backend = await initTf();
  if (!backend) return;

  try {
    if (status) status.textContent = "Loading data\u2026";
    await loadData();
    const split = splitByTimestamp();
    trainSet = split.train;
    valSet = split.val;
    testSet = split.test;
    init();
  } catch (error) {
    console.error("Initialization failed:", error);
    if (status) status.textContent = `Error: ${error.message}`;
  }
};
