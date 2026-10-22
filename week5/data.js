// ---------------------------------------------------------------------------
// week5/data.js — MovieLens 100k data layer (classic <script>, no modules)
//
// Loads u.item (movies) and u.data (ratings), builds a sparse-friendly
// rating matrix, and splits every user's ratings chronologically 80/10/10.
//
// Genre alignment note (fixes the Week-2 off-by-one):
//   A u.item row has 24 '|'-separated fields:
//     0 id | 1 title | 2 release | 3 video | 4 url | 5 unknown |
//     6 Action | 7 Adventure | ... | 23 Western
//   Field index 5 is the legacy 19th genre flag "unknown", NOT the first real
//   genre. The 18 real flags therefore live at indices 6..23, so we read them
//   with fields.slice(6, 24) and pair them with the 18-name `genreNames`
//   list below. (The earlier fields.slice(5, 24) + 18 names pairing was off
//   by one.)
// ---------------------------------------------------------------------------

// Global variables for storing movie and rating data
let movies = [];          // [{ id, title, genres: string[] }]
let ratings = [];         // [{ userId, movieId, rating, timestamp }]

// Collaborative filtering structures (populated by buildRatingMatrix)
let numUsers = 0;         // highest user id found in u.data
let numMovies = 0;        // number of parsed movies
let ratingMatrix = null;  // (numUsers + 1) x (maxMovieId + 1); null = unobserved

// Genre names for the 18 real flags (u.item field indices 6..23)
const genreNames = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// Users skipped while splitting (e.g. too few ratings), collected by
// splitByTimestamp() for reporting.
let SPLIT_EXCLUSIONS = [];

// Primary function to load data from files
async function loadData() {
    try {
        // Load and parse movie data
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesText = new TextDecoder('iso-8859-1').decode(await moviesResponse.arrayBuffer());
        parseItemData(moviesText);

        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);

        // Derive matrix dimensions, then build the rating matrix
        numUsers = ratings.reduce((max, r) => Math.max(max, r.userId), 0);
        numMovies = movies.length;
        buildRatingMatrix();
    } catch (error) {
        console.error('Error loading data:', error);
        const errorTarget = document.getElementById('result');
        if (errorTarget) {
            errorTarget.textContent =
                `Error: ${error.message}. Please make sure u.item and u.data are in the correct location.`;
            errorTarget.className = 'error';
        }
        throw error; // Re-throw so the page script can handle the error
    }
}

// Parse movie data from u.item format: "id|title|release|video|url|g0|...|g18"
// genreNames has 18 entries, matched against the real flags at indices 6..23.
function parseItemData(text) {
    const lines = text.split('\n');
    const parsed = [];
    const ids = new Set();

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.trimEnd().split('|');
        if (fields.length !== 24) throw new Error('Invalid u.item field count');

        const id = Number(fields[0]);
        const title = fields[1];
        if (!Number.isInteger(id) || id <= 0 || ids.has(id) || !title ||
            fields.slice(5).some(flag => flag !== '0' && flag !== '1')) {
            throw new Error(`Invalid or duplicate movie: ${fields[0]}`);
        }
        ids.add(id);

        // Read the 18 real genre flags (index 5 is the 'unknown' placeholder)
        const genreValues = fields.slice(6, 24).map(value => parseInt(value));
        const genres = genreNames.filter((_, index) => genreValues[index] === 1);
        if (fields[5] === '1') genres.unshift('unknown');

        parsed.push({ id, title, genres });
    }
    movies = parsed;
}

// Parse rating data from u.data format: "userId\tmovieId\trating\ttimestamp"
function parseRatingData(text) {
    const lines = text.split('\n');
    const parsed = [];
    const pairs = new Set();
    const movieIds = new Set(movies.map(m => m.id));

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.trimEnd().split('\t');
        if (fields.length !== 4) throw new Error('Invalid u.data field count');

        const [userId, movieId, rating, timestamp] = fields.map(Number);
        const pair = `${userId}:${movieId}`;
        if (![userId, movieId, rating, timestamp].every(Number.isInteger) ||
            userId <= 0 || !movieIds.has(movieId) || rating < 1 || rating > 5 ||
            timestamp <= 0 || pairs.has(pair)) throw new Error(`Invalid or duplicate rating: ${pair}`);
        pairs.add(pair);

        parsed.push({ userId, movieId, rating, timestamp });
    }
    ratings = parsed;
}

// Build the sparse-friendly rating matrix.
// Shape: (numUsers + 1) x (maxMovieId + 1), indexed by raw ids, so that
//   ratingMatrix[userId][movieId] === rating
// and an unobserved pair is null. This descriptive matrix is NEVER training input.
function buildRatingMatrix() {
    const maxMovieId = movies.reduce((max, m) => Math.max(max, m.id), 0);

    ratingMatrix = [];
    for (let u = 0; u <= numUsers; u++) {
        ratingMatrix.push(new Array(maxMovieId + 1).fill(null));
    }

    for (const r of ratings) {
        if (r.userId <= numUsers && r.movieId <= maxMovieId) {
            ratingMatrix[r.userId][r.movieId] = r.rating;
        }
    }
}

// Split EACH user's ratings by ascending timestamp into 80/10/10.
// Returns { train, val, test }, each an array of the same
// { userId, movieId, rating, timestamp } objects.
//
// Rules:
//   - counts are deterministic: nTrain = floor(0.8 * n), nVal = floor(0.1 * n),
//     nTest = n - nTrain - nVal (the remainder)
//   - users with fewer than 5 ratings are skipped and recorded in
//     SPLIT_EXCLUSIONS
//   - timestamp ties are broken by movieId so the split is deterministic
function splitByTimestamp() {
    const MIN_RATINGS = 5;

    const byUser = new Map();
    for (const r of ratings) {
        if (!byUser.has(r.userId)) byUser.set(r.userId, []);
        byUser.get(r.userId).push(r);
    }

    const train = [];
    const val = [];
    const test = [];

    SPLIT_EXCLUSIONS = [];

    for (const [userId, userRatings] of byUser) {
        const n = userRatings.length;
        if (n < MIN_RATINGS) {
            SPLIT_EXCLUSIONS.push({
                userId,
                count: n,
                reason: `fewer than ${MIN_RATINGS} ratings`
            });
            continue;
        }

        const sorted = userRatings.slice().sort((a, b) =>
            a.timestamp - b.timestamp || a.movieId - b.movieId
        );

        const nTrain = Math.floor(0.8 * n);
        const nVal = Math.floor(0.1 * n);

        for (let i = 0; i < n; i++) {
            if (i < nTrain) train.push(sorted[i]);
            else if (i < nTrain + nVal) val.push(sorted[i]);
            else test.push(sorted[i]);
        }
    }

    return { train, val, test };
}
