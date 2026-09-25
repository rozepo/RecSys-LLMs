// Global variables for storing movie and rating data
let movies = [];
let ratings = [];

// All 19 flags in u.genre order, corresponding to u.item fields 5–23.
const genreNames = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// Primary function to load data from files
async function loadData() {
    try {
        // Load and parse movie data
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesBuffer = await moviesResponse.arrayBuffer();
        const moviesText = new TextDecoder('iso-8859-1').decode(moviesBuffer);
        const parsedMovies = parseItemData(moviesText);

        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        const parsedRatings = parseRatingData(ratingsText);

        // Commit both datasets together; repeated loads replace, never append.
        movies = parsedMovies;
        ratings = parsedRatings;
    } catch (error) {
        throw new Error(`Unable to load data: ${error.message}`);
    }
}

// Reject partial numbers, empty fields, fractions and unsafe integers.
function parseIntegerField(value, minimum, maximum, context) {
    const number = Number(value);
    if (!/^\d+$/.test(value) || !Number.isSafeInteger(number) ||
        number < minimum || number > maximum) {
        throw new Error(`Invalid ${context}`);
    }
    return number;
}

// Parse movie data from u.item format
function parseItemData(text) {
    const lines = text.split(/\r?\n/);
    const parsedMovies = [];
    const seenIds = new Set();

    for (const [index, line] of lines.entries()) {
        if (line.trim() === '') continue;

        const fields = line.split('|');
        const context = `u.item line ${index + 1}`;
        if (fields.length !== 24) throw new Error(`Expected 24 fields at ${context}`);

        const id = parseIntegerField(fields[0], 1, Number.MAX_SAFE_INTEGER, context);
        const title = fields[1];
        const flags = fields.slice(5, 24);
        if (!title.trim() || seenIds.has(id) ||
            flags.some(flag => flag !== '0' && flag !== '1')) {
            throw new Error(`Invalid title, duplicate ID or genre flags at ${context}`);
        }

        const genreVector = flags.map(Number);
        const genres = genreNames.filter((_, i) => genreVector[i] === 1);
        seenIds.add(id);
        parsedMovies.push({ id, title, genres, genreVector });
    }
    if (parsedMovies.length === 0) throw new Error('No movies parsed from u.item');
    return parsedMovies;
}

// Parse rating data from u.data format
function parseRatingData(text) {
    const lines = text.split(/\r?\n/);
    const parsedRatings = [];

    for (const [index, line] of lines.entries()) {
        if (line.trim() === '') continue;

        const fields = line.split('\t');
        const context = `u.data line ${index + 1}`;
        if (fields.length !== 4) throw new Error(`Expected 4 fields at ${context}`);

        const userId = parseIntegerField(fields[0], 1, Number.MAX_SAFE_INTEGER, context);
        const itemId = parseIntegerField(fields[1], 1, Number.MAX_SAFE_INTEGER, context);
        const rating = parseIntegerField(fields[2], 1, 5, context);
        const timestamp = parseIntegerField(fields[3], 0, Number.MAX_SAFE_INTEGER, context);

        parsedRatings.push({ userId, itemId, rating, timestamp });
    }
    if (parsedRatings.length === 0) throw new Error('No ratings parsed from u.data');
    return parsedRatings;
}
