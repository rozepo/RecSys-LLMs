const movieSelectIds = ['movie-select', 'movie-select-2', 'movie-select-3'];
let dataReady = false;

// Initialize once; initializeApp() also provides the same UI handling for reloads.
window.onload = async function() {
    document.getElementById('recommendation-mode').onchange = updateMode;
    movieSelectIds.forEach(id => {
        document.getElementById(id).onchange = resetResults;
    });
    await initializeApp();
};

async function initializeApp() {
    dataReady = false;
    updateMode();
    setStatus('Loading movie data...', 'loading');
    try {
        await loadData();
        populateMoviesDropdown();
        dataReady = true;
        updateMode();
    } catch (error) {
        setStatus(error.message, 'error');
    }
}

function setStatus(message, status) {
    const resultElement = document.getElementById('result');
    resultElement.textContent = message;
    resultElement.className = status;
}

function resetResults() {
    document.getElementById('recommendations').replaceChildren();
    if (dataReady) {
        const profile = document.getElementById('recommendation-mode').value === 'profile';
        setStatus(profile ? 'Select three distinct movies.' : 'Select a movie.', 'success');
    }
}

function updateMode() {
    const modeElement = document.getElementById('recommendation-mode');
    const profile = modeElement.value === 'profile';
    modeElement.disabled = !dataReady;
    document.getElementById('recommend-btn').disabled = !dataReady;
    movieSelectIds.forEach((id, index) => {
        const active = index === 0 || profile;
        document.getElementById(id).disabled = !dataReady || !active;
        if (index > 0) document.getElementById(`${id}-group`).hidden = !active;
    });
    resetResults();
}

// Populate the movies dropdown with sorted movie titles
function populateMoviesDropdown() {
    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));
    movieSelectIds.forEach(id => {
        const selectElement = document.getElementById(id);
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = 'Select a movie';
        placeholder.disabled = true;
        placeholder.selected = true;
        selectElement.replaceChildren(placeholder);
        sortedMovies.forEach(movie => {
            const option = document.createElement('option');
            option.value = movie.id;
            option.textContent = movie.title;
            selectElement.appendChild(option);
        });
    });
}

function vectorNorm(vector) {
    if (vector.length !== 19 || vector.some(value =>
        !Number.isFinite(value) || value < 0 || value > 1)) {
        throw new Error('Invalid 19-dimensional genre vector.');
    }
    return Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
}

function cosineSimilarity(query, candidate) {
    const denominator = vectorNorm(query) * vectorNorm(candidate);
    if (denominator === 0) return 0;
    const dot = query.reduce((sum, value, i) => sum + value * candidate[i], 0);
    return dot / denominator;
}

// Canonical input + seeded Fisher-Yates: ties never inherit file or UI order.
function shuffleCandidates(candidates, queryKey) {
    let state = 2166136261;
    for (const character of queryKey) {
        state = Math.imul(state ^ character.charCodeAt(0), 16777619) >>> 0;
    }
    if (state === 0) state = 0x9e3779b9;
    const shuffled = [...candidates].sort((a, b) => a.id - b.id);
    for (let i = shuffled.length - 1; i > 0; i--) {
        state ^= state << 13;
        state ^= state >>> 17;
        state ^= state << 5;
        const random = (state >>> 0) / 4294967296;
        const j = Math.floor(random * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}

// DOM-independent ranking, shared by both UI modes.
function rankMovies(mode, selectedIds, catalog = movies) {
    const required = mode === 'single' ? 1 : mode === 'profile' ? 3 : 0;
    if (!required || selectedIds.length !== required ||
        new Set(selectedIds).size !== required ||
        selectedIds.some(id => !Number.isSafeInteger(id) || id <= 0)) {
        throw new Error('Select one movie or exactly three distinct movies for a profile.');
    }
    const canonicalIds = [...selectedIds].sort((a, b) => a - b);
    const selectedMovies = canonicalIds.map(id => catalog.find(movie => movie.id === id));
    if (selectedMovies.some(movie => !movie)) {
        throw new Error('Selected movie not found in database.');
    }
    selectedMovies.forEach(movie => vectorNorm(movie.genreVector));
    const queryVector = Array.from({ length: 19 }, (_, i) =>
        selectedMovies.reduce((sum, movie) => sum + movie.genreVector[i], 0) / required);
    if (vectorNorm(queryVector) === 0) {
        throw new Error('Cannot compute genre-based recommendations for a zero genre vector.');
    }

    const selected = new Set(canonicalIds);
    const candidates = catalog.filter(movie => !selected.has(movie.id));
    const queryKey = `${mode}:${canonicalIds.join(',')}`;
    const scoredMovies = shuffleCandidates(candidates, queryKey).map((candidate, tieRank) => ({
        ...candidate,
        score: cosineSimilarity(queryVector, candidate.genreVector),
        tieRank
    }));
    scoredMovies.sort((a, b) => b.score - a.score || a.tieRank - b.tieRank);
    return { selectedMovies, queryVector, scoredMovies };
}

// Main recommendation function, triggered by the existing button.
function getRecommendations() {
    const list = document.getElementById('recommendations');
    list.replaceChildren();
    try {
        if (!dataReady) throw new Error('Movie data is not loaded.');
        const mode = document.getElementById('recommendation-mode').value;
        const activeSelectors = mode === 'profile' ? movieSelectIds : movieSelectIds.slice(0, 1);
        const selectedIds = activeSelectors.map(id => Number(document.getElementById(id).value));
        const { selectedMovies, scoredMovies } = rankMovies(mode, selectedIds);
        const topRecommendations = scoredMovies.slice(0, 5);
        if (topRecommendations.length === 0) {
            setStatus('No recommendations found: no candidate movies remain.', 'error');
            return [];
        }

        const titles = selectedMovies.map(movie => `"${movie.title}"`).join(', ');
        setStatus(`Because you liked ${titles}, we recommend:`, 'success');
        topRecommendations.forEach(movie => {
            const entry = document.createElement('li');
            const title = document.createElement('strong');
            title.textContent = movie.title;
            const details = document.createElement('span');
            details.className = 'recommendation-details';
            details.textContent = `Genres: ${movie.genres.join(', ') || 'No genres'} | Score: ${movie.score.toFixed(4)}`;
            entry.appendChild(title);
            entry.appendChild(details);
            list.appendChild(entry);
        });
        return topRecommendations;
    } catch (error) {
        setStatus(error.message, 'error');
        return [];
    }
}
