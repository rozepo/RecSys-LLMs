You are an expert full-stack web developer who creates robust, well-commented, and modular web applications using only vanilla HTML, CSS, and JavaScript.

Your task is to generate the complete code for a "Content-Based Movie Recommender" web application based on the detailed specifications below. The application logic will be split into two separate JavaScript files: `data.js` for data loading and parsing, and `script.js` for UI and recommendation logic. Please provide the code for each of the four files—`index.html`, `style.css`, `data.js`, and `script.js`—separately and clearly labeled.

---

### **Project Specification: Content-Based Movie Recommender (Modular)**

#### **1. Overall Goal**

Build a single-page web application that recommends movies. Use `data.js` to load and parse local files (`u.item`, `u.data`) and `script.js` for the UI and recommendation logic. Support two modes: recommendations from one selected movie and recommendations from a profile built from exactly three distinct selected movies. Rank candidates by cosine similarity over 19-dimensional genre vectors and display Top-5 recommendations with their genres and scores. Ratings are loaded but are not used in scoring or tie-breaking.

#### **2. File `index.html` - The Application Structure**

-   **DOCTYPE and Language:** The document should start with `<!DOCTYPE html>` and the `<html>` tag should specify `lang="en"`.
-   **Title:** The page title should be "Content-Based Movie Recommender".
-   **Main Heading:** Include an `<h1>` with the text "Content-Based Movie Recommender".
-   **Instructions:** Explain that the user can select one movie or build a profile from exactly three distinct movies.
-   **Mode Selector:** Include a `<select id="recommendation-mode">` with values `single` and `profile`; default to `single`.
-   **Movie Selectors:** Include three labeled `<select>` elements with IDs `movie-select`, `movie-select-2`, and `movie-select-3`, each populated dynamically with movies and an empty placeholder. In single mode, show and enable only the first selector; in profile mode, show and enable all three. Ignore inactive selections.
-   **Button:** Include a `<button>` with the text "Get Recommendations". When clicked, it must call the `getRecommendations()` JavaScript function.
-   **Result Display Area:** Include a `<div id="result-box">` containing a `<p id="result">` for status messages and a query summary, and a separate `<ol id="recommendations">` for recommendation entries. Each entry must show the movie title, its genres, and its similarity score.
-   **File Linking:** This is a critical step. At the end of the `<body>`, link to **both** JavaScript files. `data.js` must be loaded **before** `script.js` because `script.js` depends on the functions and variables defined in `data.js`.
    ```
    <script src="data.js"></script>
    <script src="script.js"></script>
    ```

#### **3. File `style.css` - The Application Design**

-   **Layout:** Create a professional, modern, and user-friendly layout. All content should be centered on the page within a main container.
-   **Background:** The `<body>` should have a light, neutral background color (e.g., `#f4f7f6`).
-   **Container:** The main container holding all elements should have a white background, rounded corners (`border-radius`), and a subtle box shadow to make it pop.
-   **Typography:** Use a clean, sans-serif font like 'Helvetica' or 'Arial'.
-   **Controls:** The mode selector, movie selectors, and button should have consistent styling, with adequate padding and a clear visual hierarchy.
-   **Button:** The button should be inviting, with a distinct background color (e.g., a shade of blue), white text, and a hover effect (e.g., slightly darker background) to indicate interactivity.
-   **Result Area:** The `#result-box` should have padding and a light background. The `#result` status text should be bold; recommendation titles, genres, and scores in `#recommendations` should be easy to read.
-   **Status Colors:** Loading, error, and success colors must actually override any base `#result` color. Use selectors such as `#result.loading`, `#result.error`, and `#result.success`; plain `.error` or `.success` selectors are insufficient when `#result` sets `color`.

#### **4. File `data.js` - The Data Handling Module**

This file is responsible only for fetching and parsing the data from local files.

1.  **Global Variables:**
    -   Declare two global `let` variables, `movies` and `ratings`, initialized as empty arrays.

2.  **Primary Function: `loadData()`**
    -   This must be an `async` function.
    -   It will use the `fetch()` API to read `u.item` and `u.data`. Assume these files are in the same directory as `index.html`.
    -   Check `response.ok` for both files. Use `try...catch` to propagate loading or parsing errors with context; let `script.js` display the error in `#result` and apply the error status class.
    -   First fetch `u.item`, which is encoded as **ISO-8859-1**. Read it with `await response.arrayBuffer()`, decode it using `new TextDecoder('iso-8859-1').decode(buffer)`, and pass the decoded text to `parseItemData`. Do not use `response.text()` for `u.item`.
    -   Then fetch `u.data`, read it with `await response.text()`, and pass it to `parseRatingData`.
    -   Keep both parsed arrays local until both files have been successfully loaded and parsed. Treat an empty result from either parser as an error.
    -   On success, replace the global `movies` and `ratings` arrays with the newly parsed arrays; never append to previous loads. On failure, leave both previous arrays unchanged. Repeated loading must not duplicate records or leave partially updated data.
    -   The returned Promise resolves only after both arrays have been replaced and rejects on any loading or parsing failure.

3.  **Parsing Function: `parseItemData(text)`**
    -   Accept the decoded text and return a new array without mutating global state. Split into lines, handle LF/CRLF, and skip blank lines. Throw an error if the resulting array is empty.
    -   Use all **19** genre names in the exact order defined by MovieLens `u.genre`: `unknown`, `Action`, `Adventure`, `Animation`, `Children's`, `Comedy`, `Crime`, `Documentary`, `Drama`, `Fantasy`, `Film-Noir`, `Horror`, `Musical`, `Mystery`, `Romance`, `Sci-Fi`, `Thriller`, `War`, `Western`.
    -   Split each nonblank line by `|`. A valid `u.item` record has 24 fields. All field indices below are zero-based: ID is field 0, title is field 1, and the **19 binary genre flags occupy fields 5 through 23 inclusive**.
    -   Validate the field count, a positive integer ID, a nonempty title, and flags that are exactly `0` or `1`; report malformed records as parsing errors.
    -   Build `genreVector` as the 19 numeric flags in that order. For every index `i` from 0 through 18, map field `5 + i` to genre name `i`. Include `unknown` at index 0 and `Western` at index 18; do not shift or drop any flag.
    -   Build `genres` from the names whose corresponding flags are 1. Return movie objects `{ id, title, genres, genreVector }`.

4.  **Parsing Function: `parseRatingData(text)`**
    -   Accept the raw text and return a new array without mutating global state. Split into lines, handle LF/CRLF, and skip blank lines. Throw an error if the resulting array is empty.
    -   Split each nonblank line by `\t` into exactly four fields and create `{ userId, itemId, rating, timestamp }` with numeric values. Validate positive integer IDs, an integer rating from 1 to 5, and a nonnegative integer timestamp; report malformed records as parsing errors.

#### **5. File `script.js` - The UI and Logic Module**

This file handles the user interface and the recommendation logic. It will depend on the data loaded by `data.js`.

1.  **Initialization Logic:**
    -   Use `window.onload` to create an `async` function that initializes the application.
    -   Show a loading status and disable recommendation controls while awaiting `loadData()` from `data.js`.
    -   After successful loading, call `populateMoviesDropdown()`, enable the controls appropriate to the current mode, and show instructions for selecting one movie or three distinct movies.
    -   Catch loading/parsing failures, show the error in `#result` with the error status class, clear old recommendation entries, and do not show a success message.
    -   Handle mode changes by updating selector visibility and enabled state and clearing stale results. Any reload must use the same loading/error handling and repopulate selectors only after success.

2.  **UI Function: `populateMoviesDropdown()`**
    -   Populate all three movie selectors by their IDs. Clear their old options and recreate an empty placeholder before adding movies, so repeated population does not duplicate entries.
    -   Sort a copy of `movies` alphabetically by title for display. Create an `<option>` for each movie, setting its `value` to the movie ID and its text to the title.
    -   Dropdown order and the underlying array order must not determine recommendation tie-breaking.

3.  **Core Logic: `getRecommendations()`**
    -   This is the main function for content-based filtering, triggered by the button click. It must perform the following steps in order:
        -   **Step 1 (Get User Input):** Read the mode and active movie selectors. Require exactly one valid movie ID in single mode or exactly three distinct valid movie IDs in profile mode. Resolve every ID against `movies`; on missing, duplicate, or invalid selections, clear old recommendations, display an error, and exit.
        -   **Step 2 (Build Query Vector):** In single mode, use the selected movie's 19-dimensional binary `genreVector`. In profile mode, use the component-wise arithmetic mean of the three selected vectors: `q[i] = (v1[i] + v2[i] + v3[i]) / 3` for `i = 0..18`. Keep fractional values; do not binarize the profile or normalize each input vector before averaging.
        -   **Step 3 (Prepare Candidates):** Exclude every selected movie ID from `candidateMovies`, in both modes.
        -   **Step 4 (Calculate Scores):** Compute cosine similarity between query vector `q` and each candidate's binary vector `v`: `score = sum(q[i] * v[i]) / (sqrt(sum(q[i] ** 2)) * sqrt(sum(v[i] ** 2)))`, summing over all 19 dimensions. If the query norm is zero, show an explicit message that genre-based recommendations cannot be computed and stop. If a candidate norm is zero, assign score 0 without division. Never produce `NaN` or `Infinity`. Store `{ ...candidate, score }` and retain full precision for ranking.
        -   **Step 5 (Rank and Break Ties):** Sort by descending score. For equal scores, use a deterministic query-dependent pseudorandom permutation as specified below, never input-array order, title order, or ascending ID as the tie-breaker.
        -   **Step 6 (Select Top Recommendations):** Take the first five ranked candidates with `.slice(0, 5)`, or all remaining candidates if fewer than five exist.
        -   **Step 7 (Display Result):** Set a query summary in `#result` naming the selected movie(s). Render one entry per recommendation in `#recommendations`, showing its title, all its `genres` (or an explicit no-genres label), and its cosine `score` formatted to four decimal places. Round only for display. Use text-safe DOM APIs. Clear previous entries before rendering; if there are no candidates, show an explicit no-recommendations message.

4.  **Deterministic, Query-Dependent Tie-Breaking:**
    -   Canonicalize the request as `mode + ':' + selectedIdsSortedNumerically.join(',')`. Reordering the same three selected movies must not change the profile or recommendation order.
    -   Derive an unsigned 32-bit seed from that ASCII key using FNV-1a (initial value `2166136261`, XOR each character code, then multiply by `16777619` using `Math.imul`, keeping unsigned 32-bit results). Replace a zero seed with `0x9e3779b9`.
    -   Start from candidates sorted by numeric ID solely to establish a canonical input independent of file/UI order. Apply a seeded Fisher-Yates shuffle using xorshift32: update state with `state ^= state << 13`, `state ^= state >>> 17`, `state ^= state << 5`, then use `(state >>> 0) / 4294967296` as the next random value. For each shuffle index `i` descending from `length - 1` to 1, swap with index `floor(random * (i + 1))`.
    -   Assign each candidate its unique position in this permutation as `tieRank`. Rank by descending unrounded score, then ascending `tieRank`. Do not use unseeded `Math.random()`, timestamps, or a random sort comparator.
    -   The same query and candidate dataset must yield the same results across repeated calls and page reloads, even if input arrays are reordered. Different queries derive their own pseudorandom orders, with no systematic preference for small IDs.

---
Please now generate the complete code for the `index.html`, `style.css`, `data.js`, and `script.js` files based on these final, detailed specifications.
