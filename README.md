# Tech Stack Recommender — Project 3 (DecodeLabs)

A content-based filtering recommendation engine that matches user skills to
career paths using **TF-IDF vector mapping** and **Cosine Similarity** — the
exact pipeline described in the Project 3 training deck (Ingestion → Scoring
→ Sorting → Filtering).

## Files
- `raw_skills.csv` — dataset of job roles ("items") and their associated skills
- `recommender.py` — the recommendation engine + CLI
- `requirements.txt` — Python dependencies

## Setup in VS Code

1. Open this folder in VS Code (`File > Open Folder`).
2. Open a terminal (`` Ctrl+` ``) and create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python recommender.py
   ```
5. When prompted, enter at least 3 skills, comma-separated, e.g.:
   ```
   Python, Cloud Computing, Automation
   ```

## How it maps to the deck's concepts

| Deck Concept | Where it happens in code |
|---|---|
| Input (User State) | `input()` capturing 3+ skills in `main()` |
| Vector Mapping | `TfidfVectorizer` in `build_recommendation_engine()` |
| TF-IDF weighting | Handled automatically by `TfidfVectorizer` |
| Cosine Similarity | `cosine_similarity()` in `score_user_profile()` |
| Sorting | `.sort_values()` in `get_top_n_recommendations()` |
| Filtering (Top-N) | `.head(n)` in `get_top_n_recommendations()` |
| Cold Start bypass | `handle_cold_start()` — falls back to trending roles |

## Customize it
- Add more job roles/skills to `raw_skills.csv` — no code changes needed.
- Change `n=3` in `main()` to return more or fewer recommendations.
- Swap the dataset for movies, products, or courses — the engine is domain-agnostic.
