"""
Tech Stack Recommender
-----------------------
Project 3 Capstone — AI Recommendation Logic (Content-Based Filtering)

Pipeline (as taught in the deck):
  1. INPUT      -> Capture user state (min. 3 skills / interests)
  2. PROCESS    -> Vector Mapping (TF-IDF) + Cosine Similarity scoring
  3. OUTPUT     -> Sort results, filter to Top-N

Run:
    pip install -r requirements.txt
    python recommender.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# STEP 1: INGESTION — Load the item dataset (job roles = "items")
# ---------------------------------------------------------------------------
def load_dataset(path: str = "raw_skills.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize the skills text: lowercase, strip whitespace, comma-separated -> space-separated
    # (TF-IDF treats each row as a "document" of words)
    df["skills_clean"] = (
        df["skills"]
        .str.lower()
        .str.replace(",", " ", regex=False)
        .str.replace("  ", " ", regex=False)
    )
    return df


# ---------------------------------------------------------------------------
# STEP 2: SCORING — Vector Mapping + Cosine Similarity
# ---------------------------------------------------------------------------
def build_recommendation_engine(df: pd.DataFrame):
    """
    Fits a TF-IDF vectorizer on the item corpus (job roles).
    Returns the fitted vectorizer and the resulting item-feature matrix.
    """
    vectorizer = TfidfVectorizer()
    item_matrix = vectorizer.fit_transform(df["skills_clean"])
    return vectorizer, item_matrix


def score_user_profile(user_skills: list[str], vectorizer: TfidfVectorizer, item_matrix):
    """
    Transforms the user's raw skill input into the SAME TF-IDF vocabulary
    space as the items, then computes cosine similarity against every item.
    """
    # Join user skills into a single "document" matching the item format
    user_doc = " ".join(skill.lower().strip() for skill in user_skills)

    # IMPORTANT: use transform (not fit_transform) so the user vector
    # maps onto the exact same vocabulary dimensions as the items.
    user_vector = vectorizer.transform([user_doc])

    # Cosine similarity between user vector and every item vector
    scores = cosine_similarity(user_vector, item_matrix).flatten()
    return scores


# ---------------------------------------------------------------------------
# STEP 3 & 4: SORTING + FILTERING — Rank and truncate to Top-N
# ---------------------------------------------------------------------------
def get_top_n_recommendations(df: pd.DataFrame, scores, n: int = 3) -> pd.DataFrame:
    results = df.copy()
    results["similarity_score"] = scores
    results = results.sort_values(by="similarity_score", ascending=False)
    results = results[results["similarity_score"] > 0]  # filter out zero-score noise
    return results.head(n)[["job_role", "skills", "similarity_score"]]


# ---------------------------------------------------------------------------
# Cold Start handling (bonus, matches the deck's "Bypassing the Cold Start")
# ---------------------------------------------------------------------------
def handle_cold_start(df: pd.DataFrame, top_scores) -> pd.DataFrame:
    """If no meaningful match is found (all-zero vector), fall back to
    a 'Trending' list — here approximated as the first N roles."""
    if top_scores.empty:
        print("\n⚠️  No strong match found (Cold Start). Showing trending roles instead:\n")
        return df.head(3)[["job_role", "skills"]]
    return top_scores


# ---------------------------------------------------------------------------
# MAIN — Input-Process-Output flow
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  DecodeLabs — AI Tech Stack Recommender (Project 3)")
    print("=" * 60)

    # --- INPUT ---
    df = load_dataset("raw_skills.csv")
    print(f"\nLoaded {len(df)} job roles from raw_skills.csv\n")

    print("Enter at least 3 skills or interests, separated by commas.")
    print("Example: Python, Cloud Computing, Automation\n")
    raw_input_str = input("Your skills: ").strip()

    user_skills = [s.strip() for s in raw_input_str.split(",") if s.strip()]

    if len(user_skills) < 3:
        print("\n⚠️  Please provide at least 3 skills for accurate matching. Using defaults.")
        user_skills = ["Python", "Cloud Computing", "Automation"]

    # --- PROCESS ---
    vectorizer, item_matrix = build_recommendation_engine(df)
    scores = score_user_profile(user_skills, vectorizer, item_matrix)

    # --- OUTPUT ---
    top_matches = get_top_n_recommendations(df, scores, n=3)
    top_matches = handle_cold_start(df, top_matches)

    print("\n" + "-" * 60)
    print(f"Top career matches for: {', '.join(user_skills)}")
    print("-" * 60)
    for i, row in enumerate(top_matches.itertuples(), start=1):
        score_display = f"{row.similarity_score:.2%}" if hasattr(row, "similarity_score") else "N/A"
        print(f"{i}. {row.job_role}  (match: {score_display})")
        print(f"   Skills: {row.skills}\n")


if __name__ == "__main__":
    main()