# backend/src/cold_start.py

import numpy as np
import pandas as pd


class ColdStartHandler:
    """
    Handles recommendations for users with insufficient rating history.

    THREE STRATEGIES for CartIQ:
    ─────────────────────────────────────────────────────────────────
    1. POPULARITY FALLBACK (0 ratings)
       Bayesian average score across all products.
       Same formula used by IMDb Top 250.
       Prevents a product with 1 five-star review from outranking
       one with 10,000 near-perfect ratings.

    2. CATEGORY-WEIGHTED FALLBACK (1–4 ratings)
       Infer category preferences from sparse history.
       Score unseen products by category alignment.
       Blend with popularity to avoid obscure picks.

    3. THRESHOLD: 5+ ratings → caller should use SVD.
       Below 5, SVD latent vectors haven't converged meaningfully.
    ─────────────────────────────────────────────────────────────────

    AMAZON-SPECIFIC NOTE:
    Unlike MovieLens genres (19 binary flags), Amazon categories
    are strings ("Mice", "Headphones", "Cables").
    We build a category → product_idx map from the products_df
    and handle the string matching cleanly.
    """

    def __init__(self, min_votes=25):
        """
        Args:
            min_votes: Bayesian confidence weight.
                       A product needs roughly this many ratings
                       before its score is taken at face value.
        """
        self.min_votes = min_votes
        self.popularity_scores = None
        self.category_product_map = {}
        self.global_mean = None
        self.products_df = None
        self.ratings_df = None
        self.idx_to_product = {}
        self.product_to_idx = {}

    def fit(self, ratings_df, products_df):
        """
        Precomputes all scores and maps. Called once at startup.

        Args:
            ratings_df: Full filtered ratings DataFrame
            products_df: Product metadata DataFrame
        """
        self.ratings_df = ratings_df
        self.products_df = products_df
        self.global_mean = ratings_df["rating"].mean()

        # Store ID mappings from ratings_df attrs
        self.idx_to_product = ratings_df.attrs.get("idx_to_product", {})
        self.product_to_idx = ratings_df.attrs.get("product_map", {})

        self._compute_popularity_scores()
        self._build_category_map()

        print(f"✅ ColdStartHandler fitted")
        print(f"   Global mean: {self.global_mean:.3f}")
        print(f"   Categories indexed: {len(self.category_product_map)}")
        return self

    def _compute_popularity_scores(self):
        """
        Bayesian average score per product.

        Formula: (C × μ + Σr) / (C + n)
        where C = min_votes, μ = global mean,
              n = rating count, Σr = sum of ratings

        WHY this beats simple average:
        Product A: 1 rating of 5.0 → simple avg = 5.0 (misleading)
        Product A: 1 rating of 5.0 → Bayesian = 3.6 (pulled toward mean)
        Product B: 500 ratings avg 4.2 → Bayesian ≈ 4.2 (trusted)
        Product B correctly outranks Product A.
        """
        stats = (
            self.ratings_df
            .groupby("product_idx")["rating"]
            .agg(["count", "mean"])
            .rename(columns={"count": "n_ratings", "mean": "avg_rating"})
        )

        C = self.min_votes
        m = self.global_mean

        stats["bayesian_score"] = (
            (C * m + stats["avg_rating"] * stats["n_ratings"])
            / (C + stats["n_ratings"])
        )

        # Join with product metadata via ASIN mapping
        asin_series = pd.Series(self.idx_to_product, name="product_id")
        stats = stats.join(asin_series, how="left")

        if not self.products_df.empty:
            stats = stats.merge(
                self.products_df[["product_id", "title", "category", "price"]],
                on="product_id",
                how="left"
            )

        self.popularity_scores = stats.sort_values(
            "bayesian_score", ascending=False
        )

    def _build_category_map(self):
        """
        Maps category string → list of product_idx values,
        sorted by Bayesian score (most popular first).
        """
        if self.products_df.empty or "category" not in self.products_df.columns:
            return

        for _, row in self.products_df.iterrows():
            category = row.get("category", "Electronics")
            if not isinstance(category, str):
                continue

            product_id = row["product_id"]
            product_idx = self.product_to_idx.get(product_id)
            if product_idx is None:
                continue

            if category not in self.category_product_map:
                self.category_product_map[category] = []
            self.category_product_map[category].append(product_idx)

        # Sort each category list by Bayesian score
        score_lookup = self.popularity_scores["bayesian_score"].to_dict()
        for cat in self.category_product_map:
            self.category_product_map[cat].sort(
                key=lambda idx: score_lookup.get(idx, 0), reverse=True
            )

    def recommend_popular(self, n=10, exclude_idxs=None):
        """
        CASE 1: Pure new user — 0 ratings.
        Returns top Bayesian-scored products globally.

        Args:
            n: Number of recommendations
            exclude_idxs: product_idx values to exclude

        Returns:
            pd.DataFrame [product_idx, predicted_rating, title, category, reason]
        """
        exclude_idxs = exclude_idxs or set()

        recs = self.popularity_scores[
            ~self.popularity_scores.index.isin(exclude_idxs)
        ].head(n).reset_index()

        result = recs[["product_idx" if "product_idx" in recs.columns
                        else recs.columns[0]]].copy()

        # Rebuild cleanly
        rows = []
        for _, row in recs.iterrows():
            rows.append({
                "product_idx": row.name
                    if "product_idx" not in row else row["product_idx"],
                "predicted_rating": round(float(row["bayesian_score"]), 3),
                "title": row.get("title", ""),
                "category": row.get("category", "Electronics"),
                "price": row.get("price"),
                "reason": "Trending & highly rated"
            })

        return pd.DataFrame(rows)

    def recommend_by_category(self, user_idx, ratings_df, n=10):
        """
        CASE 2: Warm user (1–4 ratings).

        ALGORITHM:
        1. Get user's rated product_idxs and their ratings
        2. Map each rated product → category
        3. Build category preference vector:
           category_score[C] = mean rating user gave products in C
        4. Score unseen products: weighted sum of category alignment
        5. Blend 60% category preference + 40% popularity
           to avoid recommending obscure products

        Args:
            user_idx: Integer user index
            ratings_df: Full ratings DataFrame
            n: Number of recommendations

        Returns:
            pd.DataFrame with recommendation rows
        """
        user_ratings = ratings_df[ratings_df["user_idx"] == user_idx]
        rated_idxs = set(user_ratings["product_idx"].tolist())

        if user_ratings.empty:
            return self.recommend_popular(n=n, exclude_idxs=rated_idxs)

        # Build category preference from user history
        category_scores = {}
        score_lookup = self.popularity_scores["bayesian_score"].to_dict()

        for _, row in user_ratings.iterrows():
            pidx = row["product_idx"]
            asin = self.idx_to_product.get(pidx, "")
            product_row = self.products_df[
                self.products_df["product_id"] == asin
            ] if not self.products_df.empty else pd.DataFrame()

            category = (
                product_row.iloc[0]["category"]
                if not product_row.empty else "Electronics"
            )

            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(row["rating"])

        # Average per category
        category_pref = {
            cat: np.mean(ratings)
            for cat, ratings in category_scores.items()
        }

        # Normalize
        total = sum(category_pref.values())
        if total > 0:
            category_pref = {c: s / total for c, s in category_pref.items()}

        # Score all unseen products
        all_product_idxs = set(ratings_df["product_idx"].unique())
        candidates = all_product_idxs - rated_idxs

        results = []
        for pidx in candidates:
            asin = self.idx_to_product.get(pidx, "")
            product_row = self.products_df[
                self.products_df["product_id"] == asin
            ] if not self.products_df.empty else pd.DataFrame()

            category = (
                product_row.iloc[0]["category"]
                if not product_row.empty else "Electronics"
            )

            cat_score = category_pref.get(category, 0) * 5
            pop_score = score_lookup.get(pidx, self.global_mean)
            final_score = 0.6 * cat_score + 0.4 * pop_score

            results.append({
                "product_idx": pidx,
                "predicted_rating": round(float(
                    np.clip(final_score, 1.0, 5.0)
                ), 3),
                "title": (
                    product_row.iloc[0]["title"]
                    if not product_row.empty else ""
                ),
                "category": category,
                "price": (
                    product_row.iloc[0].get("price")
                    if not product_row.empty else None
                ),
                "reason": "Based on your category preferences"
            })

        if not results:
            return self.recommend_popular(n=n, exclude_idxs=rated_idxs)

        return (
            pd.DataFrame(results)
            .sort_values("predicted_rating", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def get_recommendation(self, user_idx, ratings_df, n=10):
        """
        Smart router — picks strategy based on rating count.

        0 ratings   → popularity
        1–4 ratings → category-weighted
        5+ ratings  → return None (caller uses SVD)
        """
        count = len(ratings_df[ratings_df["user_idx"] == user_idx])
        rated_idxs = set(
            ratings_df[ratings_df["user_idx"] == user_idx]["product_idx"]
        )

        if count == 0:
            print(f"   ❄️  user_idx={user_idx}: new user → popularity fallback")
            return self.recommend_popular(n=n, exclude_idxs=rated_idxs)
        elif count < 5:
            print(f"   🌡️  user_idx={user_idx}: {count} ratings → category fallback")
            return self.recommend_by_category(user_idx, ratings_df, n=n)
        else:
            return None