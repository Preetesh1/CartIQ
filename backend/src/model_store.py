# backend/src/model_store.py

import os
import joblib
import pandas as pd
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")


class ModelStore:
    """
    Central registry for all CartIQ models and data.

    This is the ONLY object the Flask API imports.
    One call to store.load() at startup — everything is ready.

    RESPONSIBILITIES:
    ─────────────────────────────────────────────────────────
    1. Load pre-trained models from disk at startup (~1–2s)
    2. Hold ratings + products DataFrames in memory
    3. Hold ID mappings (product_idx ↔ ASIN ↔ metadata)
    4. Route requests to the right model
    5. Handle cold start via ColdStartHandler
    6. Provide unified .recommend() regardless of model used
    7. Serialize all responses to JSON-safe dicts
    ─────────────────────────────────────────────────────────

    SINGLETON PATTERN:
    store = ModelStore() at module level.
    Python's import caching means this object is shared across
    all Flask worker threads — models are loaded exactly once.
    """

    SVD_MIN_RATINGS = 5

    def __init__(self):
        self.svd_model = None
        self.user_knn = None
        self.item_knn = None
        self.cold_start = None

        self.ratings_df = None
        self.products_df = None

        # ID mappings — critical for translating between
        # integer indices (used by ML models) and ASINs (used by API)
        self.idx_to_product = {}   # product_idx → ASIN
        self.product_to_idx = {}   # ASIN → product_idx
        self.idx_to_user = {}      # user_idx → original user_id string
        self.user_to_idx = {}      # original user_id → user_idx

        self.is_loaded = False
        self._load_time = None
        self._known_user_idxs = None

    # ─────────────────────────────────────────────
    # LOADING
    # ─────────────────────────────────────────────

    def load(self, models_dir=None, data_dir=None):
        """
        Loads all models and data. Called ONCE when Flask app starts.

        Load order matters:
        1. Data first (ratings, products, ID mappings)
        2. Models (need data loaded to verify compatibility)
        3. Cold start handler (needs both data and models)
        """
        models_dir = models_dir or MODELS_DIR
        data_dir = data_dir or DATA_DIR
        start = time.time()

        print("\n🔄 Loading CartIQ ModelStore...")

        self._load_data(data_dir)
        self._load_models(models_dir)
        self._init_cold_start()
        self._cache_known_users()

        self.is_loaded = True
        self._load_time = round(time.time() - start, 2)
        print(f"✅ ModelStore ready in {self._load_time}s\n")
        return self

    def _load_data(self, data_dir):
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, "src"))
        from data_loader import load_ratings, load_products

        print("   📂 Loading data...")
        self.ratings_df = load_ratings(data_dir=data_dir)

        # Extract ID mappings from DataFrame attrs
        # These were set during load_ratings() in data_loader.py
        self.idx_to_product = self.ratings_df.attrs.get("idx_to_product", {})
        self.product_to_idx = self.ratings_df.attrs.get("product_map", {})
        self.idx_to_user = self.ratings_df.attrs.get("idx_to_user", {})
        self.user_to_idx = self.ratings_df.attrs.get("user_map", {})

        product_ids = set(self.ratings_df["product_id"].unique())
        self.products_df = load_products(product_ids=product_ids)

        print(f"   Ratings: {len(self.ratings_df):,} | "
              f"Users: {self.ratings_df['user_idx'].nunique():,} | "
              f"Products: {self.ratings_df['product_idx'].nunique():,}")

    def _load_models(self, models_dir):
        print("   📂 Loading models...")

        svd_path = os.path.join(models_dir, "svd_model.pkl")
        if os.path.exists(svd_path):
            self.svd_model = joblib.load(svd_path)
            size_kb = os.path.getsize(svd_path) / 1024
            print(f"   ✅ SVD loaded ({size_kb:.0f} KB)")
        else:
            print(f"   ⚠️  SVD not found → run: python scripts/train_and_save.py")

        for name, attr, filename in [
            ("ItemKNN", "item_knn", "item_knn.pkl"),
            ("UserKNN", "user_knn", "user_knn.pkl"),
        ]:
            path = os.path.join(models_dir, filename)
            if os.path.exists(path):
                setattr(self, attr, joblib.load(path))
                print(f"   ✅ {name} loaded")

    def _init_cold_start(self):
        from cold_start import ColdStartHandler
        self.cold_start = ColdStartHandler(min_votes=25)
        self.cold_start.fit(self.ratings_df, self.products_df)

    def _cache_known_users(self):
        """Cache the set of known user_idx values for O(1) lookup."""
        self._known_user_idxs = set(self.ratings_df["user_idx"].unique())

    # ─────────────────────────────────────────────
    # CORE: RECOMMEND
    # ─────────────────────────────────────────────

    def recommend(self, user_id, n=10, model="svd"):
        """
        Main recommendation method. Flask API calls only this.

        Accepts EITHER:
        - user_id as original string ID (from API request)
        - user_id as integer user_idx (internal use)

        Routing:
        ┌─────────────────────────────────────────────┐
        │  Resolve user_id → user_idx                 │
        │  Known user + ≥5 ratings?                   │
        │    YES → SVD / k-NN based on `model` param  │
        │    NO  → Cold start handler                  │
        └─────────────────────────────────────────────┘

        Args:
            user_id: Original user ID string or integer user_idx
            n: Number of recommendations
            model: "svd" | "user_knn" | "item_knn" | "popular"

        Returns:
            dict (JSON-serializable)
        """
        self._require_loaded()

        user_idx = self._resolve_user_idx(user_id)
        rating_count = self._get_rating_count(user_idx)
        is_known = user_idx in self._known_user_idxs

        if not is_known or rating_count < self.SVD_MIN_RATINGS:
            return self._cold_start_response(user_idx, n)

        try:
            if model == "svd" and self.svd_model:
                recs_df = self.svd_model.recommend(
                    user_idx, n=n, ratings_df=self.ratings_df
                )
                model_used = "SVD"

            elif model == "item_knn" and self.item_knn:
                recs_df = self.item_knn.recommend(
                    user_idx, n=n, ratings_df=self.ratings_df
                )
                model_used = "ItemKNN"

            elif model == "user_knn" and self.user_knn:
                recs_df = self.user_knn.recommend(
                    user_idx, n=n, ratings_df=self.ratings_df
                )
                model_used = "UserKNN"

            elif model == "popular":
                return self._popularity_response(user_idx, n)

            else:
                print(f"   ⚠️  Model '{model}' unavailable, falling back to SVD")
                recs_df = self.svd_model.recommend(
                    user_idx, n=n, ratings_df=self.ratings_df
                )
                model_used = "SVD (fallback)"

            enriched = self._enrich(recs_df)
            return self._format_response(
                user_idx, enriched, model_used, fallback=False
            )

        except Exception as e:
            print(f"   ❌ Recommend error for user_idx={user_idx}: {e}")
            return self._cold_start_response(user_idx, n)

    def get_similar_products(self, product_id, n=10):
        """
        "Customers also bought" — similar products by item factor vectors.

        Args:
            product_id: ASIN string (from API) or product_idx integer
            n: Number of similar products

        Returns:
            dict with source product + similar products list
        """
        self._require_loaded()

        if not self.svd_model:
            return {"error": "SVD model not loaded"}

        product_idx = self._resolve_product_idx(product_id)
        if product_idx is None:
            return {"error": f"Product '{product_id}' not found"}

        similar_df = self.svd_model.get_similar_products(
            product_idx,
            n=n,
            products_df=self.products_df,
            idx_to_product=self.idx_to_product
        )

        if similar_df is None or similar_df.empty:
            return {"error": "Could not compute similar products"}

        source_asin = self.idx_to_product.get(product_idx, str(product_idx))
        source_row = self.products_df[
            self.products_df["product_id"] == source_asin
        ]
        source_title = (
            source_row.iloc[0]["title"]
            if not source_row.empty else source_asin
        )

        return {
            "source_product": {
                "product_id": source_asin,
                "product_idx": int(product_idx),
                "title": source_title
            },
            "similar_products": self._serialize_similar(similar_df)
        }

    def get_user_history(self, user_id, n=10):
        """
        Returns a user's highest-rated products.
        Powers the "Your ratings" panel in the React dashboard.
        """
        self._require_loaded()

        user_idx = self._resolve_user_idx(user_id)
        user_ratings = (
            self.ratings_df[self.ratings_df["user_idx"] == user_idx]
            .sort_values("rating", ascending=False)
            .head(n)
        )

        if user_ratings.empty:
            return {"user_idx": int(user_idx), "history": [],
                    "total_ratings": 0}

        rows = []
        for _, row in user_ratings.iterrows():
            asin = self.idx_to_product.get(int(row["product_idx"]), "")
            product_row = self.products_df[
                self.products_df["product_id"] == asin
            ] if not self.products_df.empty else pd.DataFrame()

            rows.append({
                "product_id": asin,
                "product_idx": int(row["product_idx"]),
                "title": (
                    product_row.iloc[0]["title"]
                    if not product_row.empty else asin
                ),
                "category": (
                    product_row.iloc[0].get("category", "Electronics")
                    if not product_row.empty else "Electronics"
                ),
                "price": (
                    float(product_row.iloc[0]["price"])
                    if not product_row.empty
                    and pd.notna(product_row.iloc[0].get("price"))
                    else None
                ),
                "rating": float(row["rating"])
            })

        return {
            "user_idx": int(user_idx),
            "total_ratings": self._get_rating_count(user_idx),
            "history": rows
        }

    def get_model_stats(self):
        """
        Metadata about loaded models and dataset.
        Exposed via GET /api/stats — shown in dashboard header.
        """
        self._require_loaded()
        return {
            "models_loaded": {
                "svd": self.svd_model is not None,
                "item_knn": self.item_knn is not None,
                "user_knn": self.user_knn is not None,
            },
            "dataset": {
                "total_ratings": int(len(self.ratings_df)),
                "total_users": int(self.ratings_df["user_idx"].nunique()),
                "total_products": int(
                    self.ratings_df["product_idx"].nunique()
                ),
                "global_mean_rating": round(
                    float(self.ratings_df["rating"].mean()), 3
                )
            },
            "svd_config": {
                "n_factors": (
                    self.svd_model.n_factors if self.svd_model else None
                ),
                "n_epochs": (
                    self.svd_model.n_epochs if self.svd_model else None
                ),
            },
            "load_time_sec": self._load_time,
            "cold_start_threshold": self.SVD_MIN_RATINGS
        }

    def get_all_users(self, limit=100):
        """
        Returns a sample of user_idx values with rating counts.
        Powers the user picker dropdown in the React dashboard.
        """
        self._require_loaded()

        counts = (
            self.ratings_df
            .groupby("user_idx")["rating"]
            .count()
            .sort_values(ascending=False)
            .head(limit)
        )

        return [
            {"user_idx": int(idx), "rating_count": int(count)}
            for idx, count in counts.items()
        ]

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _require_loaded(self):
        if not self.is_loaded:
            raise RuntimeError("Call ModelStore.load() before using it.")

    def _resolve_user_idx(self, user_id):
        """Accepts string user_id or integer user_idx, returns user_idx."""
        if isinstance(user_id, int):
            return user_id
        mapped = self.user_to_idx.get(str(user_id))
        if mapped is not None:
            return mapped
        try:
            return int(user_id)
        except (ValueError, TypeError):
            return -1  # unknown user → cold start

    def _resolve_product_idx(self, product_id):
        """Accepts ASIN string or integer product_idx, returns product_idx."""
        if isinstance(product_id, int):
            return product_id
        mapped = self.product_to_idx.get(str(product_id))
        if mapped is not None:
            return mapped
        try:
            return int(product_id)
        except (ValueError, TypeError):
            return None

    def _get_rating_count(self, user_idx):
        return int(len(
            self.ratings_df[self.ratings_df["user_idx"] == user_idx]
        ))

    def _enrich(self, recs_df):
        """
        Joins product_idx recommendations with ASIN + metadata.
        Handles gracefully when products_df is empty.
        """
        if recs_df.empty:
            return recs_df

        rows = []
        for _, row in recs_df.iterrows():
            pidx = int(row["product_idx"])
            asin = self.idx_to_product.get(pidx, "")
            pred_rating = float(row["predicted_rating"])

            if not self.products_df.empty:
                product_row = self.products_df[
                    self.products_df["product_id"] == asin
                ]
            else:
                product_row = pd.DataFrame()

            rows.append({
                "product_idx": pidx,
                "product_id": asin,
                "predicted_rating": pred_rating,
                "title": (
                    product_row.iloc[0]["title"]
                    if not product_row.empty else asin
                ),
                "category": (
                    product_row.iloc[0].get("category", "Electronics")
                    if not product_row.empty else "Electronics"
                ),
                "price": (
                    float(product_row.iloc[0]["price"])
                    if not product_row.empty
                    and pd.notna(product_row.iloc[0].get("price"))
                    else None
                ),
                "image_url": (
                    product_row.iloc[0].get("image_url", "")
                    if not product_row.empty else ""
                ),
                "reason": row.get("reason", "Recommended for you")
            })

        return pd.DataFrame(rows).sort_values(
            "predicted_rating", ascending=False
        )

    def _format_response(self, user_idx, enriched_df, model_used, fallback):
        """Converts enriched DataFrame to JSON-serializable dict."""
        recs = []
        for _, row in enriched_df.iterrows():
            recs.append({
                "product_id": str(row.get("product_id", "")),
                "product_idx": int(row.get("product_idx", 0)),
                "title": str(row.get("title", "")),
                "category": str(row.get("category", "Electronics")),
                "price": (
                    float(row["price"])
                    if pd.notna(row.get("price")) else None
                ),
                "image_url": str(row.get("image_url", "")),
                "predicted_rating": float(row["predicted_rating"]),
                "reason": str(row.get("reason", "Recommended for you"))
            })

        return {
            "user_idx": int(user_idx),
            "model_used": model_used,
            "fallback": fallback,
            "total_ratings_by_user": self._get_rating_count(user_idx),
            "recommendations": recs
        }

    def _cold_start_response(self, user_idx, n):
        recs_df = self.cold_start.get_recommendation(
            user_idx, self.ratings_df, n=n
        )
        if recs_df is None or recs_df.empty:
            recs_df = self.cold_start.recommend_popular(n=n)

        rating_count = self._get_rating_count(user_idx)
        model_used = (
            "Popularity (new user)"
            if rating_count == 0
            else "Category-weighted (warm user)"
        )

        # cold_start already enriches with title/category
        enriched = self._enrich(recs_df) if "title" not in recs_df.columns \
            else recs_df

        return self._format_response(user_idx, enriched, model_used,
                                     fallback=True)

    def _popularity_response(self, user_idx, n):
        rated_idxs = set(
            self.ratings_df[
                self.ratings_df["user_idx"] == user_idx
            ]["product_idx"]
        )
        recs_df = self.cold_start.recommend_popular(
            n=n, exclude_idxs=rated_idxs
        )
        return self._format_response(
            user_idx, recs_df, "Popularity", fallback=False
        )

    def _serialize_similar(self, similar_df):
        results = []
        for _, row in similar_df.iterrows():
            results.append({
                "product_id": str(row.get("product_id", "")),
                "product_idx": int(row.get("product_idx", 0)),
                "title": str(row.get("title", "")),
                "category": str(row.get("category", "Electronics")),
                "price": (
                    float(row["price"])
                    if pd.notna(row.get("price")) else None
                ),
                "similarity": float(row.get("similarity", 0))
            })
        return results


# ── Singleton ──
store = ModelStore()