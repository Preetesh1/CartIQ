# backend/src/recommender_knn.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")


class UserBasedKNN:
    """
    User-Based Collaborative Filtering for CartIQ.

    ALGORITHM (identical to movie version, domain changed):
    1. Build user-item matrix (rows=users, cols=products)
    2. Compute cosine similarity between all user pairs
    3. For target user + candidate product:
       a. Find K most similar users who rated that product
       b. Predict rating = similarity-weighted average of their ratings
    4. Rank all unrated products by predicted rating

    WHY cosine similarity for shopping?
    Two shoppers who both gave 5-star ratings to laptops and 1-star
    to cables are similar — even if one buys $300 items and the other
    buys $30 items. Cosine ignores magnitude (spending level),
    captures direction (taste alignment). Exactly right for this domain.

    COMPLEXITY:
    Training: O(U²) — fine for 20K users (~400M ops, <5s with numpy)
    Prediction: O(K × P) per user where P = products
    """

    def __init__(self, k=20, min_common_ratings=3):
        """
        Args:
            k (int): Number of nearest neighbors.
                     20 is well-tested; higher k = smoother but slower.
            min_common_ratings (int): Minimum shared-product threshold.
                     Two users need this many products in common to be
                     considered neighbors. Prevents spurious similarity.
        """
        self.k = k
        self.min_common_ratings = min_common_ratings
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_indices = None      # integer user_idx list
        self.product_indices = None   # integer product_idx list
        self.is_fitted = False

    def fit(self, ratings_df):
        """
        Builds user-item matrix and computes user-user cosine similarity.

        Uses user_idx / product_idx (integers) not raw IDs.
        This is because Amazon ASINs are strings — pivot tables
        on strings are slow; integer indices are fast.
        """
        print("🔧 Building User-Item matrix (CartIQ)...")

        matrix = ratings_df.pivot_table(
            index="user_idx",
            columns="product_idx",
            values="rating"
        ).fillna(0)

        self.user_indices = matrix.index.tolist()
        self.product_indices = matrix.columns.tolist()
        self.user_item_matrix = matrix

        print(f"   Matrix: {matrix.shape[0]:,} users × {matrix.shape[1]:,} products")
        print("🔧 Computing user-user cosine similarity...")

        self.similarity_matrix = cosine_similarity(matrix.values)
        np.fill_diagonal(self.similarity_matrix, 0)

        self.is_fitted = True
        print(f"✅ UserBasedKNN fitted.")
        return self

    def _get_similar_users(self, user_idx):
        """Returns K most similar users to user_idx, as (idx, score) tuples."""
        if user_idx not in self.user_indices:
            raise ValueError(f"user_idx {user_idx} not in training data.")

        u_pos = self.user_indices.index(user_idx)
        similarities = self.similarity_matrix[u_pos]
        top_k = np.argsort(similarities)[::-1][:self.k]

        return [
            (self.user_indices[i], similarities[i])
            for i in top_k if similarities[i] > 0
        ]

    def predict_rating(self, user_idx, product_idx):
        """
        Predicts the rating user_idx would give product_idx.

        Formula (weighted average):
                 Σ sim(u, v) × rating(v, product)
        pred =  ──────────────────────────────────
                          Σ |sim(u, v)|

        Returns float in [1, 5] or None if unpredictable.
        """
        similar_users = self._get_similar_users(user_idx)

        numerator = 0
        denominator = 0
        neighbors_used = 0

        for neighbor_idx, sim_score in similar_users:
            neighbor_rating = (
                self.user_item_matrix.loc[neighbor_idx, product_idx]
                if product_idx in self.product_indices else 0
            )
            if neighbor_rating > 0:
                numerator += sim_score * neighbor_rating
                denominator += abs(sim_score)
                neighbors_used += 1

        if denominator == 0 or neighbors_used < self.min_common_ratings:
            return None

        return float(np.clip(numerator / denominator, 1.0, 5.0))

    def recommend(self, user_idx, n=10, ratings_df=None):
        """
        Top-N product recommendations for user_idx.

        Returns:
            pd.DataFrame with [product_idx, predicted_rating]
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before recommend()")

        if ratings_df is not None:
            rated_product_idxs = set(
                ratings_df[ratings_df["user_idx"] == user_idx]["product_idx"]
            )
        else:
            rated_product_idxs = set(
                self.product_indices[
                    self.user_item_matrix.loc[user_idx] > 0
                ]
            )

        candidates = [p for p in self.product_indices
                      if p not in rated_product_idxs]
        predictions = []

        for product_idx in candidates:
            pred = self.predict_rating(user_idx, product_idx)
            if pred is not None:
                predictions.append({
                    "product_idx": product_idx,
                    "predicted_rating": round(pred, 3)
                })

        if not predictions:
            return pd.DataFrame(columns=["product_idx", "predicted_rating"])

        return (
            pd.DataFrame(predictions)
            .sort_values("predicted_rating", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )


class ItemBasedKNN:
    """
    Item-Based Collaborative Filtering for CartIQ.

    WHY item-based is especially strong for e-commerce:
    Product similarity is extremely stable — a gaming mouse and a
    mechanical keyboard will always be bought together. User tastes
    shift (a student becomes a professional), but product
    co-purchase patterns don't. This is why Amazon's original
    recommendation engine was item-based CF.

    This is literally the ancestral architecture of what powers
    "Customers who bought this also bought..." on Amazon today.
    """

    def __init__(self, k=20):
        self.k = k
        self.item_similarity_matrix = None
        self.user_item_matrix = None
        self.user_indices = None
        self.product_indices = None
        self.product_idx_to_pos = {}
        self.is_fitted = False

    def fit(self, ratings_df):
        print("🔧 Building Item-User matrix (CartIQ)...")

        matrix = ratings_df.pivot_table(
            index="user_idx",
            columns="product_idx",
            values="rating"
        ).fillna(0)

        self.user_indices = matrix.index.tolist()
        self.product_indices = matrix.columns.tolist()
        self.product_idx_to_pos = {
            pid: pos for pos, pid in enumerate(self.product_indices)
        }
        self.user_item_matrix = matrix

        print(f"   Matrix: {matrix.shape}")
        print("🔧 Computing item-item cosine similarity...")

        # Transpose: rows = products, cols = users
        item_matrix = matrix.values.T
        self.item_similarity_matrix = cosine_similarity(item_matrix)
        np.fill_diagonal(self.item_similarity_matrix, 0)

        self.is_fitted = True
        print(f"✅ ItemBasedKNN fitted. "
              f"Item similarity matrix: {self.item_similarity_matrix.shape}")
        return self

    def predict_rating(self, user_idx, product_idx):
        """
        Predicts rating for user_idx on product_idx.

        Logic: find K products most similar to product_idx that this
        user HAS rated → weighted average of their ratings.

        "Given what this user bought and rated, how much would
        they like this new product?"
        """
        if product_idx not in self.product_idx_to_pos:
            return None

        pos = self.product_idx_to_pos[product_idx]
        similarities = self.item_similarity_matrix[pos]
        top_k = np.argsort(similarities)[::-1][:self.k]

        if user_idx not in self.user_indices:
            return None

        user_ratings = self.user_item_matrix.loc[user_idx]
        numerator = 0
        denominator = 0

        for idx in top_k:
            similar_product_idx = self.product_indices[idx]
            sim_score = similarities[idx]
            user_rating = user_ratings.get(similar_product_idx, 0)

            if user_rating > 0:
                numerator += sim_score * user_rating
                denominator += abs(sim_score)

        if denominator == 0:
            return None

        return float(np.clip(numerator / denominator, 1.0, 5.0))

    def recommend(self, user_idx, n=10, ratings_df=None):
        """Same interface as UserBasedKNN — Day 5 API calls either transparently."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before recommend()")

        if ratings_df is not None:
            rated_product_idxs = set(
                ratings_df[ratings_df["user_idx"] == user_idx]["product_idx"]
            )
        else:
            rated_product_idxs = set()

        candidates = [p for p in self.product_indices
                      if p not in rated_product_idxs]
        predictions = []

        for product_idx in candidates:
            pred = self.predict_rating(user_idx, product_idx)
            if pred is not None:
                predictions.append({
                    "product_idx": product_idx,
                    "predicted_rating": round(pred, 3)
                })

        if not predictions:
            return pd.DataFrame(columns=["product_idx", "predicted_rating"])

        return (
            pd.DataFrame(predictions)
            .sort_values("predicted_rating", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_knn(model, test_df, sample_size=500):
    """
    RMSE and MAE on held-out test ratings.

    NOTE: sample_size=500 (vs 1000 for movies) because CartIQ's
    k-NN is slower per prediction due to larger product space.
    SVD (Day 3) will be fully vectorized and much faster.

    Returns:
        dict with rmse, mae, coverage
    """
    sample = test_df.sample(min(sample_size, len(test_df)), random_state=42)
    actuals, preds = [], []
    unpredicted = 0

    print(f"📊 Evaluating on {len(sample)} test ratings...")

    for _, row in sample.iterrows():
        pred = model.predict_rating(
            int(row["user_idx"]), int(row["product_idx"])
        )
        if pred is not None:
            actuals.append(row["rating"])
            preds.append(pred)
        else:
            unpredicted += 1

    actuals = np.array(actuals)
    preds = np.array(preds)

    rmse = np.sqrt(np.mean((actuals - preds) ** 2))
    mae = np.mean(np.abs(actuals - preds))
    coverage = len(preds) / len(sample)

    print(f"\n{'─'*45}")
    print(f"  📈 [{model.__class__.__name__}]")
    print(f"{'─'*45}")
    print(f"  RMSE:        {rmse:.4f}  (target: < 1.1)")
    print(f"  MAE:         {mae:.4f}  (target: < 0.90)")
    print(f"  Coverage:    {coverage:.1%}")
    print(f"  Unpredicted: {unpredicted} ratings")
    print(f"{'─'*45}\n")

    return {"rmse": rmse, "mae": mae, "coverage": coverage}


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(__file__))

    from data_loader import load_all

    ratings, products, matrix, train, test = load_all()

    # Pick a user_idx that has enough ratings
    active_users = train.groupby("user_idx")["rating"].count()
    test_user_idx = active_users[active_users >= 20].index[0]
    print(f"\nUsing user_idx={test_user_idx} for demo")

    # ── User-Based ──
    print("\n" + "═"*65)
    print("  USER-BASED k-NN")
    print("═"*65)
    user_knn = UserBasedKNN(k=20, min_common_ratings=3)
    user_knn.fit(train)
    recs = user_knn.recommend(user_idx=test_user_idx, n=5, ratings_df=train)
    print(recs)
    user_eval = evaluate_knn(user_knn, test, sample_size=300)

    # ── Item-Based ──
    print("\n" + "═"*65)
    print("  ITEM-BASED k-NN")
    print("═"*65)
    item_knn = ItemBasedKNN(k=20)
    item_knn.fit(train)
    recs_item = item_knn.recommend(user_idx=test_user_idx, n=5, ratings_df=train)
    print(recs_item)
    item_eval = evaluate_knn(item_knn, test, sample_size=300)

    # ── Compare ──
    print(f"\n{'Model':<20} {'RMSE':>8} {'MAE':>8} {'Coverage':>10}")
    print("─" * 50)
    print(f"{'UserBasedKNN':<20} {user_eval['rmse']:>8.4f} "
          f"{user_eval['mae']:>8.4f} {user_eval['coverage']:>10.1%}")
    print(f"{'ItemBasedKNN':<20} {item_eval['rmse']:>8.4f} "
          f"{item_eval['mae']:>8.4f} {item_eval['coverage']:>10.1%}")