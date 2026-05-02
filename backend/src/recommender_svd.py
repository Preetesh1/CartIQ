# backend/src/recommender_svd.py

import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings("ignore")


class SVDRecommender:
    """
    Matrix Factorization Recommender (Funk SVD) for CartIQ.

    LATENT FACTORS learned from Amazon Electronics data:
    - Factor 1 might capture: "Gaming peripherals enthusiast"
    - Factor 2 might capture: "Apple ecosystem buyer"
    - Factor 3 might capture: "Budget-conscious shopper"
    - Factor 4 might capture: "Professional audio/video"

    No genre labels, no price information — purely inferred from
    who bought what and how they rated it.

    AMAZON-SPECIFIC NOTES:
    ─────────────────────────────────────────────────────────
    Positivity bias: global_mean here will be ~4.1 vs ~3.5 for
    MovieLens. This means item bias terms matter MORE here —
    a product with a 3.0 average is genuinely disliked, not
    just average. The bias terms capture this signal cleanly,
    leaving factor vectors to model relative preference.

    Purchase intent: A 5-star rating in shopping means "I bought
    this and loved it" — stronger signal than a movie watch.
    But it also means the matrix is even sparser (people buy
    fewer products than they watch movies), so SVD's ability
    to generalize from few observations is critical.
    ─────────────────────────────────────────────────────────

    IMPROVEMENT over k-NN:
    - Works in dense latent factor space (not sparse 15K-dim product space)
    - Handles sparsity gracefully — learns user intent from limited data
    - Vectorized inference: all product scores in one matrix multiply
    - Typical RMSE improvement: 1.05 → 0.92 on Amazon Electronics
    """

    def __init__(self, n_factors=100, n_epochs=20, lr=0.005,
                 reg=0.02, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state

        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None

        self.user_idx_to_pos = {}
        self.item_idx_to_pos = {}
        self.pos_to_user_idx = {}
        self.pos_to_item_idx = {}

        self.is_fitted = False

    def fit(self, ratings_df):
        """
        Trains SVD using Stochastic Gradient Descent.

        Uses user_idx / product_idx integers from the filtered
        ratings DataFrame — same integer space as the k-NN models.

        WHAT WE OPTIMIZE:
        Minimize: Σ (r_ui - r̂_ui)² + λ(‖pᵤ‖² + ‖qᵢ‖² + bᵤ² + bᵢ²)

        r̂_ui = μ + bᵤ + bᵢ + pᵤ · qᵢ

        μ  = global mean (~4.1 for Amazon Electronics)
        bᵤ = user bias (harsh reviewer vs generous reviewer)
        bᵢ = item bias (premium product vs cheap accessory)
        pᵤ = user latent vector (taste profile)
        qᵢ = item latent vector (product profile)
        """
        print(f"🔧 Training CartIQ SVD "
              f"(factors={self.n_factors}, epochs={self.n_epochs})")

        np.random.seed(self.random_state)

        user_indices = sorted(ratings_df["user_idx"].unique())
        item_indices = sorted(ratings_df["product_idx"].unique())

        self.user_idx_to_pos = {uid: pos for pos, uid in enumerate(user_indices)}
        self.item_idx_to_pos = {iid: pos for pos, iid in enumerate(item_indices)}
        self.pos_to_user_idx = {v: k for k, v in self.user_idx_to_pos.items()}
        self.pos_to_item_idx = {v: k for k, v in self.item_idx_to_pos.items()}

        n_users = len(user_indices)
        n_items = len(item_indices)

        self.global_mean = ratings_df["rating"].mean()
        print(f"   Global mean rating: {self.global_mean:.3f} "
              f"(Amazon positivity bias visible vs ~3.5 for movies)")

        self.user_factors = np.random.normal(0.1, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0.1, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)

        users_arr = ratings_df["user_idx"].map(self.user_idx_to_pos).values
        items_arr = ratings_df["product_idx"].map(self.item_idx_to_pos).values
        ratings_arr = ratings_df["rating"].values.astype(float)
        n_ratings = len(ratings_arr)

        print(f"   Training on {n_ratings:,} ratings...")

        for epoch in range(self.n_epochs):
            shuffle_idx = np.random.permutation(n_ratings)
            users_s = users_arr[shuffle_idx]
            items_s = items_arr[shuffle_idx]
            ratings_s = ratings_arr[shuffle_idx]
            epoch_loss = 0

            for u_pos, i_pos, r_ui in zip(users_s, items_s, ratings_s):
                pred = (self.global_mean
                        + self.user_biases[u_pos]
                        + self.item_biases[i_pos]
                        + np.dot(self.user_factors[u_pos],
                                 self.item_factors[i_pos]))

                err = r_ui - pred
                epoch_loss += err ** 2

                self.user_biases[u_pos] += self.lr * (
                    err - self.reg * self.user_biases[u_pos]
                )
                self.item_biases[i_pos] += self.lr * (
                    err - self.reg * self.item_biases[i_pos]
                )

                pu = self.user_factors[u_pos].copy()
                qi = self.item_factors[i_pos].copy()

                self.user_factors[u_pos] += self.lr * (err * qi - self.reg * pu)
                self.item_factors[i_pos] += self.lr * (err * pu - self.reg * qi)

            if (epoch + 1) % 5 == 0:
                rmse = np.sqrt(epoch_loss / n_ratings)
                print(f"   Epoch {epoch+1:>3}/{self.n_epochs} | "
                      f"Train RMSE: {rmse:.4f}")

        self.is_fitted = True
        print("✅ SVD training complete")
        return self

    def predict_rating(self, user_idx, product_idx):
        """
        Predicts rating for user_idx on product_idx.

        Formula: μ + bᵤ + bᵢ + pᵤ · qᵢ

        Returns float in [1, 5] or None if user/product unknown.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")

        if user_idx not in self.user_idx_to_pos:
            return None
        if product_idx not in self.item_idx_to_pos:
            return None

        u_pos = self.user_idx_to_pos[user_idx]
        i_pos = self.item_idx_to_pos[product_idx]

        pred = (self.global_mean
                + self.user_biases[u_pos]
                + self.item_biases[i_pos]
                + np.dot(self.user_factors[u_pos], self.item_factors[i_pos]))

        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user_idx, n=10, ratings_df=None):
        """
        Top-N recommendations via vectorized matrix multiply.

        One dot product for ALL products simultaneously —
        ~100x faster than looping predict_rating().
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")

        if user_idx not in self.user_idx_to_pos:
            return pd.DataFrame(columns=["product_idx", "predicted_rating"])

        u_pos = self.user_idx_to_pos[user_idx]

        rated_product_idxs = set()
        if ratings_df is not None:
            rated_product_idxs = set(
                ratings_df[ratings_df["user_idx"] == user_idx]["product_idx"]
            )

        # Vectorized: score all products at once
        all_preds = (
            self.global_mean
            + self.user_biases[u_pos]
            + self.item_biases
            + self.item_factors @ self.user_factors[u_pos]
        )
        all_preds = np.clip(all_preds, 1.0, 5.0)

        results = []
        for i_pos, pred in enumerate(all_preds):
            product_idx = self.pos_to_item_idx[i_pos]
            if product_idx not in rated_product_idxs:
                results.append({
                    "product_idx": product_idx,
                    "predicted_rating": round(float(pred), 3)
                })

        return (
            pd.DataFrame(results)
            .sort_values("predicted_rating", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def get_similar_products(self, product_idx, n=10, products_df=None,
                             idx_to_product=None):
        """
        Finds products most similar to product_idx via item factor vectors.

        This is the "Customers also bought" feature.
        Similarity is learned purely from co-purchase patterns —
        no category labels, no price data used.

        A USB-C hub and a laptop stand having high similarity
        purely from rating patterns proves the model learned
        real purchasing behavior.
        """
        if product_idx not in self.item_idx_to_pos:
            return None

        i_pos = self.item_idx_to_pos[product_idx]
        target_vec = self.item_factors[i_pos]

        norms = np.linalg.norm(self.item_factors, axis=1)
        target_norm = np.linalg.norm(target_vec)
        sims = (self.item_factors @ target_vec) / (norms * target_norm + 1e-10)
        sims[i_pos] = -1  # exclude self

        top_indices = np.argsort(sims)[::-1][:n]
        results = []

        for pos in top_indices:
            pidx = self.pos_to_item_idx[pos]
            entry = {
                "product_idx": pidx,
                "similarity": round(float(sims[pos]), 4)
            }

            # Resolve ASIN if mapping provided
            if idx_to_product:
                asin = idx_to_product.get(pidx, str(pidx))
                entry["product_id"] = asin

                if products_df is not None:
                    if products_df.empty or "product_id" not in products_df.columns:
                        row = pd.DataFrame()
                    else:
                        row = products_df[products_df["product_id"] == asin]

                    if not row.empty:
                        entry["title"] = row.iloc[0]["title"]
                        entry["category"] = row.iloc[0]["category"]
                        entry["price"] = row.iloc[0].get("price")

            results.append(entry)  # ← moved out of if idx_to_product block

        return pd.DataFrame(results)