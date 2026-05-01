# backend/src/data_loader.py

import os
import json
import gzip
import requests
import pandas as pd
import numpy as np
from io import BytesIO

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Amazon Electronics subset — ratings only (no review text, keeps it lean)
RATINGS_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv"

# Product metadata (title, category, price, description)
META_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz"

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


# ─────────────────────────────────────────────
# STEP 1: DOWNLOAD DATA
# ─────────────────────────────────────────────

def download_amazon_electronics(force=False):
    """
    Downloads Amazon Electronics ratings + product metadata.

    WHY this dataset?
    - 1,689,188 ratings from 192,403 users on 63,001 products
    - Rating scale: 1–5 stars (identical to MovieLens)
    - Real Amazon purchase/review data — this is the actual domain
    - Standard academic benchmark — interviewers recognize it

    Dataset source: UCSD Amazon Product Data (He & McAuley, 2016)
    Same group whose datasets Amazon researchers use for benchmarking.

    Files:
        ratings_Electronics.csv  — userId, productId, rating, timestamp
        meta_Electronics.json.gz — product title, price, category
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    ratings_path = os.path.join(RAW_DATA_DIR, "ratings_Electronics.csv")
    meta_path = os.path.join(RAW_DATA_DIR, "meta_Electronics.json.gz")

    if os.path.exists(ratings_path) and not force:
        print("✅ Ratings already downloaded. Skipping.")
    else:
        print("⬇️  Downloading Amazon Electronics ratings (~75MB)...")
        r = requests.get(RATINGS_URL, stream=True)
        r.raise_for_status()
        with open(ratings_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Ratings saved → {ratings_path}")

    if os.path.exists(meta_path) and not force:
        print("✅ Metadata already downloaded. Skipping.")
    else:
        print("⬇️  Downloading product metadata (~300MB compressed)...")
        r = requests.get(META_URL, stream=True)
        r.raise_for_status()
        with open(meta_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Metadata saved → {meta_path}")

    return ratings_path, meta_path


# ─────────────────────────────────────────────
# STEP 2: LOAD & FILTER RATINGS
# ─────────────────────────────────────────────

def load_ratings(data_dir=None, min_user_ratings=10, min_product_ratings=10,
                 sample_users=20000):
    """
    Loads and filters the Amazon Electronics ratings.

    WHY filter?
    Raw dataset: 1.7M ratings, 192K users, 63K products.
    Most users have rated only 1–2 products — useless for CF.
    Most products have only 1–2 ratings — can't learn from them.

    Filtering to "active" users and "popular" products:
    - Reduces noise dramatically
    - Keeps the meaningful collaborative signal
    - Reduces training time while preserving model quality
    - This is standard practice in all production rec systems

    WHY sample_users?
    Training SVD on 192K users × 63K items takes significant RAM
    and time on a laptop. 20K active users gives you the same
    RMSE signal at 1/10th the compute — fine for a portfolio project.
    At interview: "I subsampled to 20K active users for compute
    efficiency; the architecture scales to the full dataset."

    Args:
        min_user_ratings: Drop users with fewer ratings than this
        min_product_ratings: Drop products with fewer ratings than this
        sample_users: Cap on number of users to keep (None = keep all)

    Returns:
        pd.DataFrame with columns: user_id, product_id, rating, timestamp
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR

    filepath = os.path.join(data_dir, "ratings_Electronics.csv")
    print("📂 Loading Amazon Electronics ratings...")

    df = pd.read_csv(
        filepath,
        names=["user_id", "product_id", "rating", "timestamp"],
        dtype={"user_id": str, "product_id": str,
               "rating": float, "timestamp": int}
    )

    print(f"   Raw: {len(df):,} ratings | "
          f"{df['user_id'].nunique():,} users | "
          f"{df['product_id'].nunique():,} products")

    # ── Filter active users ──
    user_counts = df["user_id"].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    df = df[df["user_id"].isin(active_users)]

    # ── Filter popular products ──
    product_counts = df["product_id"].value_counts()
    popular_products = product_counts[
        product_counts >= min_product_ratings
    ].index
    df = df[df["product_id"].isin(popular_products)]

    # ── Sample users for compute efficiency ──
    if sample_users and df["user_id"].nunique() > sample_users:
        sampled_users = (
            df["user_id"].value_counts()
            .head(sample_users)  # most active users
            .index
        )
        df = df[df["user_id"].isin(sampled_users)]

    # ── Re-index user and product IDs to integers ──
    # WHY: Matrix operations need integer indices, not Amazon ASINs
    # We keep the original ID mapping to reverse-lookup later
    unique_users = sorted(df["user_id"].unique())
    unique_products = sorted(df["product_id"].unique())

    user_map = {uid: idx + 1 for idx, uid in enumerate(unique_users)}
    product_map = {pid: idx + 1 for idx, pid in enumerate(unique_products)}

    df["user_idx"] = df["user_id"].map(user_map)
    df["product_idx"] = df["product_id"].map(product_map)

    # Store mappings as DataFrame attributes for later use
    df.attrs["user_map"] = user_map
    df.attrs["product_map"] = product_map
    df.attrs["idx_to_user"] = {v: k for k, v in user_map.items()}
    df.attrs["idx_to_product"] = {v: k for k, v in product_map.items()}

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    print(f"✅ Filtered: {len(df):,} ratings | "
          f"{df['user_id'].nunique():,} users | "
          f"{df['product_id'].nunique():,} products")

    return df


# ─────────────────────────────────────────────
# STEP 3: LOAD PRODUCT METADATA
# ─────────────────────────────────────────────

def load_products(data_dir=None, product_ids=None):
    """
    Loads product metadata from the compressed JSON file.

    Each line in meta_Electronics.json.gz is a JSON object:
    {
        "asin": "B000068O48",           ← Amazon product ID
        "title": "Samsung 50-inch TV",
        "price": 499.99,
        "categories": [["Electronics", "TVs"]],
        "imUrl": "https://..."
    }

    WHY process lazily?
    The full metadata file has 498K products. Parsing all of them
    takes ~2 mins. We filter to only products in our ratings dataset.

    Args:
        product_ids: Set of ASINs to keep (from ratings dataset)
                     If None, loads all (slow).

    Returns:
        pd.DataFrame with product metadata
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR

    filepath = os.path.join(data_dir, "meta_Electronics.json.gz")

    if not os.path.exists(filepath):
        print("⚠️  Product metadata not found. Products will have ASIN-only titles.")
        return pd.DataFrame(columns=["product_id", "title", "price", "category"])

    print("📂 Loading product metadata (this takes ~60s)...")
    records = []
    loaded = 0

    with gzip.open(filepath, "rb") as f:
        for line in f:
            try:
                item = json.loads(line.decode("utf-8", errors="replace"))
                asin = item.get("asin", "")

                if product_ids and asin not in product_ids:
                    continue

                # Parse category: [["Electronics", "Televisions", "LED"]]
                # → take the most specific (last in first list)
                cats = item.get("categories", [[]])
                category = cats[0][-1] if cats and cats[0] else "Electronics"

                # Clean price: "$499.99" → 499.99
                price_raw = item.get("price", "")
                try:
                    price = float(str(price_raw).replace("$", "").replace(",", ""))
                except (ValueError, TypeError):
                    price = None

                records.append({
                    "product_id": asin,
                    "title": item.get("title", f"Product {asin}"),
                    "price": price,
                    "category": category,
                    "image_url": item.get("imUrl", "")
                })

                loaded += 1
                if loaded % 10000 == 0:
                    print(f"   Parsed {loaded:,} products...")

            except (json.JSONDecodeError, KeyError):
                continue

    df = pd.DataFrame(records)
    print(f"✅ Products loaded: {len(df):,} with metadata")
    return df


# ─────────────────────────────────────────────
# STEP 4: BUILD USER-ITEM MATRIX
# ─────────────────────────────────────────────

def build_user_item_matrix(ratings_df):
    """
    Builds the User-Item matrix for collaborative filtering.

    Shape: (n_users) × (n_products)
    Value: rating (1–5), 0 if not rated

    For CartIQ this will be roughly:
    ~20,000 users × ~15,000 products = 300M cells
    But sparsity ~99.9% — so in practice ~300K non-zero entries.

    WHY use user_idx/product_idx (integers) not raw IDs (ASINs)?
    Matrix operations require integer row/column indices.
    The idx ↔ original_id mapping in ratings_df.attrs lets us
    translate back to ASINs when returning results to the API.

    Returns:
        pd.DataFrame (user-item matrix, 0-filled)
    """
    print("🔧 Building User-Item matrix...")

    matrix = ratings_df.pivot_table(
        index="user_idx",
        columns="product_idx",
        values="rating"
    ).fillna(0)

    sparsity = 1 - (
        len(ratings_df) / (matrix.shape[0] * matrix.shape[1])
    )

    print(f"✅ Matrix: {matrix.shape[0]:,} users × {matrix.shape[1]:,} products")
    print(f"   Sparsity: {sparsity:.1%} of cells are unrated")
    return matrix


# ─────────────────────────────────────────────
# STEP 5: TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

def split_data(ratings_df, test_size=0.2, random_state=42):
    """
    Splits at the rating level — identical logic to MovieLens version.
    Stratify by rating to preserve score distribution.
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        ratings_df,
        test_size=test_size,
        random_state=random_state,
        stratify=ratings_df["rating"].astype(int)
    )

    print(f"✅ Split → Train: {len(train_df):,} | Test: {len(test_df):,}")
    return train_df, test_df


# ─────────────────────────────────────────────
# STEP 6: DATA QUALITY CHECKS
# ─────────────────────────────────────────────

def validate_data(ratings_df, products_df):
    """
    Sanity checks before model training.
    Catches issues early — silent data bugs are the hardest to debug.
    """
    issues = []

    invalid = ratings_df[~ratings_df["rating"].between(1, 5)]
    if not invalid.empty:
        issues.append(f"⚠️  {len(invalid)} ratings outside 1–5 range")

    dupes = ratings_df.duplicated(subset=["user_id", "product_id"])
    if dupes.any():
        issues.append(f"⚠️  {dupes.sum()} duplicate user-product pairs")

    if not products_df.empty:
        rated_ids = set(ratings_df["product_id"].unique())
        known_ids = set(products_df["product_id"].unique())
        missing = rated_ids - known_ids
        if missing:
            issues.append(
                f"⚠️  {len(missing)} rated products missing from metadata "
                f"(will show ASIN as title)"
            )

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ All data validation checks passed")

    return len(issues) == 0


# ─────────────────────────────────────────────
# CONVENIENCE: load everything at once
# ─────────────────────────────────────────────

def load_all():
    """
    Single call that returns everything Day 2+ modules need.

    Returns:
        ratings_df, products_df, user_item_matrix, train_df, test_df
    """
    download_amazon_electronics()
    ratings_df = load_ratings()

    # Load metadata only for products in our filtered ratings
    product_ids = set(ratings_df["product_id"].unique())
    products_df = load_products(product_ids=product_ids)

    validate_data(ratings_df, products_df)

    user_item_matrix = build_user_item_matrix(ratings_df)
    train_df, test_df = split_data(ratings_df)

    return ratings_df, products_df, user_item_matrix, train_df, test_df


if __name__ == "__main__":
    ratings, products, matrix, train, test = load_all()

    print("\n── Sample Ratings ──")
    print(ratings[["user_id", "product_id", "rating", "timestamp"]].head())

    print("\n── Sample Products ──")
    print(products[["product_id", "title", "category", "price"]].head())

    print(f"\n── Matrix: {matrix.shape[0]} users × {matrix.shape[1]} products ──")