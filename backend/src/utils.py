# backend/src/utils.py

import pandas as pd
import numpy as np


def get_products_rated_by_user(user_id, ratings_df):
    """Returns all products a user has rated, sorted by rating desc."""
    return (
        ratings_df[ratings_df["user_id"] == user_id]
        .sort_values("rating", ascending=False)
    )


def get_unrated_products(user_id, ratings_df, products_df):
    """
    Returns products the user has NOT yet rated.
    These are the candidates for recommendation.
    """
    rated_ids = set(ratings_df[ratings_df["user_id"] == user_id]["product_id"])
    all_ids = set(products_df["product_id"])
    unrated_ids = all_ids - rated_ids
    return products_df[products_df["product_id"].isin(unrated_ids)]


def enrich_recommendations(rec_df, products_df):
    """
    Joins recommendation scores with product metadata.
    Every recommender returns (product_idx, predicted_rating).
    This adds title, category, price for display.

    Args:
        rec_df: DataFrame with [product_idx, predicted_rating]
        products_df: Full product metadata DataFrame

    Returns:
        Enriched DataFrame sorted by predicted_rating desc
    """
    # rec_df uses product_idx (integer). products_df uses product_id (ASIN).
    # We need to join on product_idx which is stored in ratings_df.attrs.
    # For direct use, callers should pass enriched df via model_store._enrich()
    return rec_df.sort_values("predicted_rating", ascending=False)


def print_recommendations(enriched_df, user_id, method="k-NN"):
    """Pretty-prints recommendations to terminal."""
    print(f"\n{'═'*65}")
    print(f"  🛒 Top Recommendations for User {user_id}  [{method}]")
    print(f"{'═'*65}")
    for i, row in enriched_df.iterrows():
        title = row.get("title", f"Product {row['product_idx']}")
        category = row.get("category", "Electronics")
        price = row.get("price", None)
        price_str = f"${price:.2f}" if price else "N/A"

        print(f"  {i+1}. {title[:55]}")
        print(f"     Category: {category}")
        print(f"     Price: {price_str}   "
              f"Predicted Rating: {'⭐' * round(row['predicted_rating'])} "
              f"({row['predicted_rating']:.2f})")
        print()