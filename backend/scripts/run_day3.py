import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_all
from recommender_knn import UserBasedKNN, ItemBasedKNN
from recommender_svd import SVDRecommender
from evaluator import generate_comparison_report

ratings, products, matrix, train, test = load_all()

# ── Train all three ──
user_knn = UserBasedKNN(k=20).fit(train)
item_knn = ItemBasedKNN(k=20).fit(train)
svd = SVDRecommender(n_factors=100, n_epochs=20).fit(train)

# ── Full report ──
generate_comparison_report(
    models_dict={
        "UserBased-KNN": user_knn,
        "ItemBased-KNN": item_knn,
        "SVD": svd
    },
    test_df=test,
    output_dir="backend/notebooks"
)

# ── Similar products (the "customers also bought" feature) ──
idx_to_product = ratings.attrs.get("idx_to_product", {})
# Pick the most-rated product as demo
top_product_idx = (
    train.groupby("product_idx")["rating"].count().idxmax()
)
print(f"\n── Products similar to product_idx={top_product_idx} ──")
similar = svd.get_similar_products(
    top_product_idx, n=5,
    products_df=products,
    idx_to_product=idx_to_product
)
if similar is not None:
    print(similar[["title", "category", "similarity"]].to_string(index=False))

# ── Save ──
os.makedirs("backend/models", exist_ok=True)
svd.save("backend/models/svd_model.pkl")