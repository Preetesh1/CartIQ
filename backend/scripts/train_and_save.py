# backend/scripts/train_and_save.py

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_all
from recommender_knn import UserBasedKNN, ItemBasedKNN
from recommender_svd import SVDRecommender
from evaluator import generate_comparison_report
import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("═" * 60)
print("  🏋️  TRAINING CARTIQ MODELS")
print("═" * 60)

ratings, products, matrix, train, test = load_all()

print("\n[1/3] Training SVD...")
svd = SVDRecommender(n_factors=100, n_epochs=20, lr=0.005, reg=0.02)
svd.fit(train)
joblib.dump(svd, os.path.join(MODELS_DIR, "svd_model.pkl"))

print("\n[2/3] Training ItemKNN...")
item_knn = ItemBasedKNN(k=20)
item_knn.fit(train)
joblib.dump(item_knn, os.path.join(MODELS_DIR, "item_knn.pkl"))

print("\n[3/3] Training UserKNN...")
user_knn = UserBasedKNN(k=20)
user_knn.fit(train)
joblib.dump(user_knn, os.path.join(MODELS_DIR, "user_knn.pkl"))

generate_comparison_report(
    models_dict={
        "UserBased-KNN": user_knn,
        "ItemBased-KNN": item_knn,
        "SVD": svd
    },
    test_df=test,
    output_dir=os.path.join(os.path.dirname(__file__), "..", "notebooks")
)

print("\n✅ All CartIQ models saved.")
print(f"   Files: {os.listdir(MODELS_DIR)}")