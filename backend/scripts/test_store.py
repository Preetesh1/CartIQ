import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model_store import store

store.load()

# Test 1: Normal user
print("\n── Test 1: SVD recommendation ──")
active_users = store.get_all_users(limit=5)
test_user_idx = active_users[0]["user_idx"]
result = store.recommend(user_id=test_user_idx, n=5)
print(f"Model: {result['model_used']} | Fallback: {result['fallback']}")
for r in result["recommendations"]:
    print(f"  {r['title'][:50]} → {r['predicted_rating']:.2f}★")

# Test 2: New user cold start
print("\n── Test 2: New user (idx=999999) ──")
result2 = store.recommend(user_id=999999, n=3)
print(f"Model: {result2['model_used']}")
for r in result2["recommendations"]:
    print(f"  {r['title'][:50]} → {r['predicted_rating']:.2f}★")

# Test 3: Similar products
print("\n── Test 3: Similar products ──")
sample_idx = store.ratings_df.groupby("product_idx")["rating"].count().idxmax()
similar = store.get_similar_products(product_id=sample_idx, n=4)
if "error" not in similar:
    print(f"Source: {similar['source_product']['title'][:50]}")
    for p in similar["similar_products"]:
        print(f"  → {p.get('title', p['product_id'])[:45]} "
              f"(sim={p['similarity']:.3f})")

# Test 4: User history
print("\n── Test 4: User history ──")
history = store.get_user_history(user_id=test_user_idx, n=4)
print(f"Total ratings: {history['total_ratings']}")
for h in history["history"]:
    print(f"  {h.get('title','?')[:45]} → {h['rating']}★")

# Test 5: Model stats
print("\n── Test 5: Model stats ──")
print(json.dumps(store.get_model_stats(), indent=2))