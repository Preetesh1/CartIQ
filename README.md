![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red?style=for-the-badge)

<p align="center">
  <img src="./assets/cartiq.png" alt="CartIQ Banner" width="450"/>
</p>

# 🛒 CartIQ – AI-Powered Shopping Recommendation Engine

**CartIQ** is a full-stack recommendation system that predicts what products a user is likely to love — before they search for them. Built on Amazon Electronics data with collaborative filtering and SVD matrix factorization, it mirrors the core recommendation architecture used at companies like Amazon, Flipkart, and Netflix.

> *"Find what users want, before they know they want it."*

---

## 🚀 What CartIQ Can Do

- 🧠 **SVD Matrix Factorization** — learns hidden taste profiles from purchase patterns
- 👥 **User-Based & Item-Based k-NN** — collaborative filtering baselines with cosine similarity
- ❄️ **Cold Start Handling** — intelligent fallbacks for new users (popularity + category-weighted)
- 🛍️ **"Customers Also Bought"** — similar product discovery via learned item factor vectors
- ⚡ **Sub-50ms Inference** — vectorized matrix multiply, models hot-loaded at startup
- 📊 **Full Evaluation Report** — RMSE, MAE, and coverage across all three models
- 🌐 **Deployable REST API** — Flask backend with clean JSON responses
- 💻 **React Dashboard** — product cards, user selector, real-time recommendations

---

## 📊 Model Performance (Amazon Electronics — 1.7M ratings)

| Model | RMSE | MAE | Coverage |
|-------|------|-----|----------|
| User-Based k-NN | ~1.05 | ~0.88 | ~91% |
| Item-Based k-NN | ~1.01 | ~0.84 | ~89% |
| **SVD (production)** | **~0.92** | **~0.74** | **~99.8%** |

> SVD's near-100% coverage comes from learning dense latent vectors for every user and product during training — no "not enough neighbors" failures.

---

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| ML Core | NumPy, scikit-learn | SVD, k-NN, cosine similarity |
| Data Pipeline | pandas | Dataset loading, filtering, matrix building |
| Model Persistence | joblib | Serialization of trained models |
| Backend API | Flask | REST endpoints for recommendations |
| Frontend | React | Product recommendation dashboard |
| Deployment | Render + Vercel | Backend + frontend hosting |

---

## 🧠 How It Works

```
Amazon Electronics Ratings
        ↓
  Data Pipeline (data_loader.py)
  Filter active users ≥10 ratings
  Filter popular products ≥10 ratings
  Build User-Item Matrix (20K × 15K)
        ↓
  ┌─────────────────────────────────┐
  │     SVD Matrix Factorization     │
  │                                  │
  │  R ≈ U × Σ × Vᵀ                 │
  │                                  │
  │  User vectors  → taste profiles  │
  │  Item vectors  → product profiles│
  │  Bias terms    → rating shifts   │
  └─────────────────────────────────┘
        ↓
  r̂_ui = μ + bᵤ + bᵢ + pᵤ · qᵢ
        ↓
  Flask REST API → React Dashboard
```

**Latent factors learned (no labels given):**
- Factor 1 → "Gaming peripherals buyer"
- Factor 2 → "Apple ecosystem user"
- Factor 3 → "Budget shopper"
- Factor 4 → "Professional audio/video"

---

## 📁 Project Structure

```
cartiq/
│
├── backend/
│   ├── data/
│   │   └── raw/                    ← Amazon Electronics dataset
│   ├── src/
│   │   ├── data_loader.py          ← Download, filter, build matrix
│   │   ├── recommender_knn.py      ← User-based & item-based k-NN
│   │   ├── recommender_svd.py      ← SVD matrix factorization (SGD)
│   │   ├── evaluator.py            ← RMSE/MAE comparison report
│   │   ├── cold_start.py           ← Popularity + category fallbacks
│   │   ├── model_store.py          ← Singleton model registry
│   │   └── utils.py                ← Shared helpers
│   ├── api/
│   │   └── app.py                  ← Flask REST API
│   ├── models/                     ← Serialized .pkl files
│   ├── notebooks/
│   │   ├── eda.ipynb               ← Exploratory data analysis
│   │   └── model_comparison.png    ← Evaluation chart
│   └── scripts/
│       ├── train_and_save.py       ← Train all models, save to disk
│       └── test_store.py           ← Smoke tests for ModelStore
│
├── frontend/                       ← React dashboard
│   └── src/
│       ├── components/
│       │   ├── ProductCard.jsx
│       │   ├── UserSelector.jsx
│       │   └── SimilarProducts.jsx
│       └── App.jsx
│
└── README.md
```

---

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Preetesh1/cartiq.git
cd cartiq
```

### 2. Set up the Python environment

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r backend/requirements.txt
```

### 3. Train and save all models

```bash
# Downloads dataset, trains SVD + k-NN, saves .pkl files (~3 mins)
python backend/scripts/train_and_save.py
```

### 4. Start the Flask API

```bash
cd backend
python api/app.py
# Server runs on http://localhost:5000
```

### 5. Start the React frontend

```bash
cd frontend
npm install
npm run dev
# Dashboard runs on http://localhost:5173
```

---

## 📡 API Endpoints

### Recommendations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/recommend?user_id=42&n=10` | Top-N recommendations for a user |
| GET | `/api/recommend?user_id=42&model=item_knn` | Use a specific model |
| GET | `/api/similar?product_id=B001E4KFG0&n=5` | "Customers also bought" |
| GET | `/api/history?user_id=42` | User's top-rated products |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stats` | Model metadata and dataset info |
| GET | `/api/health` | Health check |

### Example Response

```json
{
  "user_id": 42,
  "model_used": "SVD",
  "fallback": false,
  "total_ratings_by_user": 38,
  "recommendations": [
    {
      "product_id": "B001E4KFG0",
      "title": "Logitech MX Master 3 Wireless Mouse",
      "category": "Mice",
      "price": 99.99,
      "predicted_rating": 4.83,
      "reason": "Recommended for you"
    }
  ]
}
```

---

## ❄️ Cold Start Handling

CartIQ handles three cold start scenarios without ever returning an error:

| Scenario | Trigger | Strategy |
|----------|---------|---------|
| New user | 0 ratings | Bayesian popularity ranking (IMDb formula) |
| Warm user | 1–4 ratings | Genre/category-weighted scoring |
| Known user | 5+ ratings | Full SVD inference |

**Why Bayesian average for popularity?**
A product with 1 five-star review shouldn't rank above one with 10,000 near-perfect reviews. The formula `(C × μ + Σr) / (C + n)` pulls low-count scores toward the global mean — the same method used by IMDb's Top 250.

---

## 🔑 Key Engineering Decisions

**Why Funk SVD over sklearn's TruncatedSVD?**
sklearn's version treats missing values as 0, biasing predictions. Funk's SGD-SVD updates only on observed ratings, making it correct for sparse recommendation data.

**Why filter to active users (≥10 ratings)?**
Users with 1–2 ratings provide almost no collaborative signal — they have no "neighborhood." Filtering them out reduces noise and improves RMSE significantly with negligible coverage loss.

**Why item-based k-NN for the fallback?**
Product co-purchase similarity is stable over time — a mechanical keyboard and a wrist rest will always be bought together. User similarity drifts as tastes change. Amazon's original recommendation engine was item-based CF for exactly this reason.

**Why singleton ModelStore?**
Python's module import caching ensures the ModelStore is instantiated once per process. All Flask request threads share the same hot-loaded models — no per-request disk reads, no retraining.

---

## 📈 EDA Insights

From `notebooks/eda.ipynb` on the Amazon Electronics dataset:

- **Positivity bias**: ~60% of all ratings are 4–5 stars (vs ~40% for MovieLens). Bias terms in SVD are critical to avoid over-predicting for all users.
- **Long tail distribution**: Even after filtering for ≥10 ratings, most users cluster at 10–15 ratings. Cold start is not an edge case — it's the majority of users.
- **Category concentration**: Cables, cases, and batteries account for the most ratings by volume. Headphones and laptops have the highest average ratings.
- **Sparsity**: ~99.7% of the user-product matrix is unrated — significantly sparser than MovieLens (93.7%). This is why SVD's latent space generalization matters more here.

---

## 🗓️ Build Timeline

| Day | Focus | Output |
|-----|-------|--------|
| 1 | Data pipeline + EDA | `data_loader.py`, `eda.ipynb` |
| 2 | k-NN collaborative filtering | `recommender_knn.py` |
| 3 | SVD matrix factorization | `recommender_svd.py`, evaluation report |
| 4 | Model persistence + cold start | `model_store.py`, `cold_start.py` |
| 5 | Flask REST API | `app.py`, all endpoints live |
| 6 | API hardening | Auth, rate limiting, CORS, error handling |
| 7 | React core | User selector, product cards |
| 8 | React polish | Search, filters, score visualization |
| 9 | Deployment | Render (backend) + Vercel (frontend) |
| 10 | Final polish | README, demo GIF, resume-ready GitHub |

---

## 🚀 Live Demo

> **[cartiq.vercel.app]()** ← Frontend  
> **[cartiq-api.onrender.com]()** ← API

*Cold start on Render's free tier may take 30s to spin up on first request.*

---

> 🚫 **Notice**
>
> This project is publicly visible for demonstration and portfolio purposes only.
> All rights reserved by the author. Unauthorized use, reproduction, or distribution is prohibited.
>
> 🛡️ Project by Preetesh Sharma | All Rights Reserved
