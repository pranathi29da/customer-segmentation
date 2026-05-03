import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from mlxtend.frequent_patterns import apriori, association_rules

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("ecommerce customer segmentation_unsupervised.csv")

# CLEAN
df.drop_duplicates(inplace=True)
df = df.ffill().bfill()
df.fillna(0, inplace=True)

# SAMPLE (LIGHTWEIGHT)
df = df.sample(n=400, random_state=42)

# ======================
# FEATURE ENGINEERING
# ======================
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
df['TransactionDate'] = df['TransactionDate'].ffill().bfill()

df['DaysSinceTransaction'] = (pd.Timestamp.now() - df['TransactionDate']).dt.days
df.drop('TransactionDate', axis=1, inplace=True)

df['Rating'] = df['Rating'].astype(float).round(1)

# ======================
# ENCODING
# ======================
le = LabelEncoder()
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ======================
# SELECT FEATURES
# ======================
features = [
    "Price","Quantity","DiscountApplied",
    "Rating","SessionDuration",
    "Device","Browser","ShippingType"
]

X_df = df[features]

# ======================
# SCALING
# ======================
scaler = StandardScaler()
X = scaler.fit_transform(X_df)

# ======================
# MODELS
# ======================
models = {
    "KMeans": KMeans(n_clusters=3, random_state=42, n_init=10),
    "Hierarchical": AgglomerativeClustering(n_clusters=3),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "GMM": GaussianMixture(n_components=3, random_state=42)
}

scores = {}
labels_dict = {}

for name, model in models.items():

    if name == "GMM":
        labels = model.fit_predict(X)
    else:
        labels = model.fit_predict(X)

    labels_dict[name] = labels

    if len(set(labels)) > 1 and len(set(labels)) < len(X):
        try:
            score = silhouette_score(X, labels)
            scores[name] = score
        except:
            scores[name] = -1
    else:
        scores[name] = -1

# ======================
# BEST MODEL
# ======================
best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]
best_labels = labels_dict[best_model_name]

print("Scores:", scores)
print("Best Model:", best_model_name)

# Final fit
if best_model_name == "GMM":
    best_model.fit(X)
else:
    best_model.fit(X)

df["Segment"] = best_labels

# ======================
# ASSOCIATION RULES
# ======================
basket = pd.get_dummies(df[[
    "Device","Browser","ShippingType","ReturnStatus"
]].astype(str))

freq_items = apriori(basket, min_support=0.1, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.3)

rules = rules.sort_values(by="confidence", ascending=False)

# ======================
# SAVE
# ======================
pickle.dump((best_model, scaler, features, rules), open("model.pkl", "wb"))

print("✅ Training Completed Successfully!")