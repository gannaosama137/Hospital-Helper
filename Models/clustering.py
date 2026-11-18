from pathlib import Path
import pandas as pd
import numpy as np  
from preprocessing.cardio import clustering_preprocessing
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ====================Set project paths===========================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
file_path = DATA_DIR / "cardio_train.csv"
PLOT_DIR = ROOT / "Plots"
PLOT_DIR.mkdir(exist_ok=True)



# ====================Calculate NMI for selected features=========

def nmi_of(feats):
    Z = StandardScaler().fit_transform(df[feats])
    lab = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(Z)
    return normalized_mutual_info_score(df['cardio'], lab)



# =====================Load and preprocess data===================

df = pd.read_csv(file_path , sep=';')
df , cluster_features = clustering_preprocessing(df) 


# =====================Analyze feature importance using NMI=======

base = nmi_of(cluster_features)

rows = {}
for f in cluster_features:
    rest = [c for c in cluster_features if c != f]
    rows[f] = nmi_of(rest) - base

res = pd.Series(rows).sort_values()



# =====================Plot feature importance====================

res.plot.barh()
plt.axvline(0, color='k')
plt.xlabel("NMI change when feature is removed")
plt.tight_layout()
plt.savefig(PLOT_DIR / "feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()



# =====================Train the final KMeans model===============

features = ['age', 'bmi', 'cholesterol', 'gluc', 'ap_hi', 'ap_lo']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features]) 
km = KMeans(n_clusters=4, n_init=10, random_state=42)
df['cluster'] = km.fit_predict(scaled_features)



# =====================Visualize clusters=========================

pts = PCA(n_components=2).fit_transform(scaled_features)
sample = np.random.RandomState(0).choice(len(df), 5000, replace=False)
lab = df['cluster'].values[sample]

names = {
    0: "Young, normal values",
    1: "Older, near-normal",
    2: "High glucose + cholesterol",
    3: "High BP + high BMI",
}

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
a.scatter(pts[sample, 0], pts[sample, 1], s=5)
a.set_title("Unlabeled data")

colors = plt.get_cmap('tab10').colors

for c, name in names.items():
    m = lab == c
    rate = df.loc[df['cluster'] == c, 'cardio'].mean() * 100
    b.scatter(pts[sample][m, 0], pts[sample][m, 1], s=5, color=colors[c],
              label=f"{c}: {name} (cardio {rate:.0f}%)")
b.set_title("Clustered data")
b.legend(markerscale=3, fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_DIR / "clustered_data.png", dpi=300, bbox_inches='tight')
plt.show()



# =====================Evaluate and profile the clusters==========

print("NMI of final model:",
 round(normalized_mutual_info_score(df['cardio'], df['cluster']), 4))

extra = ['smoke', 'alco', 'gender', 'active', 'cardio']
prof = df.groupby('cluster')[features + extra].mean().round(2)
prof['size_%'] = (df['cluster'].value_counts(normalize=True).sort_index() * 100  ).round(1)
print(prof.T)



# =====================Save the trained model=====================

(ROOT / "Save_Models").mkdir(exist_ok=True)
joblib.dump({"scaler": scaler, "model": km, "features": features},
            ROOT / "Save_Models" / "cluster_model.pkl")
print("Model saved.")

