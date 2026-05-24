import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
import joblib
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('datasetcoba.csv')

print("=" * 50)
print("DATA AWAL INFO")
print("=" * 50)
print(f"Shape awal: {df.shape}")

# ============================================================
# 2. PREPROCESSING & FEATURE ENGINEERING
# ============================================================
df['pain'] = pd.to_numeric(df['pain'], errors='coerce')
df['pain'].fillna(df['pain'].median(), inplace=True)

base_features = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
df_clean = df.dropna(subset=base_features + ['acuity']).reset_index(drop=True)

df_clean['shock_index'] = df_clean['heartrate'] / df_clean['sbp']
df_clean['MAP'] = (df_clean['sbp'] + 2 * df_clean['dbp']) / 3

features_final = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'shock_index', 'MAP']

X = df_clean[features_final]
y = df_clean['acuity'].astype(int)

# --- SIMPAN MEDIAN UNTUK WEB ---
medians_dict = X.median().to_dict()

# ============================================================
# 3. SPLIT DATA & 4. TRAINING (DIKEMBALIKAN KE SETTINGAN AWAL)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None,          
    min_samples_split=2,     
    min_samples_leaf=1,
    class_weight='balanced', 
    random_state=42
)
best_rf.fit(X_train, y_train)

# ============================================================
# 5. PREDIKSI & EVALUASI (MUNCUL LAGI DI TERMINAL)
# ============================================================
y_pred = best_rf.predict(X_test)

print("\n" + "=" * 50)
print("HASIL EVALUASI MODEL OPTIMAL")
print("=" * 50)
print(f"Akurasi Model : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# ============================================================
# 6. EXPORT PAKET LENGKAP
# ============================================================
importances = best_rf.feature_importances_
importance_dict = dict(zip(features_final, importances))

export_package = {
    'model': best_rf,
    'medians': medians_dict,
    'importances': importance_dict
}

joblib.dump(export_package, 'model_triage_clinical.pkl')
print("\n[SUKSES] Paket Model berhasil disimpan sebagai: 'model_triage_clinical.pkl'")