import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Memulai skrip training CI/MLProject...")
print(f"MLFLOW_TRACKING_URI diatur ke: {os.environ.get('MLFLOW_TRACKING_URI')}")

# HAPUS BARIS 'set_experiment' DARI SINI

# --- 2. Muat Data ---
DATA_PATH = 'data_bersih.csv' 
df = pd.read_csv(DATA_PATH)

X = df.drop('stroke', axis=1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Training Model (TANPA autolog) ---
params = {
    'n_estimators': 100,
    'max_depth': 10,
    'class_weight': 'balanced',
    'random_state': 42
}
model = RandomForestClassifier(**params)

print("Melatih model...")
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Training CI Selesai. Akurasi: {acc:.4f}")

# --- 4. Logging Manual ---
# Konteks run sekarang diatur oleh 'mlflow run --experiment-name'
print("Logging parameter dan metrik secara manual...")
mlflow.log_params(params)
mlflow.log_metric("accuracy", acc)
mlflow.sklearn.log_model(model, "model")

# --- 5. Simpan Run ID ---
run = mlflow.active_run()
if run:
    run_id = run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    print(f"Run ID disimpan: {run_id}")
else:
    print("Error: Tidak dapat menemukan active run!")
    exit(1) # Keluar dengan error jika run tidak ditemukan

print("Skrip CI selesai.")
