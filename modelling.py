import os
import pandas as pd
import mlflow
import mlflow.sklearn
# HAPUS: import dagshub (Sudah terhapus)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Memulai skrip training CI/MLProject...")

# --- 1. HAPUS SEMUA BLOK INISIALISASI DAGSHUB ---
# Kredensial dan URI diambil otomatis dari Environment Variables (GitHub Secrets)
print(f"MLFLOW_TRACKING_URI diatur ke: {os.environ.get('MLFLOW_TRACKING_URI')}")

mlflow.set_experiment("CI - Automated Training")

# --- 2. Muat Data ---
DATA_PATH = 'data_bersih.csv' 
df = pd.read_csv(DATA_PATH)

X = df.drop('stroke', axis=1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Mulai MLflow Run ---
with mlflow.start_run(run_name="CI_RF_Run") as run:
    
    mlflow.sklearn.autolog()

    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        class_weight='balanced',
        random_state=42
    )
    
    print("Melatih model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Training CI Selesai. Akurasi: {acc:.4f}")
    
    # --- 4. Simpan Run ID ---
    run_id = run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    print(f"Run ID disimpan: {run_id}")

print("Skrip CI selesai.")
