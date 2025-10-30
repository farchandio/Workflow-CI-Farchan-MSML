import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Memulai skrip training CI/MLProject...")

# --- 1. Inisialisasi DagsHub ---
# Kredensial akan diambil dari GitHub Secrets (diatur di file workflow)
try:
    dagshub.init(repo_owner='farchandio',
                 repo_name='Eksperimen_SML_FarchanV2',
                 mlflow=True)
    print("Inisialisasi DagsHub berhasil.")
except Exception as e:
    print(f"Error inisialisasi DagsHub: {e}")
    # Jika gagal (misal: berjalan lokal tanpa setup), set URI secara manual
    if "MLFLOW_TRACKING_URI" not in os.environ:
         mlflow.set_tracking_uri(f"https://dagshub.com/farchandio/Eksperimen_SML_FarchanV2.mlflow")

mlflow.set_experiment("CI - Automated Training")

# --- 2. Muat Data ---
# Path ini relatif terhadap root repositori CI
DATA_PATH = 'namadataset_preprocessing/data_bersih.csv'
df = pd.read_csv(DATA_PATH)

X = df.drop('stroke', axis=1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Mulai MLflow Run ---
with mlflow.start_run(run_name="CI_RF_Run") as run:
    
    # Gunakan autolog untuk kesederhanaan di CI
    mlflow.sklearn.autolog()

    # Model sederhana (bukan tuning)
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        class_weight='balanced', # Tetap gunakan ini untuk data imbalanced
        random_state=42
    )
    
    print("Melatih model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Training CI Selesai. Akurasi: {acc:.4f}")
    
    # --- 4. Simpan Run ID ---
    # Ini penting agar langkah "build-docker" tahu model mana yang harus diambil
    run_id = run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    print(f"Run ID disimpan: {run_id}")

print("Skrip CI selesai.")
