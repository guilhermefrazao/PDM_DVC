# train.py — Exemplo com DVC e dataset de imagens (CIFAR-10)

import os
import joblib
from dvclive import Live
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10


os.makedirs("data", exist_ok=True)


(X_train_full, y_train_full), (X_test_full, y_test_full) = cifar10.load_data()

X_train_flat = X_train_full.reshape(len(X_train_full), -1)
X_test_flat = X_test_full.reshape(len(X_test_full), -1)
y_train_flat = y_train_full.ravel()
y_test_flat = y_test_full.ravel()

X_train, _, y_train, _ = train_test_split(
    X_train_flat, y_train_flat, test_size=0.8, random_state=42
)

with Live() as live:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1  # usa todos os núcleos da CPU
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test_flat)
    accuracy = accuracy_score(y_test_flat, y_pred)

    print(f"Acurácia do modelo: {accuracy:.2f}")

    # Registrar métrica
    live.log_metric("accuracy", accuracy)

    # Salvar o modelo
    model_path = "data/model_cifar10.joblib"
    joblib.dump(model, model_path)

    # Logar o modelo como artefato DVC
    live.log_artifact(model_path, type="model")
