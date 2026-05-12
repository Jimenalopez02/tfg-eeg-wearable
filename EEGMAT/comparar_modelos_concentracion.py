from pathlib import Path
import numpy as np

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = Path(".")
in_path = DATA_DIR / "dataset_concentracion_de.npz"

if not in_path.exists():
    raise FileNotFoundError(f"No encuentro {in_path}")

data = np.load(in_path, allow_pickle=True)
X = data["X_feat"]      # (n_windows, 10)
y = data["y"]           # (n_windows,)
groups = data["groups"] # sujeto por ventana

print("X shape:", X.shape)
print("y shape:", y.shape)
print("groups únicos:", np.unique(groups))

logo = LeaveOneGroupOut()

models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=500,
            random_state=42
        ))
    ]),
}

results = []

for name, model in models.items():
    print(f"\n===== Evaluando {name} =====")

    accs = []
    f1s = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        accs.append(acc)
        f1s.append(f1)

        test_subject = np.unique(groups[test_idx])
        print(f"Fold {fold} | sujeto test {test_subject} | ACC={acc:.4f} | F1={f1:.4f}")

    results.append({
        "model": name,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
    })

results_sorted = sorted(results, key=lambda r: (-r["f1_mean"], -r["acc_mean"]))

print("\n===== RANKING FINAL CONCENTRACIÓN =====")
for i, r in enumerate(results_sorted, start=1):
    print(
        f"{i}. {r['model']}: "
        f"ACC={r['acc_mean']:.4f} ± {r['acc_std']:.4f}, "
        f"F1={r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
    )