"""Entrena el clasificador de retorno a 30 días de CoffeeTime Café y lo guarda en model.joblib.

Uso:
    python train_model.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).parent
DATA = BASE / "clientes_historicos.csv"
MODEL = BASE / "model.joblib"

ID = "cliente_id"
TARGET = "regresa_30_dias"
CATEGORICAS = ["canal_compra", "producto_favorito"]
NUMERICAS = [
    "edad",
    "visitas_ultimos_90_dias",
    "gasto_promedio",
    "dias_desde_ultima_visita",
    "usa_programa_fidelidad",  # ya viene codificada como 0/1
    "descuentos_utilizados",
    "calificacion_promedio",
    "frecuencia_compras",
]
SEED = 42


def crear_pipeline() -> Pipeline:
    """Preprocesamiento + modelo en un solo objeto, para que la app reciba los datos crudos."""
    preprocesamiento = ColumnTransformer(
        [
            # Random Forest no necesita escalar las variables numéricas.
            ("num", "passthrough", NUMERICAS),
            # Una categoría nueva no rompe la predicción: se codifica como ceros.
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ]
    )
    modelo = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,  # árboles poco profundos: la señal es débil y se sobreajusta fácil
        min_samples_leaf=3,
        class_weight="balanced",  # 64 % regresa / 36 % no: compensa la clase minoritaria
        random_state=SEED,
        n_jobs=-1,
    )
    return Pipeline([("prep", preprocesamiento), ("model", modelo)])


def metricas(y_real, y_pred, y_prob) -> dict:
    return {
        "Accuracy": accuracy_score(y_real, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_real, y_pred),
        "Precision (regresa)": precision_score(y_real, y_pred, zero_division=0),
        "Recall (regresa)": recall_score(y_real, y_pred, zero_division=0),
        "F1 (regresa)": f1_score(y_real, y_pred, zero_division=0),
        "Recall (no regresa)": recall_score(y_real, y_pred, pos_label=0, zero_division=0),
        "ROC-AUC": roc_auc_score(y_real, y_prob),
    }


# 1. Datos -------------------------------------------------------------------
# El CSV trae BOM al inicio; con utf-8-sig la primera columna se lee como "cliente_id".
df = pd.read_csv(DATA, encoding="utf-8-sig")

assert not df[ID].duplicated().any(), "Hay clientes duplicados"
assert df[NUMERICAS + CATEGORICAS + [TARGET]].notna().all().all(), "Hay valores faltantes"

print(f"Registros: {len(df)}")
print("Distribución de la variable objetivo:")
print(df[TARGET].value_counts(normalize=True).round(3).to_string(), "\n")

# cliente_id es un identificador, no una característica del cliente: se excluye.
X = df[NUMERICAS + CATEGORICAS]
y = df[TARGET]

# 2. Partición estratificada (mantiene la proporción 64/36 en train y test) --
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# 3. Validación cruzada sobre train ------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_res = cross_validate(
    crear_pipeline(),
    X_train,
    y_train,
    cv=cv,
    scoring=["balanced_accuracy", "roc_auc", "f1"],
)
print("Validación cruzada (5 folds, train):")
for m in ["balanced_accuracy", "roc_auc", "f1"]:
    s = cv_res[f"test_{m}"]
    print(f"  {m:18s} {s.mean():.3f} ± {s.std():.3f}")

# 4. Evaluación en test frente a una línea base ------------------------------
pipeline = crear_pipeline().fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Línea base: decir siempre "regresa" (la clase mayoritaria).
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

tabla = pd.DataFrame(
    {
        "Línea base": metricas(y_test, base.predict(X_test), base.predict_proba(X_test)[:, 1]),
        "Random Forest": metricas(y_test, y_pred, y_prob),
    }
).round(3)
print("\nEvaluación en test (20 %):")
print(tabla.to_string())

print("\nMatriz de confusión (filas = real, columnas = predicho):")
print(
    pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=[0, 1]),
        index=["Real: no regresa", "Real: regresa"],
        columns=["Pred: no regresa", "Pred: regresa"],
    ).to_string()
)
print("\n" + classification_report(y_test, y_pred, target_names=["no regresa", "regresa"]))

# 5. Qué variables usa el modelo (importancia por permutación en test) --------
imp = permutation_importance(
    pipeline, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=SEED
)
importancias = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
print("Importancia por permutación (caída de ROC-AUC al desordenar la variable):")
print(importancias.round(3).to_string())

# 6. Modelo final: se reentrena con todos los datos históricos y se guarda ----
modelo_final = crear_pipeline().fit(X, y)
joblib.dump(modelo_final, MODEL, compress=3)
print(f"\nModelo guardado en {MODEL.name} ({MODEL.stat().st_size / 1024:.0f} KB)")
