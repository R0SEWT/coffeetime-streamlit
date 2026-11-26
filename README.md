#  CoffeeTime Café: predicción de retorno de clientes

Aplicación web en Streamlit que estima si un cliente de CoffeeTime Café regresará durante los
próximos 30 días, para orientar las acciones de fidelización. 

| Elemento | Definición |
|---|---|
| Problema | Predecir si un cliente regresará en los próximos 30 días |
| Tipo | Clasificación binaria |
| Variable objetivo | `regresa_30_dias` (1 = sí, 0 = no) |
| Modelo | Random Forest Classifier (scikit-learn) |
| Salida | Clase estimada y probabilidad de retorno |

## Dataset

`clientes_historicos.csv`: 1 500 clientes sintéticos del curso, sin valores faltantes ni
duplicados. El 64.4 % regresa y el 35.6 % no.

| Tipo | Columnas |
|---|---|
| Numéricas | `edad`, `visitas_ultimos_90_dias`, `gasto_promedio`, `dias_desde_ultima_visita`, `descuentos_utilizados`, `calificacion_promedio`, `frecuencia_compras` |
| Binaria (0/1) | `usa_programa_fidelidad` |
| Categóricas | `canal_compra` (Presencial, Delivery, Aplicación), `producto_favorito` (Café, Postres, Bebidas frías, Otros) |
| Excluida | `cliente_id` (identificador, no describe al cliente) |

Solo se entrena con el histórico. `clientes_actuales.csv` y las predicciones anteriores no se usan.

## Modelo y decisiones

- **Un solo pipeline** (`ColumnTransformer` + `RandomForestClassifier`) guardado en `model.joblib`.
  La app le pasa los datos crudos y el pipeline aplica el mismo One-Hot Encoding del entrenamiento.
- **Partición estratificada 80/20** y validación cruzada de 5 folds sobre el 80 % de entrenamiento.
- **`class_weight="balanced"`**: con 64/36, un modelo sin ajuste tiende a decir "regresa" a casi
  todos. Balancear las clases mejora la detección de quienes no regresan, que son el público de
  las campañas de fidelización.
- **Árboles poco profundos** (`max_depth=8`, `min_samples_leaf=3`): la señal del dataset es débil
  y un bosque sin límite se sobreajusta.
- **Modelo final**: tras evaluar, se reentrena con los 1 500 registros antes de guardarlo.

## Resultados (conjunto de prueba, 300 clientes)

| Métrica | Línea base (siempre "regresa") | Random Forest |
|---|---|---|
| Accuracy | 0.643 | 0.630 |
| Balanced accuracy | 0.500 | 0.615 |
| Precision (regresa) | 0.643 | 0.733 |
| Recall (regresa) | 1.000 | 0.668 |
| F1 (regresa) | 0.783 | 0.699 |
| Recall (no regresa) | 0.000 | 0.561 |
| ROC-AUC | 0.500 | 0.654 |

Validación cruzada: ROC-AUC 0.650 ± 0.021 y balanced accuracy 0.607 ± 0.022.

|  | Pred: no regresa | Pred: regresa |
|---|---|---|
| **Real: no regresa** | 60 | 47 |
| **Real: regresa** | 64 | 129 |

**Lectura para el negocio.**

- La accuracy y el F1 de la clase "regresa" no sirven para comparar aquí. La línea base saca
  F1 = 0.78 sin aprender nada, porque acierta con todos los que regresan y nunca detecta a quien
  se va.
- El modelo sí separa los dos grupos: ROC-AUC 0.65 frente a 0.50. Encuentra al 56 % de los
  clientes que no regresan, y cuando dice "regresa" acierta el 73 % de las veces, frente al 64 %
  de la línea base.
- El error más costoso es el falso positivo: predecir que regresa a alguien que se va, porque
  ese cliente queda fuera de la campaña. En la prueba hubo 47 casos así.
- El falso negativo cuesta menos: es un beneficio entregado a alguien que igual iba a volver.
  Hubo 64 casos.
- El poder predictivo es moderado. Conviene usar la probabilidad para ordenar a quién contactar
  primero, no como una decisión automática.
- La variable más influyente es `dias_desde_ultima_visita`, seguida de
  `visitas_ultimos_90_dias` y `descuentos_utilizados`. La importancia se midió por permutación
  sobre la caída de ROC-AUC. Indica asociación, no causalidad.

## Cómo ejecutarlo

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt

python train_model.py             # entrena, imprime métricas y genera model.joblib
streamlit run app.py              # abre http://localhost:8501
```

Las versiones de `requirements.txt` están fijadas a las del entrenamiento (Python 3.12) para que
`model.joblib` cargue sin errores. Si se cambia de versión de scikit-learn hay que reentrenar.

## Despliegue

Publicado en Streamlit Community Cloud desde este repositorio: rama `main`, archivo principal
`app.py`, Python 3.12.

## Estructura

```text
coffeetime-streamlit/
├── clientes_historicos.csv   # datos sintéticos de entrenamiento
├── train_model.py            # entrenamiento y evaluación
├── app.py                    # aplicación Streamlit
├── model.joblib              # pipeline entrenado (preprocesamiento + Random Forest)
├── requirements.txt          # dependencias con versiones fijas
└── README.md
```
