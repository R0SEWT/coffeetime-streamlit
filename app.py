from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL = Path(__file__).parent / "model.joblib"

st.set_page_config(page_title="CoffeeTime Café", page_icon="☕")
st.title("☕ CoffeeTime Café")
st.subheader("Predicción de retorno de clientes")
st.write("Estima si un cliente regresará durante los próximos 30 días.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL)


model = load_model()

# Las opciones del formulario salen del propio modelo: son exactamente las
# categorías que vio en clientes_historicos.csv durante el entrenamiento.
encoder = model.named_steps["prep"].named_transformers_["cat"]
categorias = {col: list(cats) for col, cats in zip(encoder.feature_names_in_, encoder.categories_)}


def opcion_inicial(opciones, preferida):
    """Muestra primero la categoría más frecuente del histórico."""
    return opciones.index(preferida) if preferida in opciones else 0


with st.form("customer_form"):
    c1, c2 = st.columns(2)
    with c1:
        edad = st.number_input("Edad", min_value=18, max_value=100, value=35)
        visitas = st.number_input("Visitas últimos 90 días", min_value=0, value=5)
        gasto = st.number_input("Gasto promedio (S/)", min_value=0.0, value=22.0, step=0.5)
        dias = st.number_input("Días desde última visita", min_value=0, value=15)
        descuentos = st.number_input("Descuentos utilizados", min_value=0, value=1)
    with c2:
        calificacion = st.number_input(
            "Calificación promedio", min_value=1.0, max_value=5.0, value=4.1, step=0.1
        )
        frecuencia = st.number_input("Frecuencia de compras", min_value=0.0, value=1.6, step=0.1)
        canales = categorias["canal_compra"]
        productos = categorias["producto_favorito"]
        canal = st.selectbox(
            "Canal de compra", canales, index=opcion_inicial(canales, "Presencial")
        )
        producto = st.selectbox(
            "Producto favorito", productos, index=opcion_inicial(productos, "Café")
        )
        fidelidad = st.selectbox("¿Usa programa de fidelidad?", ["Sí", "No"])
    submitted = st.form_submit_button("Predecir retorno")

if submitted:
    # En el CSV usa_programa_fidelidad está codificada como 1 = sí, 0 = no.
    usa_fidelidad = 1 if fidelidad == "Sí" else 0
    cliente = pd.DataFrame(
        [
            {
                "edad": edad,
                "visitas_ultimos_90_dias": visitas,
                "gasto_promedio": gasto,
                "dias_desde_ultima_visita": dias,
                "canal_compra": canal,
                "producto_favorito": producto,
                "usa_programa_fidelidad": usa_fidelidad,
                "descuentos_utilizados": descuentos,
                "calificacion_promedio": calificacion,
                "frecuencia_compras": frecuencia,
            }
        ]
    )
    pred = int(model.predict(cliente)[0])
    prob = float(model.predict_proba(cliente)[0][1])
    st.divider()
    st.metric("Probabilidad estimada de retorno", f"{prob:.1%}")
    if pred == 1:
        st.success("El modelo estima que el cliente REGRESARÁ.")
    else:
        st.warning("El modelo estima que el cliente NO REGRESARÁ.")
    st.caption("Es una estimación del modelo, no una garantía.")

with st.expander("Sobre el modelo"):
    st.markdown(
        "- Random Forest entrenado con 1 500 clientes históricos (datos sintéticos del curso).\n"
        "- En el conjunto de prueba: ROC-AUC 0.65 y balanced accuracy 0.62. "
        "Detecta 56 % de los clientes que no regresan.\n"
        "- La variable que más pesa es **días desde la última visita**, seguida de las "
        "**visitas en los últimos 90 días**.\n"
        "- Úsalo para priorizar acciones de fidelización, no como decisión automática."
    )
