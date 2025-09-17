import streamlit as st
from transformers import pipeline
import pandas as pd

st.set_page_config(page_title="Clasificador Zero‑Shot", page_icon="🔎", layout="centered")

st.title("🔎 Clasificador de Tópicos (Zero‑Shot)")
st.write(
    "Ingresa un **texto** y una lista de **etiquetas** separadas por comas. "
    "El modelo evaluará la afinidad de cada etiqueta con el texto usando NLI (Zero‑Shot)."
)
st.caption("Modelo por defecto: `facebook/bart-large-mnli`. Para español, prueba también `joeddav/xlm-roberta-large-xnli`.")

@st.cache_resource(show_spinner=True)
def load_classifier(model_name: str):
    # Detecta GPU si está disponible; en caso contrario usa CPU
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except Exception:
        device = -1
    return pipeline(
        task="zero-shot-classification",
        model=model_name,
        device=device,
    )

with st.sidebar:
    st.header("⚙️ Opciones")
    model_name = st.selectbox(
        "Modelo",
        ["facebook/bart-large-mnli", "joeddav/xlm-roberta-large-xnli"],
        index=0,
        help=(
            "BART (inglés) es rápido y preciso en textos en inglés. "
            "XLM‑R funciona mejor de forma multilingüe (incl. español)."
        ),
    )
    multi_label = st.checkbox("Permitir múltiples etiquetas (multi‑label)", value=True)
    normalize = st.checkbox("Normalizar etiquetas (minúsculas / trim)", value=True)

classifier = load_classifier(model_name)

st.subheader("Entrada")
texto = st.text_area(
    "Texto a analizar",
    value="El equipo ganó el partido y ahora lidera la liga.",
    height=160,
    placeholder="Pega aquí el texto…",
)

etiquetas_raw = st.text_input(
    "Posibles categorías (separadas por comas)",
    value="deportes, política, economía",
    placeholder="ej.: deportes, política, economía",
)

def parse_labels(s: str, to_lower: bool = True):
    items = [x.strip() for x in s.split(",")]
    items = [x for x in items if x]  # quita vacíos
    if to_lower:
        items = [x.lower() for x in items]
    # quita duplicados conservando el orden
    seen = set()
    deduped = []
    for it in items:
        if it not in seen:
            deduped.append(it)
            seen.add(it)
    return deduped

if st.button("Clasificar", type="primary"):
    labels = parse_labels(etiquetas_raw, to_lower=normalize)
    if not texto.strip():
        st.warning("Por favor, ingresa un texto.")
    elif not labels:
        st.warning("Por favor, ingresa al menos una etiqueta.")
    else:
        with st.spinner("Clasificando…"):
            result = classifier(
                sequences=texto,
                candidate_labels=labels,
                multi_label=multi_label,
            )
        # El pipeline devuelve 'labels' y 'scores' alineados
        labels_out = result["labels"]
        scores_out = result["scores"]
        # A % y ordenar desc
        df = pd.DataFrame({"Etiqueta": labels_out, "Puntuación (%)": [round(s * 100, 2) for s in scores_out]})
        df = df.sort_values("Puntuación (%)", ascending=False).set_index("Etiqueta")

        st.subheader("Resultados")
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df)

        mejor = df.iloc[0]
        st.success(f"Etiqueta más afín: **{df.index[0]}** ({mejor['Puntuación (%)']} %)")
        st.caption("Las puntuaciones indican afinidad; no son probabilidades calibradas.")

st.markdown("---")
st.markdown(
    "💡 **Consejos**: "
    "- Si tu texto está en español, prueba el modelo XLM‑R para mejorar resultados. "
    "- Activa *multi‑label* cuando varias categorías puedan aplicar simultáneamente."
)
