# ============================================================
# 🚀 STREAMLIT DASHBOARD – CLUSTERING SEMÁNTICO 2 NIVELES
# Compatible con Streamlit Cloud
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -----------------------------------------------------------------
# 1. CONFIGURACIÓN STREAMLIT
# -----------------------------------------------------------------

st.set_page_config(
    page_title="Dashboard – Clustering Semántico",
    layout="wide"
)

st.title(" Dashboard – Clustering Semántico (Nivel 2)")
st.write("Versión desplegada en Streamlit Cloud")

# -----------------------------------------------------------------
# 2. CARGAR DATASET (debes subir el CSV al repo)
# -----------------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("dataset_clustering_semantico_2nivel_nombres.csv")

df = load_data()

# Columnas
COL_TITULO = "puesto_cluster_ready"
COL_CLUSTER = "cluster_refinado_sub"
COL_CAT_ORIGINAL = "Categoría"
COL_SILHOUETTE = "silhouette_score"
COL_CAT_SEM = "categoria_semantica_final"

# -----------------------------------------------------------------
# 3. MÉTRICAS GENERALES
# -----------------------------------------------------------------

st.subheader(" Métricas Generales")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de registros", len(df))
col2.metric("Categorías originales", df[COL_CAT_ORIGINAL].nunique())
col3.metric("Clusters refinados", df[COL_CLUSTER].nunique())
col4.metric("Categorías semánticas finales", df[COL_CAT_SEM].nunique())

# -----------------------------------------------------------------
# 4. DISTRIBUCIÓN DE CATEGORÍAS SEMÁNTICAS
# -----------------------------------------------------------------

st.subheader(" Distribución por Categoría Semántica Final")

count_df = df[COL_CAT_SEM].value_counts().reset_index()
count_df.columns = ["categoria", "frecuencia"]

fig = px.bar(
    count_df,
    x="categoria",
    y="frecuencia",
    title="Distribución de categorías semánticas",
    labels={"categoria": "Categoría Semántica", "frecuencia": "Cantidad"}
)
st.plotly_chart(fig, use_container_width=True)

# =================================================================
# SANKEY MEJORADO: Se limita a las top N categorías originales
# =================================================================

st.subheader("🔗 Flujo (Reducido): Categoría Original → Categoría Semántica Final")

TOP_N = st.slider(
    "Cantidad de categorías originales a mostrar (recomendado 10–20)",
    min_value=5,
    max_value=40,
    value=15
)

def sankey_simplificado(df, col_source, col_target, top_n=15):
    # Eliminar filas sin datos
    df_clean = df.dropna(subset=[col_source, col_target])

    # Obtener top N categorías originales
    top_sources = (
        df_clean[col_source]
        .value_counts()
        .nlargest(top_n)
        .index
    )

    # Filtrar
    df_top = df_clean[df_clean[col_source].isin(top_sources)]

    # Agrupar
    links = df_top.groupby([col_source, col_target]).size().reset_index(name="count")

    # Crear nodos únicos
    all_labels = list(links[col_source].unique()) + list(links[col_target].unique())
    label_to_id = {label: i for i, label in enumerate(all_labels)}

    # Dibujar sankey
    fig = px.sankey(
        links,
        node=dict(label=all_labels, pad=20, thickness=15),
        source=links[col_source].map(label_to_id),
        target=links[col_target].map(label_to_id),
        value="count",
    )

    fig.update_layout(
        title=f"Sankey – Top {top_n} categorías → categorías semánticas",
        height=600
    )
    return fig

# Mostrar sankey
st.plotly_chart(
    sankey_simplificado(df, COL_CAT_ORIGINAL, COL_CAT_SEM, TOP_N),
    use_container_width=True
)


# -----------------------------------------------------------------
# 6. WORDCLOUD
# -----------------------------------------------------------------

st.subheader(" Nube de Palabras por Categoría Semántica")

categoria_wc = st.selectbox("Selecciona categoría semántica:", df[COL_CAT_SEM].unique())
subset_wc = df[df[COL_CAT_SEM] == categoria_wc]

text = " ".join(subset_wc[COL_TITULO].dropna().astype(str))

wc = WordCloud(width=800, height=400, background_color="white").generate(text)

fig_wc, ax = plt.subplots(figsize=(10, 4))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig_wc)

# -----------------------------------------------------------------
# 7. SILHOUETTE
# -----------------------------------------------------------------

st.subheader(" Análisis de Silhouette Score")

fig_sil = px.histogram(
    df,
    x=COL_SILHOUETTE,
    nbins=30,
    title="Distribución de Silhouette Score"
)
st.plotly_chart(fig_sil, use_container_width=True)

# -----------------------------------------------------------------
# 8. TABLA Y EXPORTACIÓN
# -----------------------------------------------------------------

st.subheader(" Tabla detallada y exportación")

filtro = st.multiselect(
    "Filtrar por categoría semántica",
    df[COL_CAT_SEM].unique(),
    default=df[COL_CAT_SEM].unique()
)

df_filtrado = df[df[COL_CAT_SEM].isin(filtro)]
st.dataframe(df_filtrado, use_container_width=True, height=400)

csv_data = df_filtrado.to_csv(index=False).encode("utf-8")
st.download_button(
    label=" Descargar CSV filtrado",
    data=csv_data,
    file_name="cluster_filtrado.csv",
    mime="text/csv"
)

