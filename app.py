# ============================================================
# 🚀 STREAMLIT DASHBOARD – CLUSTERING SEMÁNTICO (Sin Administración/Oficina y sin "Otros")
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Dashboard – Clustering Semántico", layout="wide")
st.title("Dashboard – Clustering Semántico")

# ---------------------------
# CONSTANTES
# ---------------------------
COL_TITULO = "puesto_cluster_ready"
COL_CLUSTER = "cluster_refinado_sub"
COL_CAT_ORIGINAL = "Categoría"
COL_SILHOUETTE = "silhouette_score"
COL_CAT_SEM = "categoria_semantica_final"

# ---------------------------
# CARGAR CSV
# ---------------------------
@st.cache_data
def load_data(path="dataset_clustering_semantico_2nivel_nombres.csv"):
    return pd.read_csv(path)

df = load_data()

# ---------------------------
# LIMPIEZA: quitar Administración / Oficina
# ---------------------------
def clean_admin(df):
    df = df.copy()
    df[COL_CAT_ORIGINAL] = df[COL_CAT_ORIGINAL].astype(str).fillna("").str.strip()
    df[COL_CAT_SEM] = df[COL_CAT_SEM].astype(str).fillna("").str.strip()
    df[COL_TITULO] = df[COL_TITULO].astype(str).fillna("")

    mask_orig = df[COL_CAT_ORIGINAL].str.contains(r"administración|oficina|admin", case=False)
    mask_sem = df[COL_CAT_SEM].str.contains(r"administración|oficina|admin", case=False)
    return df[~(mask_orig | mask_sem)]

df = clean_admin(df)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("Filtros")
min_cluster_size = st.sidebar.slider("Excluir clusters con menos de X registros:", 0, 200, 3)
top_src = st.sidebar.slider("Top categorías originales para Sankey:", 3, 30, 8)
top_tgt = st.sidebar.slider("Top categorías semánticas para Sankey:", 3, 30, 8)

# (❗ Eliminado el bloque de sugerencias completamente)
# st.sidebar.markdown("### Sugerencias")  ← eliminado

# ---------------------------
# FILTRAR CLUSTERS PEQUEÑOS
# ---------------------------
cluster_counts = df[COL_CAT_SEM].value_counts()
valid_clusters = cluster_counts[cluster_counts >= min_cluster_size].index
df = df[df[COL_CAT_SEM].isin(valid_clusters)]

# ---------------------------
# MÉTRICAS
# ---------------------------
st.subheader(" Métricas Generales (Administración/Oficina excluido)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total registros", len(df))
c2.metric("Categorías originales", df[COL_CAT_ORIGINAL].nunique())
c3.metric("Clusters refinados", df[COL_CLUSTER].nunique())
c4.metric("Categorías semánticas finales", df[COL_CAT_SEM].nunique())

st.markdown("---")

# ---------------------------
# GRÁFICO DE BARRAS
# ---------------------------
st.subheader(" Distribución por Categoría Semántica")

counts_sem = df[COL_CAT_SEM].value_counts().reset_index()
counts_sem.columns = ["categoria_semantica", "count"]

fig_bar = px.bar(
    counts_sem.sort_values("count", ascending=True),
    x="count",
    y="categoria_semantica",
    orientation="h",
)
fig_bar.update_layout(height=600)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ============================================================
#  FUNCIÓN SANKEY SIN “OTROS”
# ============================================================
def prepare_sankey_no_otros(df, source_col, target_col, top_src, top_tgt):
    # Top categorías
    top_sources = df[source_col].value_counts().nlargest(top_src).index.tolist()
    top_targets = df[target_col].value_counts().nlargest(top_tgt).index.tolist()

    # Filtrar SOLO pares que están en top ambas
    df_f = df[df[source_col].isin(top_sources) & df[target_col].isin(top_targets)]

    # Agrupar
    agg = df_f.groupby([source_col, target_col]).size().reset_index(name="count")

    # Crear nodos
    nodes_src = list(agg[source_col].unique())
    nodes_tgt = list(agg[target_col].unique())
    nodes = nodes_src + nodes_tgt

    node_index = {label: i for i, label in enumerate(nodes)}

    sources = agg[source_col].map(node_index).tolist()
    targets = agg[target_col].map(node_index).tolist()
    values = agg["count"].tolist()

    return nodes, sources, targets, values

# ---------------------------
# SANKEY FINAL
# ---------------------------
st.subheader(" Sankey: Categoría Original → Categoría Semántica (sin 'Otros')")

nodes, sources, targets, values = prepare_sankey_no_otros(df, COL_CAT_ORIGINAL, COL_CAT_SEM, top_src, top_tgt)

if len(values) == 0:
    st.warning("No hay datos suficientes con estos filtros para construir el Sankey.")
else:
    fig_sankey = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=10,
            thickness=14,
            line=dict(color="black", width=0.3),
            label=nodes,
            color="#444",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(0,0,0,0.25)"
        )
    )])

    fig_sankey.update_layout(height=700)
    st.plotly_chart(fig_sankey, use_container_width=True)

st.markdown("---")

# ---------------------------
# WORDCLOUD
# ---------------------------
st.subheader(" Nube de Palabras")

options_sem = sorted(df[COL_CAT_SEM].unique())
cat_sel = st.selectbox("Selecciona categoría:", options_sem)
text = " ".join(df[df[COL_CAT_SEM] == cat_sel][COL_TITULO])

wc = WordCloud(width=1200, height=400, background_color="white").generate(text)
fig_wc, ax = plt.subplots(figsize=(12, 4))
ax.imshow(wc)
ax.axis("off")
st.pyplot(fig_wc)

st.markdown("---")

# ---------------------------
# TABLE EXPORT
# ---------------------------
st.subheader(" Tabla detallada")

filtro = st.multiselect("Filtrar categorías semánticas:", options_sem, default=options_sem)
df_filtrado = df[df[COL_CAT_SEM].isin(filtro)]

st.dataframe(df_filtrado, use_container_width=True, height=420)

csv = df_filtrado.to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ Descargar CSV", csv, "cluster_filtrado.csv", "text/csv")
