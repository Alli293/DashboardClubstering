# ============================================================
# 🚀 STREAMLIT DASHBOARD – CLUSTERING SEMÁNTICO 2 NIVELES
# Versión corregida: omite "Administración / Oficina" y simplifica Sankey
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Dashboard – Clustering Semántico ", layout="wide")
st.title("Dashboard – Clustering Semántico ")

# ---------------------------
# CONSTANTES (columnas)
# ---------------------------
COL_TITULO = "puesto_cluster_ready"
COL_CLUSTER = "cluster_refinado_sub"
COL_CAT_ORIGINAL = "Categoría"
COL_SILHOUETTE = "silhouette_score"
COL_CAT_SEM = "categoria_semantica_final"

# ---------------------------
# CARGAR DATOS
# ---------------------------
@st.cache_data
def load_data(path="dataset_clustering_semantico_2nivel_nombres.csv"):
    df = pd.read_csv(path)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error cargando el CSV: {e}")
    st.stop()

# ---------------------------
# PREPROCESADO: eliminar Administración/Oficina y filas raras
# ---------------------------
def clean_and_filter_admin(df):
    df = df.copy()
    # Normalizar strings y evitar NaNs
    for col in [COL_CAT_ORIGINAL, COL_CAT_SEM, COL_TITULO]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()
        else:
            st.error(f"Columna obligatoria no encontrada: {col}")
            st.stop()

    # Excluir textos que contengan "administración" o "oficina" u "admin" (case-insensitive)
    mask_admin_orig = df[COL_CAT_ORIGINAL].str.contains(r"administración|oficina|admin", case=False, na=False)
    mask_admin_sem = df[COL_CAT_SEM].str.contains(r"administración|oficina|admin", case=False, na=False)

    df_filtered = df[~(mask_admin_orig | mask_admin_sem)].copy()
    return df_filtered

df = clean_and_filter_admin(df)

# ---------------------------
# SIDEBAR: controles
# ---------------------------
st.sidebar.header("Filtros y configuración")
min_cluster_size = st.sidebar.slider("Excluir clusters con menos de (registros)", 0, 200, 3, step=1)
top_categories_sankey = st.sidebar.slider("Top categorías originales (Sankey)", 3, 30, 8)
top_semantic_sankey = st.sidebar.slider("Top categorías semánticas (Sankey)", 3, 30, 8)
st.sidebar.markdown("Nota: Administración/Oficina ya está excluida automáticamente.")

# ---------------------------
# Filtrar clusters pequeños
# ---------------------------
cluster_counts = df[COL_CAT_SEM].value_counts()
big_clusters = cluster_counts[cluster_counts >= min_cluster_size].index.tolist()
df = df[df[COL_CAT_SEM].isin(big_clusters)].copy()

# ---------------------------
# SECCIÓN: Métricas generales
# ---------------------------
st.subheader(" Métricas Generales (sin Administración / Oficina)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de registros (filtrados)", f"{len(df):,}")
col2.metric("Categorías originales", df[COL_CAT_ORIGINAL].nunique())
col3.metric("Clusters refinados (ids)", df[COL_CLUSTER].nunique())
col4.metric("Categorías semánticas finales (filtradas)", df[COL_CAT_SEM].nunique())

st.markdown("---")

# ---------------------------
# DISTRIBUCIÓN BARRAS (asegurando formato correcto)
# ---------------------------
st.subheader(" Distribución por Categoría Semántica Final")

try:
    counts_sem = df[COL_CAT_SEM].value_counts().reset_index(name="count")
    counts_sem.columns = ["categoria_semantica", "count"]
    fig_bar = px.bar(
        counts_sem.sort_values("count", ascending=True),
        x="count",
        y="categoria_semantica",
        orientation="h",
        labels={"categoria_semantica": "Categoría Semántica", "count": "Frecuencia"},
        title="Distribución de categorías semánticas (ordenada)"
    )
    fig_bar.update_layout(height=600)
    st.plotly_chart(fig_bar, use_container_width=True)
except Exception as e:
    st.error(f"Error generando gráfico de barras: {e}")

st.markdown("---")

# ---------------------------
# FUNCIÓN: preparar sankey simplificado
# ---------------------------
def prepare_sankey(df, source_col, target_col, top_src=8, top_tgt=8):
    # Contar
    links = df.groupby([source_col, target_col]).size().reset_index(name="count")

    # Top source categories
    top_sources = df[source_col].value_counts().nlargest(top_src).index.tolist()
    # Top target categories (semantic)
    top_targets = df[target_col].value_counts().nlargest(top_tgt).index.tolist()

    # Map everything else to 'Otros' for both sides
    links["source_mod"] = links[source_col].where(links[source_col].isin(top_sources), "Otros")
    links["target_mod"] = links[target_col].where(links[target_col].isin(top_targets), "Otros")

    agg = links.groupby(["source_mod", "target_mod"], as_index=False)["count"].sum()

    # Crear nodos
    all_nodes = list(pd.Index(agg["source_mod"].unique()).append(pd.Index(agg["target_mod"].unique())).unique())
    label_to_idx = {label: i for i, label in enumerate(all_nodes)}

    # Crear sankey arrays
    sources = agg["source_mod"].map(label_to_idx).tolist()
    targets = agg["target_mod"].map(label_to_idx).tolist()
    values = agg["count"].tolist()

    # Colores simples (opcional)
    node_colors = ["#444"] * len(all_nodes)
    link_colors = ["rgba(0,0,0,0.2)"] * len(values)

    return all_nodes, sources, targets, values, node_colors, link_colors

# ---------------------------
# DIAGRAMA SANKEY (simplificado)
# ---------------------------
st.subheader(" Sankey: Categoría Original → Categoría Semántica (simplificado)")

try:
    nodes, sources, targets, values, node_colors, link_colors = prepare_sankey(
        df, COL_CAT_ORIGINAL, COL_CAT_SEM, top_src=top_categories_sankey, top_tgt=top_semantic_sankey
    )

    if len(values) == 0:
        st.warning("No hay enlaces para mostrar en Sankey con los filtros actuales.")
    else:
        sankey_fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=10,
                thickness=15,
                line=dict(color="black", width=0.5),
                label=nodes,
                color=node_colors
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors
            ))])

        sankey_fig.update_layout(title_text="Flujo: Categoría Original → Categoría Semántica (agrupe 'Otros')", height=700)
        st.plotly_chart(sankey_fig, use_container_width=True)
except Exception as e:
    st.error(f"Error generando Sankey: {e}")

st.markdown("---")

# ---------------------------
# WORDCLOUD POR CATEGORÍA SEMÁNTICA
# ---------------------------
st.subheader(" Nube de Palabras por Categoría Semántica")

try:
    options_sem = sorted(df[COL_CAT_SEM].unique())
    if not options_sem:
        st.warning("No hay categorías semánticas disponibles para WordCloud.")
    else:
        categoria_wc = st.selectbox("Selecciona categoría semántica (WordCloud):", options_sem)
        subset_wc = df[df[COL_CAT_SEM] == categoria_wc]
        text = " ".join(subset_wc[COL_TITULO].dropna().astype(str))

        if text.strip():
            wc = WordCloud(width=1200, height=400, background_color="white", max_words=150).generate(text)

            fig_wc, ax = plt.subplots(figsize=(12, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)
        else:
            st.warning("No hay texto suficiente para generar la nube de palabras en esa categoría.")
except Exception as e:
    st.error(f"Error en WordCloud: {e}")

st.markdown("---")

# ---------------------------
# ANÁLISIS DE SILHOUETTE
# ---------------------------
st.subheader(" Análisis de Silhouette")

try:
    if COL_SILHOUETTE in df.columns and pd.api.types.is_numeric_dtype(df[COL_SILHOUETTE]):
        fig_sil = px.histogram(df, x=COL_SILHOUETTE, nbins=30, title="Distribución de Silhouette Score")
        st.plotly_chart(fig_sil, use_container_width=True)
    else:
        st.info("Columna de silhouette no encontrada o no numérica. Omitiendo gráfico de silhouette.")
except Exception as e:
    st.error(f"Error generando gráfico de silhouette: {e}")

st.markdown("---")

# ---------------------------
# TABLA DETALLADA Y EXPORTACIÓN
# ---------------------------
st.subheader(" Tabla detallada y exportación (filtrada)")

try:
    filtro = st.multiselect("Filtrar por categoría semántica (tabla):", sorted(df[COL_CAT_SEM].unique()), default=sorted(df[COL_CAT_SEM].unique()))
    df_filtrado = df[df[COL_CAT_SEM].isin(filtro)].copy()
    st.dataframe(df_filtrado.reset_index(drop=True), use_container_width=True, height=400)

    csv_data = df_filtrado.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label="⬇️ Descargar CSV filtrado", data=csv_data, file_name="cluster_filtrado.csv", mime="text/csv")
except Exception as e:
    st.error(f"Error mostrando tabla/exportando: {e}")

st.markdown("---")

# ---------------------------
# FOOTER / SUGERENCIAS
# ---------------------------
st.sidebar.markdown("### Sugerencias")
st.sidebar.markdown("- Ajusta 'Excluir clusters con menos de' para quitar clusters muy pequeños.\n- Reduce 'Top categorías' en Sankey para simplificar el diagrama.\n- Administración/Oficina ya se excluye automáticamente.")

