import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Análise Demográfica",
    page_icon="📊",
    layout="wide"
)

# --- Função de Carregamento de Dados ---
@st.cache_data
def load_data():
    """Carrega o dataset de Alzheimer a partir de um arquivo CSV."""
    try:
        df = pd.read_csv('alzheimers_disease_data.csv')
        df.drop(columns=["PatientID", "DoctorInCharge"], inplace=True)
        return df
    except FileNotFoundError:
        st.error("Erro: 'alzheimers_disease_data.csv' não encontrado.")
        return None

df = load_data()

st.title("📊 Análise Demográfica")

if df is not None:
    # --- Análise de Distribuição por Idade ---
    st.header('Distribuição por Idade')
    col1, col2 = st.columns(2)
    with col1:
        age_box_plot = alt.Chart(df).mark_boxplot(extent='min-max').encode(
            x=alt.X('Diagnosis:O', title='Diagnóstico', axis=alt.Axis(labels=False, ticks=False, domain=False)),
            y=alt.Y('Age:Q', title='Idade'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico'))
        ).properties(title='Distribuição da Idade por Diagnóstico')
        st.altair_chart(age_box_plot, use_container_width=True)
    with col2:
        age_hist = alt.Chart(df).mark_bar().encode(
            x=alt.X('Age', bin=alt.Bin(maxbins=20), title='Idade'),
            y=alt.Y('count()', title='Contagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['Diagnosis:N', 'count()']
        ).properties(title='Distribuição da Idade por Diagnóstico')
        st.altair_chart(age_hist, use_container_width=True)

    # --- Análise por Gênero ---
    st.header('Distribuição por Gênero')
    col1, col2 = st.columns(2)
    with col1:
        gender_count_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('Gender:N', title='Gênero'),
            y=alt.Y('count():Q', title='Contagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['Gender', 'Diagnosis', 'count()']
        ).properties(title='Distribuição do Diagnóstico por Gênero')
        st.altair_chart(gender_count_plot, use_container_width=True)
    with col2:
        gender_proportion_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('Gender:N', title='Gênero'),
            y=alt.Y('count():Q', stack='normalize', axis=alt.Axis(format='%'), title='Porcentagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['Gender', 'Diagnosis']
        ).properties(title='Proporção do Diagnóstico por Gênero')
        st.altair_chart(gender_proportion_plot, use_container_width=True)

    # --- Análise por Etnia ---
    st.header('Distribuição por Etnia')
    col1, col2 = st.columns(2)
    with col1:
        ethnicity_count_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('Ethnicity:N', title='Etnia'),
            y=alt.Y('count():Q', title='Contagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['Ethnicity', 'Diagnosis', 'count()']
        ).properties(title='Distribuição do Diagnóstico por Etnia')
        st.altair_chart(ethnicity_count_plot, use_container_width=True)
    with col2:
        ethnicity_proportion_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('Ethnicity:N', title='Etnia'),
            y=alt.Y('count():Q', stack='normalize', axis=alt.Axis(format='%'), title='Porcentagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['Ethnicity', 'Diagnosis']
        ).properties(title='Proporção do Diagnóstico por Etnia')
        st.altair_chart(ethnicity_proportion_plot, use_container_width=True)

    # --- Análise por Nível de Escolaridade ---
    st.header('Distribuição por Nível de Escolaridade')
    col1, col2 = st.columns(2)
    with col1:
        education_count_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('EducationLevel:N', title='Escolaridade', sort=['Nenhum', 'Ensino médio', 'Bacharelado', 'Pós-graduação']),
            y=alt.Y('count():Q', title='Contagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['EducationLevel', 'Diagnosis', 'count()']
        ).properties(title='Distribuição do Diagnóstico por Escolaridade')
        st.altair_chart(education_count_plot, use_container_width=True)
    with col2:
        education_proportion_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('EducationLevel:N', title='Escolaridade', sort=['Nenhum', 'Ensino médio', 'Bacharelado', 'Pós-graduação']),
            y=alt.Y('count():Q', stack='normalize', axis=alt.Axis(format='%'), title='Porcentagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['EducationLevel', 'Diagnosis']
        ).properties(title='Proporção do Diagnóstico por Escolaridade')
        st.altair_chart(education_proportion_plot, use_container_width=True)
