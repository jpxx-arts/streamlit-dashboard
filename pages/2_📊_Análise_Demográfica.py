import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Define a configuração da página, título e ícone
st.set_page_config(
    page_title="Análise Demográfica",
    page_icon="📊",
    layout="wide"
)

# --- Função de Carregamento de Dados ---
@st.cache_data
def load_data():
    """Carrega o dataset de Alzheimer a partir de um arquivo CSV e remove colunas desnecessárias."""
    try:
        # Tenta carregar os dados do arquivo CSV
        df = pd.read_csv('alzheimers_disease_data.csv')
        # Remove colunas que não serão utilizadas na análise
        df.drop(columns=["PatientID", "DoctorInCharge"], inplace=True)
        return df
    except FileNotFoundError:
        # Exibe uma mensagem de erro se o arquivo não for encontrado
        st.error("Erro: 'alzheimers_disease_data.csv' não encontrado. Por favor, certifique-se de que o arquivo está no mesmo diretório que o app.py.")
        return None

df = load_data()

# Título principal do aplicativo
st.title("📊 Análise Demográfica")

if df is not None:
    # --- Análise de Distribuição por Idade ---
    st.header('Distribuição por Idade')
    col1, col2 = st.columns(2)
    with col1:
        # Cria um box plot da idade por diagnóstico
        fig_age_box = px.box(
            df,
            x='Diagnosis',
            y='Age',
            color='Diagnosis',
            title='Distribuição da Idade por Diagnóstico',
            labels={'Age': 'Idade', 'Diagnosis': 'Diagnóstico'}
        )
        st.plotly_chart(fig_age_box, use_container_width=True)
    with col2:
        # Cria um histograma da idade, colorido pelo diagnóstico
        fig_age_hist = px.histogram(
            df,
            x='Age',
            color='Diagnosis',
            nbins=20,
            title='Distribuição da Idade por Diagnóstico',
            labels={'Age': 'Idade', 'count': 'Contagem'}
        )
        st.plotly_chart(fig_age_hist, use_container_width=True)

    # --- Análise por Gênero ---
    st.header('Distribuição por Gênero')
    col1, col2 = st.columns(2)
    with col1:
        # Gráfico de contagem por gênero
        fig_gender_count = px.histogram(
            df,
            x='Gender',
            color='Diagnosis',
            barmode='group',
            title='Distribuição do Diagnóstico por Gênero',
            labels={'Gender': 'Gênero', 'count': 'Contagem'}
        )
        st.plotly_chart(fig_gender_count, use_container_width=True)
    with col2:
        # Gráfico de proporção por gênero
        fig_gender_prop = px.histogram(
            df,
            x='Gender',
            color='Diagnosis',
            barnorm='percent',
            text_auto='.2f',
            title='Proporção do Diagnóstico por Gênero (%)',
            labels={'Gender': 'Gênero', 'percent': 'Porcentagem'}
        )
        st.plotly_chart(fig_gender_prop, use_container_width=True)

    # --- Análise por Etnia ---
    st.header('Distribuição por Etnia')
    col1, col2 = st.columns(2)
    with col1:
        # Gráfico de contagem por etnia
        fig_ethnicity_count = px.histogram(
            df,
            x='Ethnicity',
            color='Diagnosis',
            barmode='group',
            title='Distribuição do Diagnóstico por Etnia',
            labels={'Ethnicity': 'Etnia', 'count': 'Contagem'}
        )
        st.plotly_chart(fig_ethnicity_count, use_container_width=True)
    with col2:
        # Gráfico de proporção por etnia
        fig_ethnicity_prop = px.histogram(
            df,
            x='Ethnicity',
            color='Diagnosis',
            barnorm='percent',
            text_auto='.2f',
            title='Proporção do Diagnóstico por Etnia (%)',
            labels={'Ethnicity': 'Etnia', 'percent': 'Porcentagem'}
        )
        st.plotly_chart(fig_ethnicity_prop, use_container_width=True)

    # --- Análise por Nível de Escolaridade ---
    st.header('Distribuição por Nível de Escolaridade')
    education_order = ['No Schooling', 'Primary School', 'High School', 'Bachelors Degree', 'Graduate Degree']
    col1, col2 = st.columns(2)
    with col1:
        # Gráfico de contagem por escolaridade
        fig_education_count = px.histogram(
            df,
            x='EducationLevel',
            color='Diagnosis',
            barmode='group',
            category_orders={'EducationLevel': education_order},
            title='Distribuição do Diagnóstico por Escolaridade',
            labels={'EducationLevel': 'Nível de Escolaridade', 'count': 'Contagem'}
        )
        st.plotly_chart(fig_education_count, use_container_width=True)
    with col2:
        # Gráfico de proporção por escolaridade
        fig_education_prop = px.histogram(
            df,
            x='EducationLevel',
            color='Diagnosis',
            barnorm='percent',
            text_auto='.2f',
            category_orders={'EducationLevel': education_order},
            title='Proporção do Diagnóstico por Escolaridade (%)',
            labels={'EducationLevel': 'Nível de Escolaridade', 'percent': 'Porcentagem'}
        )
        st.plotly_chart(fig_education_prop, use_container_width=True)
