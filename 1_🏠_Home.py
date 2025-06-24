import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")

st.title("Dashboard - Alzheimer's Disease")

st.markdown('### Trabalho 2 - Dashboard')
st.markdown("**Departamento de Engenharia de Computação e Automação - DCA UFRN - 2025**")
st.markdown("**Disciplina:** Ciência de Dados (DCA-3501)")
st.markdown("**Professor:** Luiz Affonso Guedes")
st.markdown("**Alunos:** João Pedro Araújo Ramalho & Kiev Luiz Freitas Guedes")

st.markdown("### Dataset")
st.markdown("[Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset/)")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('alzheimers_disease_data.csv')
        df.drop(columns=["PatientID", "DoctorInCharge"], inplace=True)
        return df
    except FileNotFoundError:
        st.error("'alzheimers_disease_data.csv' não encontrado")
        return None

df = load_data()

if df is not None:
    st.header('Matriz de Correlação')
    st.write("Relação entre as variáveis do conjunto.")

    corr_matrix = df.corr(numeric_only=True)
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        labels=dict(color="Correlação"),
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig_corr.update_layout(title='Matriz de Correlação das Variáveis Numéricas')
    st.plotly_chart(fig_corr, use_container_width=True)

    st.header(f"Dataset e Estatísticas")

    tab1, tab2 = st.tabs(["Tabela", "Estatísticas"])

    with tab1:
        st.markdown(f"Dados dos {df.shape[0]} pacientes:")

        st.dataframe(df)

    with tab2:
        st.markdown("Estatística de cada variável:")
        st.dataframe(df.describe().T)


    st.header('Análise do Diagnóstico e Idade')
    col1, col2, col3 = st.columns(3)

    with col1:
        fig_age_box = px.box(
            df,
            x='Diagnosis',
            y='Age',
            color='Diagnosis',
            labels={'Age': 'Idade', 'Diagnosis': 'Diagnóstico'},
            title='Distribuição da Idade por Diagnóstico'
        )
        st.plotly_chart(fig_age_box, use_container_width=True)

    with col2:
        fig_age_hist = px.histogram(
            df,
            x='Age',
            color='Diagnosis',
            nbins=20,
            labels={'Age': 'Idade', 'count': 'Contagem'},
            title='Distribuição da Idade por Diagnóstico'
        )
        st.plotly_chart(fig_age_hist, use_container_width=True)
    
    with col3:
        diagnosis_counts = df['Diagnosis'].value_counts().reset_index()
        diagnosis_counts.columns = ['Diagnosis', 'Count']
        
        fig_pie_diagnosis = px.pie(
            diagnosis_counts,
            names='Diagnosis',
            values='Count',
            title='Proporção de Diagnósticos',
            hole=0.3,
            labels={'Diagnosis': 'Diagnóstico', 'Count': 'Número de Pacientes'}
        )
        fig_pie_diagnosis.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie_diagnosis, use_container_width=True)