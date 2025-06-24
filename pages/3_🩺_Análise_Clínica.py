import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Análise Clínica",
    page_icon="🩺",
    layout="wide"
)

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

st.title("🩺 Análise Clínica e de Comorbidades")

if df is not None:
    st.header('Análise de Relações Clínicas')
    
    fig_mmse_age_scatter = px.scatter(
        df,
        x='MMSE',
        y='Age',
        color='Diagnosis',
        hover_data=['Gender', 'EducationLevel'],
        labels={'MMSE': 'MMSE (Mini-Exame do Estado Mental)', 'Age': 'Idade', 'Diagnosis': 'Diagnóstico'},
        title='Relação entre MMSE e Idade'
    )
    st.plotly_chart(fig_mmse_age_scatter, use_container_width=True)

    fig_adl_functional_scatter = px.scatter(
        df,
        x='FunctionalAssessment',
        y='ADL',
        color='Diagnosis',
        hover_data=['Gender', 'EducationLevel'],
        labels={'FunctionalAssessment': 'Avaliação Funcional', 'ADL': 'ADL (Atividades da Vida Diária)', 'Diagnosis': 'Diagnóstico'},
        title='Relação entre ADL e Avaliação Funcional'
    )
    st.plotly_chart(fig_adl_functional_scatter, use_container_width=True)

    st.header('Análise de Comorbidades')
    
    st.subheader('Depressão')
    col1, col2 = st.columns(2)
    with col1:
        fig_depression_count = px.histogram(
            df, 
            x='Depression', 
            color='Diagnosis', 
            barmode='group',
            title='Contagem de Diagnóstico por Depressão',
            labels={'Depression': 'Depressão', 'count': 'Contagem', 'Diagnosis': 'Diagnóstico'}
        )
        st.plotly_chart(fig_depression_count, use_container_width=True)
    with col2:
        fig_depression_prop = px.histogram(
            df, 
            x='Depression', 
            color='Diagnosis', 
            barnorm='percent',
            text_auto='.2f',
            title='Proporção de Diagnóstico por Depressão (%)',
            labels={'Depression': 'Depressão', 'percent': 'Porcentagem', 'Diagnosis': 'Diagnóstico'}
        )
        st.plotly_chart(fig_depression_prop, use_container_width=True)

    st.subheader('Histórico de Lesão na Cabeça')
    col1, col2 = st.columns(2)
    with col1:
        fig_head_injury_count = px.histogram(
            df, 
            x='HeadInjury', 
            color='Diagnosis', 
            barmode='group',
            title='Contagem de Diagnóstico por Lesão na Cabeça',
            labels={'HeadInjury': 'Lesão na Cabeça', 'count': 'Contagem', 'Diagnosis': 'Diagnóstico'}
        )
        st.plotly_chart(fig_head_injury_count, use_container_width=True)
    with col2:
        fig_head_injury_prop = px.histogram(
            df, 
            x='HeadInjury', 
            color='Diagnosis', 
            barnorm='percent',
            text_auto='.2f',
            title='Proporção de Diagnóstico por Lesão na Cabeça (%)',
            labels={'HeadInjury': 'Lesão na Cabeça', 'percent': 'Porcentagem', 'Diagnosis': 'Diagnóstico'}
        )
        st.plotly_chart(fig_head_injury_prop, use_container_width=True)
    
    st.subheader('Contagem e proporção de variáveis com alta correlação')

    diagnosis_classif = st.selectbox("Variáveis:", ["MMSE", "FunctionalAssessment", "MemoryComplaints", "BehavioralProblems", "ADL"])

    col1, col2 = st.columns(2)
    with col1:
        fig_head_injury_count = px.histogram(
            df, 
            x=diagnosis_classif, 
            color='Diagnosis', 
            barmode='group',
            title=f'Contagem de Diagnóstico por {diagnosis_classif}',
            labels={'count': 'Contagem', 'Diagnosis': 'Diagnóstico'}
        )
        st.plotly_chart(fig_head_injury_count, use_container_width=True)
    with col2:
        fig_head_injury_prop = px.histogram(
            df, 
            x=diagnosis_classif, 
            color='Diagnosis', 
            barnorm='percent',
            text_auto='.2f',
            title=f'Proporção de Diagnóstico por {diagnosis_classif}',
            labels={'percent': 'Porcentagem', 'Diagnosis': 'Diagnóstico'}
        )
        st.plotly_chart(fig_head_injury_prop, use_container_width=True)


