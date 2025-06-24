import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Análise Clínica",
    page_icon="🩺",
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

st.title("🩺 Análise Clínica e de Comorbidades")

if df is not None:
    # --- Scatter Plots de Relações ---
    st.header('Análise de Relações Clínicas')
    
    # MMSE vs. Age
    mmse_age_scatter = alt.Chart(df).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X('MMSE:Q', title='MMSE (Mini-Exame do Estado Mental)'),
        y=alt.Y('Age:Q', title='Idade'),
        color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
        tooltip=['MMSE', 'Age', 'Diagnosis']
    ).properties(title='Relação entre MMSE e Idade').interactive()
    st.altair_chart(mmse_age_scatter, use_container_width=True)

    # ADL vs. Functional Assessment
    adl_functional_scatter = alt.Chart(df).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X('FunctionalAssessment:Q', title='Avaliação Funcional'),
        y=alt.Y('ADL:Q', title='ADL (Atividades da Vida Diária)'),
        color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
        tooltip=['FunctionalAssessment', 'ADL', 'Diagnosis']
    ).properties(title='Relação entre ADL e Avaliação Funcional').interactive()
    st.altair_chart(adl_functional_scatter, use_container_width=True)

    # --- Análise de Depressão e Lesão na Cabeça ---
    st.header('Análise de Comorbidades')
    
    # Análise de Depressão
    st.subheader('Depressão')
    col1, col2 = st.columns(2)
    with col1:
        depression_count_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('Depression:N', title='Depressão'),
            y=alt.Y('count():Q', title='Contagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['Depression', 'Diagnosis', 'count()']
        ).properties(title='Contagem de Diagnóstico por Depressão')
        st.altair_chart(depression_count_plot, use_container_width=True)
    with col2:
        depression_proportion_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('Depression:N', title='Depressão'),
            y=alt.Y('count():Q', stack='normalize', axis=alt.Axis(format='%'), title='Porcentagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['Depression', 'Diagnosis']
        ).properties(title='Proporção de Diagnóstico por Depressão')
        st.altair_chart(depression_proportion_plot, use_container_width=True)

    # Análise de Lesão na Cabeça
    st.subheader('Histórico de Lesão na Cabeça')
    col1, col2 = st.columns(2)
    with col1:
        head_injury_count_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('HeadInjury:N', title='Lesão na Cabeça'),
            y=alt.Y('count():Q', title='Contagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['HeadInjury', 'Diagnosis', 'count()']
        ).properties(title='Contagem de Diagnóstico por Lesão na Cabeça')
        st.altair_chart(head_injury_count_plot, use_container_width=True)
    with col2:
        head_injury_proportion_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('HeadInjury:N', title='Lesão na Cabeça'),
            y=alt.Y('count():Q', stack='normalize', axis=alt.Axis(format='%'), title='Porcentagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['HeadInjury', 'Diagnosis']
        ).properties(title='Proporção de Diagnóstico por Lesão na Cabeça')
        st.altair_chart(head_injury_proportion_plot, use_container_width=True)

    # --- Distribuição do MMSE por Nível de Escolaridade ---
    st.header('Distribuição do MMSE por Nível de Escolaridade')
    mmse_education_boxplot = alt.Chart(df).mark_boxplot(extent='min-max').encode(
        x=alt.X('EducationLevel:N', title='Nível de Escolaridade', sort=['Nenhum', 'Ensino médio', 'Bacharelado', 'Pós-graduação']),
        y=alt.Y('MMSE:Q', title='MMSE'),
        color=alt.Color('EducationLevel:N', legend=None)
    ).properties(title='Distribuição do MMSE por Nível de Escolaridade')
    st.altair_chart(mmse_education_boxplot, use_container_width=True)
