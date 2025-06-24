import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Set the page title and layout
st.set_page_config(layout="wide")

# Title of the Streamlit App
st.title('Análise da Doença de Alzheimer')

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    """Loads the Alzheimer's dataset from a CSV file and drops unnecessary columns."""
    try:
        df = pd.read_csv('alzheimers_disease_data.csv')
        df.drop(columns=["PatientID", "DoctorInCharge"], inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: 'alzheimers_disease_data.csv' not found. Please make sure the file is in the same directory as app.py.")
        return None

df = load_data()

# --- Main App Logic ---
if df is not None:
    # --- Introduction and Data Display ---
    st.header('Visão Geral do Conjunto de Dados!')
    st.write("Abaixo estão as primeiras 5 linhas do conjunto de dados.")
    st.dataframe(df.head())

    # --- Correlation Matrix ---
    st.header('Matriz de Correlação')
    st.write("A matriz de correlação mostra a relação entre as diferentes variáveis numéricas no conjunto de dados.")

    corr_df = df.corr().stack().reset_index().rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'})
    corr_chart = alt.Chart(corr_df).mark_rect().encode(
        x=alt.X('variable:O', title=None),
        y=alt.Y('variable2:O', title=None),
        color=alt.Color('correlation:Q', scale=alt.Scale(scheme='viridis'), legend=alt.Legend(title="Correlação")),
        tooltip=[
            alt.Tooltip('variable:O', title='Variável 1'),
            alt.Tooltip('variable2:O', title='Variável 2'),
            alt.Tooltip('correlation:Q', title='Correlação', format='.2f')
        ]
    ).properties(
        title='Matriz de Correlação'
    ).interactive()
    st.altair_chart(corr_chart, use_container_width=True)


    # --- Age Distribution Analysis ---
    st.header('Análise da Distribuição por Idade')
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Distribuição da Idade por Diagnóstico')
        age_box_plot = alt.Chart(df).mark_boxplot(extent='min-max').encode(
            x=alt.X('Diagnosis:O', title='Diagnóstico', axis=alt.Axis(labels=False, ticks=False, domain=False)),
            y=alt.Y('Age:Q', title='Idade'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico'))
        ).properties(
            title='Distribuição da Idade por Diagnóstico'
        )
        st.altair_chart(age_box_plot, use_container_width=True)

    with col2:
        st.subheader('Distribuição da Idade com Diagnóstico')
        age_hist = alt.Chart(df).mark_bar().encode(
            x=alt.X('Age', bin=alt.Bin(maxbins=20), title='Idade'),
            y=alt.Y('count()', title='Contagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['Diagnosis:N', 'count()']
        ).properties(
            title='Distribuição da Idade por Diagnóstico'
        )
        st.altair_chart(age_hist, use_container_width=True)


    # --- Analysis by Gender, Ethnicity, and Education Level ---
    st.header('Análise por Gênero, Etnia e Nível de Escolaridade')

    # Gender Analysis
    st.subheader('Análise por Gênero')
    col1, col2 = st.columns(2)

    with col1:
        gender_count_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('Gender:N', title='Gênero'),
            y=alt.Y('count():Q', title='Contagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['Gender', 'Diagnosis', 'count()']
        ).properties(
            title='Distribuição do Diagnóstico por Gênero'
        )
        st.altair_chart(gender_count_plot, use_container_width=True)

    with col2:
        gender_proportion_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('Gender:N', title='Gênero'),
            y=alt.Y('count():Q', stack='normalize', axis=alt.Axis(format='%'), title='Porcentagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['Gender', 'Diagnosis']
        ).properties(
            title='Proporção do Diagnóstico por Gênero'
        )
        st.altair_chart(gender_proportion_plot, use_container_width=True)

    # Ethnicity Analysis
    st.subheader('Análise por Etnia')
    col1, col2 = st.columns(2)

    with col1:
        ethnicity_count_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('Ethnicity:N', title='Etnia'),
            y=alt.Y('count():Q', title='Contagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['Ethnicity', 'Diagnosis', 'count()']
        ).properties(
            title='Distribuição do Diagnóstico por Etnia'
        )
        st.altair_chart(ethnicity_count_plot, use_container_width=True)

    with col2:
        ethnicity_proportion_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('Ethnicity:N', title='Etnia'),
            y=alt.Y('count():Q', stack='normalize', axis=alt.Axis(format='%'), title='Porcentagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['Ethnicity', 'Diagnosis']
        ).properties(
            title='Proporção do Diagnóstico por Etnia'
        )
        st.altair_chart(ethnicity_proportion_plot, use_container_width=True)

    # Education Level Analysis
    st.subheader('Análise por Nível de Escolaridade')
    col1, col2 = st.columns(2)

    with col1:
        education_count_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('EducationLevel:N', title='Escolaridade', sort=['Nenhum', 'Ensino médio', 'Bacharelado', 'Pós-graduação']),
            y=alt.Y('count():Q', title='Contagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['EducationLevel', 'Diagnosis', 'count()']
        ).properties(
            title='Distribuição do Diagnóstico por Escolaridade'
        )
        st.altair_chart(education_count_plot, use_container_width=True)

    with col2:
        education_proportion_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('EducationLevel:N', title='Escolaridade', sort=['Nenhum', 'Ensino médio', 'Bacharelado', 'Pós-graduação']),
            y=alt.Y('count():Q', stack='normalize', axis=alt.Axis(format='%'), title='Porcentagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['EducationLevel', 'Diagnosis']
        ).properties(
            title='Proporção do Diagnóstico por Escolaridade'
        )
        st.altair_chart(education_proportion_plot, use_container_width=True)


    # --- Scatter Plots ---
    st.header('Análise de Relações')

    # MMSE vs. Age
    st.subheader('MMSE vs. Idade')
    mmse_age_scatter = alt.Chart(df).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X('MMSE:Q', title='MMSE (Mini-Exame do Estado Mental)'),
        y=alt.Y('Age:Q', title='Idade'),
        color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
        tooltip=['MMSE', 'Age', 'Diagnosis']
    ).properties(
        title='Relação entre MMSE e Idade'
    ).interactive()
    st.altair_chart(mmse_age_scatter, use_container_width=True)

    # ADL vs. Functional Assessment
    st.subheader('ADL vs. Avaliação Funcional')
    adl_functional_scatter = alt.Chart(df).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X('FunctionalAssessment:Q', title='Avaliação Funcional'),
        y=alt.Y('ADL:Q', title='ADL (Atividades da Vida Diária)'),
        color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
        tooltip=['FunctionalAssessment', 'ADL', 'Diagnosis']
    ).properties(
        title='Relação entre ADL e Avaliação Funcional'
    ).interactive()
    st.altair_chart(adl_functional_scatter, use_container_width=True)


    # --- Depression and Head Injury Analysis ---
    st.header('Análise de Depressão e Lesão na Cabeça')

    # Depression Analysis
    st.subheader('Análise de Depressão')
    col1, col2 = st.columns(2)
    with col1:
        depression_count_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('Depression:N', title='Depressão'),
            y=alt.Y('count():Q', title='Contagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['Depression', 'Diagnosis', 'count()']
        ).properties(
            title='Contagem de Diagnóstico por Depressão'
        )
        st.altair_chart(depression_count_plot, use_container_width=True)

    with col2:
        depression_proportion_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('Depression:N', title='Depressão'),
            y=alt.Y('count():Q', stack='normalize', axis=alt.Axis(format='%'), title='Porcentagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['Depression', 'Diagnosis']
        ).properties(
            title='Proporção de Diagnóstico por Depressão'
        )
        st.altair_chart(depression_proportion_plot, use_container_width=True)

    # Head Injury Analysis
    st.subheader('Análise de Lesão na Cabeça')
    col1, col2 = st.columns(2)
    with col1:
        head_injury_count_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('HeadInjury:N', title='Lesão na Cabeça'),
            y=alt.Y('count():Q', title='Contagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['HeadInjury', 'Diagnosis', 'count()']
        ).properties(
            title='Contagem de Diagnóstico por Lesão na Cabeça'
        )
        st.altair_chart(head_injury_count_plot, use_container_width=True)

    with col2:
        head_injury_proportion_plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('HeadInjury:N', title='Lesão na Cabeça'),
            y=alt.Y('count():Q', stack='normalize', axis=alt.Axis(format='%'), title='Porcentagem'),
            color=alt.Color('Diagnosis:N', legend=alt.Legend(title='Diagnóstico')),
            tooltip=['HeadInjury', 'Diagnosis']
        ).properties(
            title='Proporção de Diagnóstico por Lesão na Cabeça'
        )
        st.altair_chart(head_injury_proportion_plot, use_container_width=True)

    # --- MMSE Distribution by Education Level ---
    st.header('Distribuição do MMSE por Nível de Escolaridade')
    mmse_education_boxplot = alt.Chart(df).mark_boxplot(extent='min-max').encode(
        x=alt.X('EducationLevel:N', title='Nível de Escolaridade', sort=['Nenhum', 'Ensino médio', 'Bacharelado', 'Pós-graduação']),
        y=alt.Y('MMSE:Q', title='MMSE'),
        color=alt.Color('EducationLevel:N', legend=None)
    ).properties(
        title='Distribuição do MMSE por Nível de Escolaridade'
    )
    st.altair_chart(mmse_education_boxplot, use_container_width=True)

