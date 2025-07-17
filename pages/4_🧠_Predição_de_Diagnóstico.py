import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

st.set_page_config(
    page_title="Classificador de Alzheimer",
    page_icon="🧠",
    layout="wide"
)

DIAGNOSIS_MAP = {0: 'Não Alzheimer', 1: 'Alzheimer'}
GENDER_MAP = {0: 'Homem', 1: 'Mulher'}
EDUCATION_MAP = {0: 'Nenhum', 1: 'Ensino Médio', 2: 'Bacharelado', 3: 'Superior'}
BOOL_MAP = {0: 'Não', 1: 'Sim'}

@st.cache_data
def load_data():
    """Carrega o dataset e o retorna, mantendo em cache para performance."""
    try:
        df = pd.read_csv('alzheimers_disease_data.csv')
        df = df.drop(columns=["PatientID", "DoctorInCharge"])
        return df
    except FileNotFoundError:
        st.error("Arquivo 'alzheimers_disease_data.csv' não encontrado. Verifique se ele está na pasta correta.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o arquivo: {e}")
        return None

@st.cache_resource
def train_classification_model(df):
    """
    Usa GridSearchCV para encontrar os melhores hiperparâmetros e treinar o modelo.
    """
    try:
        FEATURES = [col for col in df.columns if col != 'Diagnosis']
        TARGET = 'Diagnosis'

        X = df[FEATURES]
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [10, 20, None], # None = sem limite de profundidade
            'min_samples_split': [2, 5]
        }

        rf = RandomForestClassifier(random_state=42)

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring='accuracy'
        )

        st.write("Iniciando a busca pelos melhores hiperparâmetros...")
        grid_search.fit(X_train, y_train)
        st.write("O melhor modelo foi encontrado.")

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        feature_importances = pd.DataFrame({
            'feature': FEATURES,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        return best_model, X_test, y_test, FEATURES, feature_importances, best_params

    except KeyError as e:
        st.error(f"Erro: A coluna {e} não foi encontrada no seu dataset.")
        return None, None, None, None, None, None


st.title("🧠 Classificador de Doença de Alzheimer")
st.markdown("Esta ferramenta utiliza uma Random Forest para prever a probabilidade de um diagnóstico de Alzheimer com base nos dados do paciente.")

df_data = load_data()

if df_data is not None:
    model_components = train_classification_model(df_data)
    
    if model_components and all(comp is not None for comp in model_components):
        model, X_test, y_test, FEATURES, feature_importances, best_params = model_components 

        st.sidebar.header("Insira os Dados do Paciente")
        input_data = {}

        for feature in FEATURES:
            if df_data[feature].dtype in ['int64', 'float64'] and df_data[feature].nunique() > 5:
                min_val = float(df_data[feature].min())
                max_val = float(df_data[feature].max())
                mean_val = float(df_data[feature].mean())
                input_data[feature] = st.sidebar.slider(f'**{feature}**', min_val, max_val, value=mean_val, help=f"Faixa: {min_val} a {max_val}")
            
            elif feature == 'EducationLevel':
                input_data[feature] = st.sidebar.selectbox(f'**{feature}**', options=list(EDUCATION_MAP.keys()), format_func=lambda x: EDUCATION_MAP[x])
            
            elif feature == 'Gender':
                input_data[feature] = st.sidebar.selectbox(f'**{feature}**', options=list(GENDER_MAP.keys()), format_func=lambda x: GENDER_MAP[x])

            else:
                input_data[feature] = st.sidebar.selectbox(f'**{feature}**', options=[0, 1], format_func=lambda x: BOOL_MAP.get(x, str(x)))
        
        if st.sidebar.button("Classificar Diagnóstico", type="primary"):
            input_df = pd.DataFrame([input_data])[FEATURES]

            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            predicted_diagnosis_str = DIAGNOSIS_MAP[prediction[0]]
            confidence = prediction_proba[0][prediction[0]]

            st.header("Resultado da Classificação")
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction[0] == 1:
                    st.error(f"Diagnóstico Previsto: **{predicted_diagnosis_str}**")
                else:
                    st.success(f"Diagnóstico Previsto: **{predicted_diagnosis_str}**")
                
                st.metric(label="Confiança do Modelo na Previsão", value=f"{confidence:.2%}")
            
            with col2:
                fig_proba = px.bar(
                    x=[f"{p:.1%}" for p in prediction_proba[0]],
                    y=list(DIAGNOSIS_MAP.values()),
                    orientation='h',
                    labels={'x': 'Probabilidade', 'y': 'Diagnóstico'},
                    title='Probabilidade de Cada Classe',
                    text=[f"{p:.1%}" for p in prediction_proba[0]]
                )
                fig_proba.update_traces(marker_color=['green', 'red'])
                st.plotly_chart(fig_proba, use_container_width=True)
        else:
            st.info("Ajuste os parâmetros na barra lateral e clique em 'Classificar Diagnóstico'.")

        with st.expander("Ver Performance e Detalhes do Modelo"):
            st.subheader("Performance (em dados de teste)")
            y_pred_test = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred_test)
            st.metric(label="Acurácia Geral", value=f"{accuracy:.2%}")

            st.subheader("Matriz de Confusão")
            cm = confusion_matrix(y_test, y_pred_test)
            fig_cm = px.imshow(cm, text_auto=True,
                               labels=dict(x="Predição", y="Verdadeiro"),
                               x=list(DIAGNOSIS_MAP.values()), y=list(DIAGNOSIS_MAP.values()),
                               color_continuous_scale='Blues')
            st.plotly_chart(fig_cm)

            st.subheader("Importância das Variáveis")
            st.markdown("Mostra o impacto de cada variável na decisão do modelo.")
            fig_imp = px.bar(feature_importances.head(15), 
                             x='importance', y='feature', orientation='h',
                             title='Top 15 Variáveis Mais Importantes')
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp)

            st.subheader("Melhores Hiperparâmetros Encontrados")
            st.json(best_params)