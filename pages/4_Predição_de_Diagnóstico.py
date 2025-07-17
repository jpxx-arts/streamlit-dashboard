import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# --- Configuração da Página ---
st.set_page_config(
    page_title="Predição de Diagnóstico",
    page_icon="🤖",
    layout="wide"
)

# --- Carregamento e Cache dos Dados ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('alzheimers_disease_data.csv')
        # Remove colunas que não serão usadas como features
        df.drop(columns=["PatientID", "DoctorInCharge"], inplace=True)
        return df
    except FileNotFoundError:
        st.error("Arquivo 'alzheimers_disease_data.csv' não encontrado. Por favor, certifique-se que o arquivo está na pasta correta.")
        return None

df = load_data()

# Mapeamento para legendas (usado nas visualizações e inputs)
DIAGNOSIS_MAP = {0: 'Normal', 1: 'Comprometimento Cognitivo Leve', 2: 'Alzheimer'}
GENDER_MAP = {0: 'Homem', 1: 'Mulher'}
BOOL_MAP = {0: 'Não', 1: 'Sim'}

# --- Treinamento e Cache do Modelo ---
@st.cache_resource
def train_model(data):
    """
    Prepara os dados, treina um modelo RandomForest e retorna o modelo,
    dados de teste e colunas de features.
    """
    # Garante que a coluna 'Diagnosis' exista
    if 'Diagnosis' not in data.columns:
        st.error("A coluna 'Diagnosis' não foi encontrada no dataset.")
        return None, None, None, None, None

    # Lida com valores ausentes (se houver)
    data = data.dropna()
    
    # Define as features (X) e o alvo (y)
    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']
    
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Cria e treina o modelo
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=10)
    model.fit(X_train, y_train)
    
    # Calcula a importância das features
    feature_importances = pd.DataFrame(
        {'feature': X_train.columns, 'importance': model.feature_importances_}
    ).sort_values('importance', ascending=False)
    
    return model, X_test, y_test, X_train.columns, feature_importances

if df is not None:
    model, X_test, y_test, feature_names, feature_importances = train_model(df)

    st.title("🤖 Predição de Diagnóstico de Alzheimer")
    st.markdown("""
    Esta seção utiliza um modelo de *Machine Learning* (Random Forest) para prever o diagnóstico de um paciente. 
    Insira as informações na barra lateral esquerda para obter uma previsão. O modelo foi treinado com os dados do dataset e sua performance está detalhada no final da página.
    """)

    # --- Barra Lateral para Input do Usuário ---
    st.sidebar.header("Insira os Dados do Paciente:")

    def user_input_features():
        """Cria os widgets na barra lateral para o input do usuário."""
        inputs = {}
        
        # Inputs mais importantes primeiro (baseado na análise de feature importance)
        inputs['MMSE'] = st.sidebar.slider('MMSE (Mini-Exame do Estado Mental)', 0, 30, 25)
        inputs['CDR'] = st.sidebar.slider('CDR (Avaliação Clínica de Demência)', 0.0, 2.0, 0.5, 0.5)
        inputs['ADL'] = st.sidebar.slider('ADL (Atividades da Vida Diária)', 0.0, 1.0, 0.8, 0.1)
        inputs['Age'] = st.sidebar.slider('Idade', int(df['Age'].min()), int(df['Age'].max()), 70)
        inputs['FunctionalAssessment'] = st.sidebar.slider('Avaliação Funcional', 0.0, 10.0, 5.0, 0.5)
        inputs['EducationLevel'] = st.sidebar.selectbox('Nível de Escolaridade', options=list(range(5)), format_func=lambda x: {0: 'Sem Escolaridade', 1: 'Fundamental', 2: 'Médio', 3: 'Superior', 4: 'Pós-Graduação'}[x])
        
        with st.sidebar.expander("Outros Indicadores Clínicos e de Estilo de Vida"):
            inputs['Gender'] = st.sidebar.selectbox('Gênero', options=[0, 1], format_func=lambda x: GENDER_MAP[x])
            inputs['MemoryComplaints'] = st.sidebar.selectbox('Queixas de Memória', options=[0, 1], format_func=lambda x: BOOL_MAP[x])
            inputs['BehavioralProblems'] = st.sidebar.selectbox('Problemas Comportamentais', options=[0, 1], format_func=lambda x: BOOL_MAP[x])
            inputs['Depression'] = st.sidebar.selectbox('Depressão', options=[0, 1], format_func=lambda x: BOOL_MAP[x])
            inputs['HeadInjury'] = st.sidebar.selectbox('Histórico de Lesão na Cabeça', options=[0, 1], format_func=lambda x: BOOL_MAP[x])
            inputs['FamilyHistory'] = st.sidebar.selectbox('Histórico Familiar de Alzheimer', options=[0, 1], format_func=lambda x: BOOL_MAP[x])
            inputs['Hypertension'] = st.sidebar.selectbox('Hipertensão', options=[0, 1], format_func=lambda x: BOOL_MAP[x])
            inputs['HeartDisease'] = st.sidebar.selectbox('Doença Cardíaca', options=[0, 1], format_func=lambda x: BOOL_MAP[x])
            inputs['Diabetes'] = st.sidebar.selectbox('Diabetes', options=[0, 1], format_func=lambda x: BOOL_MAP[x])
            inputs['Smoking'] = st.sidebar.selectbox('Fumante', options=[0, 1], format_func=lambda x: BOOL_MAP[x])

            # Preenchimento de valores restantes com a média/mediana para simplificar a UI
            for col in feature_names:
                if col not in inputs:
                    inputs[col] = df[col].median()

        input_df = pd.DataFrame([inputs])
        input_df = input_df[feature_names] # Garante a ordem correta das colunas
        return input_df

    input_df = user_input_features()

    # --- Exibição da Predição ---
    st.header("Resultado da Predição")

    if st.sidebar.button("Realizar Predição"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        predicted_diagnosis = DIAGNOSIS_MAP[prediction[0]]
        confidence = prediction_proba.max()

        col1, col2 = st.columns([1, 2])
        
        with col1:
            if predicted_diagnosis == 'Alzheimer':
                st.error(f"**Diagnóstico Previsto:**\n## {predicted_diagnosis}")
            elif predicted_diagnosis == 'Comprometimento Cognitivo Leve':
                st.warning(f"**Diagnóstico Previsto:**\n## {predicted_diagnosis}")
            else:
                st.success(f"**Diagnóstico Previsto:**\n## {predicted_diagnosis}")
            st.metric(label="Nível de Confiança do Modelo", value=f"{confidence:.2%}")

        with col2:
            fig_proba = go.Figure(go.Bar(
                x=[p * 100 for p in prediction_proba[0]],
                y=[DIAGNOSIS_MAP[i] for i in range(len(prediction_proba[0]))],
                orientation='h',
                text=[f'{p:.1%}' for p in prediction_proba[0]],
                textposition='auto',
                marker_color=['#2ca02c', '#ff7f0e', '#d62728'] # Verde, Laranja, Vermelho
            ))
            fig_proba.update_layout(
                title_text='Probabilidade de cada Diagnóstico',
                xaxis_title='Probabilidade (%)',
                yaxis_title='Diagnóstico',
                yaxis_categoryorder='total ascending'
            )
            st.plotly_chart(fig_proba, use_container_width=True)

    else:
        st.info("Clique no botão 'Realizar Predição' na barra lateral para ver o resultado.")

    # --- Seção de Performance do Modelo ---
    st.markdown("---")
    with st.expander("Clique para ver a Performance e Detalhes do Modelo"):
        st.header("Avaliação de Performance do Modelo")
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.metric(label="**Acurácia do Modelo no Conjunto de Teste**", value=f"**{accuracy:.2%}**")
        
        col1, col2 = st.columns(2)

        with col1:
            # --- CORREÇÃO APLICADA AQUI ---
            # Define a ordem explícita das classes para garantir consistência
            class_labels = [0, 1, 2]
            class_names = [DIAGNOSIS_MAP[label] for label in class_labels]

            # Força a matriz de confusão a ter sempre o tamanho 3x3, usando os labels definidos
            cm = confusion_matrix(y_test, y_pred, labels=class_labels)
            
            fig_cm = px.imshow(cm,
                               labels=dict(x="Predição", y="Verdadeiro", color="Contagem"),
                               x=class_names, # Usa os nomes das classes
                               y=class_names, # Usa os nomes das classes
                               text_auto=True,
                               color_continuous_scale='Blues')
            fig_cm.update_layout(title='Matriz de Confusão')
            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            # Garante que o relatório também use a mesma ordem de labels
            class_labels = [0, 1, 2]
            class_names = [DIAGNOSIS_MAP[label] for label in class_labels]
            
            report = classification_report(y_test, y_pred, labels=class_labels, target_names=class_names, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(2))

        # Importância das Features
        st.header("Importância das Variáveis (Feature Importance)")
        st.markdown("Este gráfico mostra quais variáveis o modelo considera mais importantes para fazer uma predição. Quanto maior a barra, maior o impacto da variável no diagnóstico.")
        fig_importance = px.bar(feature_importances.head(15), 
                                x='importance', 
                                y='feature', 
                                orientation='h',
                                title='Top 15 Variáveis Mais Importantes',
                                labels={'importance': 'Importância', 'feature': 'Variável'})
        fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)

else:
    st.warning("Não foi possível carregar o dataset para iniciar a aplicação.")