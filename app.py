# app.py - Versão Final do Consultor de Dados IA
# Autor: Gemini & m4fagundes
# Data: 2025-10-03
# Descrição: Uma ferramenta híbrida que usa algoritmos para análise e um LLM para recomendação e ação.

import streamlit as st
import pandas as pd
import numpy as np
from thefuzz import process
import openpyxl
import google.generativeai as genai
import json
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    layout="wide",
    page_title="Consultor de Dados IA",
    page_icon="🧠"
)

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    LLM_OK = True
except (KeyError, AttributeError):
    LLM_OK = False


def impute_median(df, column):
    """Preenche valores nulos com a mediana da coluna."""
    if pd.api.types.is_numeric_dtype(df[column]):
        median_val = df[column].median()
        df[column].fillna(median_val, inplace=True)
        st.toast(f"Nulos em '{column}' preenchidos com a mediana ({median_val:.2f}).")
    return df

def remove_outliers_iqr(df, column):
    """Remove outliers de uma coluna usando o método IQR."""
    if pd.api.types.is_numeric_dtype(df[column]):
        Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior, limite_superior = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        
        original_rows = len(df)
        df_filtered = df[(df[column] >= limite_inferior) & (df[column] <= limite_superior)]
        removed_rows = original_rows - len(df_filtered)
        st.toast(f"{removed_rows} outliers removidos de '{column}'.")
        return df_filtered
    return df

def log_transform(df, column):
    """Aplica a transformação logarítmica (log1p)."""
    if pd.api.types.is_numeric_dtype(df[column]) and (df[column] >= 0).all():
        df[column] = np.log1p(df[column])
        st.toast(f"Transformação de Log aplicada em '{column}'.")
    return df

ACTION_MAP = {
    "IMPUTE_MEDIAN": impute_median,
    "REMOVE_OUTLIERS_IQR": remove_outliers_iqr,
    "LOG_TRANSFORM": log_transform,
}



def analisar_nulos(df):
    nulos = df.isnull().sum()[lambda x: x > 0]
    if not nulos.empty:
        return {"tipo": "Valores Nulos", "detalhes": nulos.to_dict()}
    return None

def analisar_outliers(series):
    if pd.api.types.is_numeric_dtype(series):
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior, limite_superior = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = series[(series < limite_inferior) | (series > limite_superior)]
        if not outliers.empty:
            return {
                "tipo": "Outliers", "coluna": series.name,
                "detalhes": {
                    "método": "IQR", "contagem": len(outliers),
                    "exemplos": outliers.head(3).tolist()
                }
            }
    return None

def formatar_achados_para_prompt(findings):
    if not findings: return "Nenhum problema algorítmico significativo foi detectado."
    prompt_text = "Aqui estão os resultados detalhados da análise algorítmica:\n\n"
    for finding in findings:
        prompt_text += f"- **Tipo:** {finding.get('tipo', 'N/A')}\n"
        if 'coluna' in finding:
            prompt_text += f"  - **Coluna:** {finding['coluna']}\n"
        prompt_text += f"  - **Detalhes:**\n```json\n{json.dumps(finding['detalhes'], indent=2)}\n```\n\n"
    return prompt_text

def gerar_plano_de_acao_com_llm(target, user_prompt, findings_text):
    model = genai.GenerativeModel('gemini-2.0-flash')
    master_prompt = f"""
    **Persona:** Você é um Cientista de Dados Sênior e um engenheiro de software.
    **Tarefa:** Analise o dossiê de evidências e o contexto do usuário para criar um plano de ação. Sua resposta DEVE SER UM OBJETO JSON VÁLIDO.
    **Contexto:**
    - Variável Alvo: `{target}`
    - Objetivo do Modelo: "{user_prompt}"
    **Dossiê de Evidências:**
    {findings_text}
    **Instruções:**
    1.  Priorize os 2-3 problemas mais críticos.
    2.  Para cada recomendação, crie um objeto JSON com: "action_code" (um de [`IMPUTE_MEDIAN`, `REMOVE_OUTLIERS_IQR`, `LOG_TRANSFORM`]), "column", "justification", e "parameters" (dicionário vazio por enquanto).
    3.  Retorne uma lista desses objetos na chave "action_plan". NÃO inclua nada fora do objeto JSON.
    """

    try:
        response = model.generate_content(master_prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return cleaned_response
    except Exception as e:
        return json.dumps({"error": f"Erro na API: {e}"})



st.title("🧠 Consultor de Dados IA")
st.write("Um sistema híbrido que combina análise algorítmica com a inteligência de um LLM para criar e aplicar seu plano de tratamento de dados.")

uploaded_file = st.file_uploader("Escolha um arquivo (.csv ou .xlsx)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    if 'df' not in st.session_state or st.session_state.file_name != uploaded_file.name:
        st.session_state.df = pd.read_excel(uploaded_file, engine='openpyxl') if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        st.session_state.file_name = uploaded_file.name
        if 'action_plan' in st.session_state:
            del st.session_state.action_plan

    df = st.session_state.df

    st.sidebar.header("🎯 Configuração do Modelo")
    target_column = st.sidebar.selectbox("1. Selecione sua variável alvo:", options=[None] + df.columns.tolist())
    user_context = st.sidebar.text_area("2. Descreva o objetivo do seu modelo:", placeholder="Ex: Prever o risco de inadimplência. O modelo precisa ser justo, explicável e robusto.")
    run_llm_analysis = st.sidebar.button("Gerar Plano de Ação com IA 🧠")

    st.divider()
    st.subheader(f"Visão Geral do Dataset: `{st.session_state.file_name}`")
    st.dataframe(df.head())
    st.write(f"Shape atual: **{df.shape[0]} linhas** e **{df.shape[1]} colunas**.")

    achados_estruturados = []
    resultado_nulos = analisar_nulos(df)
    if resultado_nulos: achados_estruturados.append(resultado_nulos)

    features_df = df.drop(columns=[target_column]) if target_column else df
    for col in features_df.columns:
        resultado_outliers = analisar_outliers(features_df[col])
        if resultado_outliers: achados_estruturados.append(resultado_outliers)

    st.subheader("Dossiê de Evidências (Análise Algorítmica)")
    if not achados_estruturados:
        st.success("A análise algorítmica inicial não encontrou problemas significativos.")
    else:
        st.info("Os seguintes pontos foram identificados e serão enviados ao especialista de IA:")
        st.json(achados_estruturados, expanded=False)

    if run_llm_analysis:
        if not LLM_OK:
            st.error("A API do LLM não está configurada. Verifique seu arquivo .streamlit/secrets.toml.")
        elif not target_column or not user_context:
            st.error("Por favor, selecione a variável alvo e descreva o objetivo do modelo na barra lateral.")
        else:
            with st.spinner("🧠 O especialista de IA está analisando o dossiê e preparando seu plano de ação..."):
                texto_dos_achados = formatar_achados_para_prompt(achados_estruturados)
                plano_de_acao_json_str = gerar_plano_de_acao_com_llm(target_column, user_context, texto_dos_achados)
                
                try:
                    plano = json.loads(plano_de_acao_json_str)
                    if "error" in plano:
                         st.error(f"A API retornou um erro: {plano['error']}")
                         st.session_state.action_plan = []
                    else:
                        st.session_state.action_plan = plano.get("action_plan", [])
                except json.JSONDecodeError:
                    st.error("A IA retornou uma resposta em um formato JSON inválido. Tente novamente.")
                    st.code(plano_de_acao_json_str)
                    st.session_state.action_plan = []

    if 'action_plan' in st.session_state and st.session_state.action_plan:
        st.subheader("✅ Plano de Ação Interativo", divider='rainbow')
        for i, action in enumerate(st.session_state.action_plan):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Ação:** `{action.get('action_code', 'N/A')}` na coluna **`{action.get('column', 'N/A')}`**")
                st.info(f"**Justificativa da IA:** {action.get('justification', 'N/A')}")
            with col2:
                if st.button("Aplicar esta Ação ✨", key=f"apply_{i}", use_container_width=True):
                    action_func = ACTION_MAP.get(action['action_code'])
                    if action_func:
                        st.session_state.df = action_func(st.session_state.df, action['column'])
                        st.session_state.action_plan.pop(i)
                        st.rerun()
                    else:
                        st.error(f"Ação '{action['action_code']}' não implementada.")