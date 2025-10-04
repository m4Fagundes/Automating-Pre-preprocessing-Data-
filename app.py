# app.py - Versão Final de Apresentação
# Autor: Gemini & m4fagundes
# Data: 2025-10-04 (07:35 AM)
# Descrição: Ferramenta híbrida com datasets de exemplo e upload de arquivo.

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from zipfile import ZipFile
from datetime import datetime
import google.generativeai as genai
import json
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. CONFIGURAÇÃO INICIAL
# ==============================================================================
st.set_page_config(layout="wide", page_title="Consultor de Dados IA", page_icon="🚀")

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    LLM_OK = True
except (KeyError, AttributeError):
    LLM_OK = False

# ==============================================================================
# 2. FUNÇÕES AUXILIARES, DE AÇÃO E DE ANÁLISE
# ==============================================================================

# --- Funções de Ação e Transformação ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def impute_median(df, column):
    if pd.api.types.is_numeric_dtype(df[column]):
        median_val = df[column].median()
        df[column] = df[column].fillna(median_val)
        st.toast(f"Nulos em '{column}' preenchidos com a mediana ({median_val:.2f}).", icon="✅")
    return df

def remove_outliers_iqr(df, column):
    if pd.api.types.is_numeric_dtype(df[column]):
        Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior, limite_superior = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        original_rows = len(df)
        df_filtered = df[(df[column] >= limite_inferior) & (df[column] <= limite_superior)]
        removed_rows = original_rows - len(df_filtered)
        st.toast(f"{removed_rows} outliers removidos de '{column}'.", icon="✅")
        return df_filtered
    return df

def log_transform(df, column):
    if pd.api.types.is_numeric_dtype(df[column]):
        if (df[column] <= 0).any():
            st.warning(f"Coluna '{column}' contém valores não-positivos, log1p será usado.")
            df[column] = np.log1p(df[column] - df[column].min())
        else:
            df[column] = np.log1p(df[column])
        st.toast(f"Transformação de Log aplicada em '{column}'.", icon="✅")
    return df

ACTION_MAP = {
    "IMPUTE_MEDIAN": impute_median,
    "REMOVE_OUTLIERS_IQR": remove_outliers_iqr,
    "LOG_TRANSFORM": log_transform,
}

# --- Funções de Análise Algorítmica ---
def analisar_nulos(df):
    nulos = df.isnull().sum()[lambda x: x > 0]
    if not nulos.empty: return {"tipo": "Valores Nulos", "detalhes": nulos.to_dict()}
    return None

def analisar_outliers(series):
    if pd.api.types.is_numeric_dtype(series):
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior, limite_superior = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = series[(series < limite_inferior) | (series > limite_superior)]
        if not outliers.empty: return {"tipo": "Outliers", "coluna": series.name, "detalhes": {"método": "IQR", "contagem": len(outliers), "exemplos": outliers.head(3).tolist()}}
    return None

# --- Funções de Relatório e Interação com LLM ---
def formatar_achados_para_prompt(findings):
    if not findings: return "Nenhum problema algorítmico significativo foi detectado."
    prompt_text = "Resultados detalhados da análise algorítmica:\n\n"
    for finding in findings:
        prompt_text += f"- **Tipo:** {finding.get('tipo', 'N/A')}\n"
        if 'coluna' in finding: prompt_text += f"  - **Coluna:** {finding['coluna']}\n"
        prompt_text += f"  - **Detalhes:**\n```json\n{json.dumps(finding['detalhes'], indent=2)}\n```\n\n"
    return prompt_text

def gerar_plano_de_acao_com_llm(target, user_prompt, findings_text):
    model = genai.GenerativeModel('gemini-2.0-flash')
    master_prompt = f"""
    **Persona:** Você é um Cientista de Dados Sênior. **Tarefa:** Analise o dossiê e o contexto para criar um plano de ação em JSON. **Contexto:** - Variável Alvo: `{target}` - Objetivo: "{user_prompt}" **Dossiê:** {findings_text} **Instruções:** 1. Priorize os problemas mais críticos. 2. Para cada um, crie um objeto JSON com: "action_code", "column", "justification". Use os action_codes: [`IMPUTE_MEDIAN`, `REMOVE_OUTLIERS_IQR`, `LOG_TRANSFORM`]. 3. Retorne uma lista na chave "action_plan". Apenas o JSON.
    """
    try:
        response = model.generate_content(master_prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return cleaned_response
    except Exception as e:
        return json.dumps({"error": f"Erro na API: {e}"})

def gerar_texto_relatorio(file_name, original_shape, final_shape, target_column, user_context, applied_actions):
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    actions_md_table = "| Ação Aplicada | Coluna | Justificativa da IA |\n| :--- | :--- | :--- |\n"
    if not applied_actions: actions_md_table = "Nenhuma ação foi aplicada manualmente."
    else:
        for action in applied_actions:
            actions_md_table += f"| `{action.get('action_code', 'N/A')}` | `{action.get('column', 'N/A')}` | {action.get('justification', 'N/A')} |\n"
    report_content = f"""
# Relatório de Preparação de Dados
- **Dataset:** `{file_name}`
- **Data:** `{report_date}`
- **Preparado por:** Consultor de Dados IA
---
### 1. Objetivo do Modelo
> "{user_context}"
- **Variável Alvo:** `{target_column}`
### 2. Transformações Aplicadas
{actions_md_table}
### 3. Estado Final do Dataset
- **Shape Original:** {original_shape[0]} linhas, {original_shape[1]} colunas.
- **Shape Final:** {final_shape[0]} linhas, {final_shape[1]} colunas.
"""
    return report_content

def criar_pacote_zip(df_final, texto_relatorio, nome_arquivo_original):
    in_memory_zip = BytesIO()
    nome_base = nome_arquivo_original.split('/')[-1].split('.')[0]
    with ZipFile(in_memory_zip, 'w') as zipf:
        zipf.writestr(f"processed_{nome_base}.csv", convert_df_to_csv(df_final))
        zipf.writestr("relatorio_de_preparacao.md", texto_relatorio)
    return in_memory_zip.getvalue()

# ==============================================================================
# 4. INTERFACE PRINCIPAL E FLUXO DA APLICAÇÃO
# ==============================================================================

st.title("🚀 Consultor de Dados IA")
st.write("Uma ferramenta híbrida que usa algoritmos e IA para criar e aplicar seu plano de tratamento de dados.")

# --- Seletor de Fonte de Dados ---
st.header("1. Escolha sua Fonte de Dados")
DATASET_OPTIONS = {
    "Selecione uma opção": None,
    "Exemplo: Titanic (train.csv) - Para Teste de Classificação": "sample_datasets/test.csv",
    "Fazer upload do meu próprio arquivo (.csv ou .xlsx)": "upload"
}
selected_option_key = st.selectbox("Gostaria de usar um dataset de exemplo ou fazer o upload do seu?", options=list(DATASET_OPTIONS.keys()))
st.caption("Nota: Os datasets de exemplo são os arquivos `train.csv` de competições, que contêm a variável alvo necessária para a análise.")

source_identifier = DATASET_OPTIONS[selected_option_key]
uploaded_file = None
df = None

if source_identifier == "upload":
    uploaded_file = st.file_uploader("Para garantir a melhor análise, seu arquivo deve conter a coluna alvo.", type=['csv', 'xlsx'])
    if uploaded_file: source_identifier = uploaded_file.name

# --- Lógica de Carregamento e Gerenciamento de Estado ---
if source_identifier and source_identifier != "upload":
    if 'current_source' not in st.session_state or st.session_state.current_source != source_identifier:
        st.session_state.current_source = source_identifier
        df_to_load = pd.read_csv(source_identifier) if not uploaded_file else (pd.read_excel(uploaded_file, engine='openpyxl') if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file))
        st.session_state.df = df_to_load.copy()
        st.session_state.original_shape = df_to_load.shape
        st.session_state.data_modified = False
        st.session_state.applied_actions = []
        if 'action_plan' in st.session_state: del st.session_state.action_plan
    
    df = st.session_state.df

# --- O RESTO DO APP SÓ RODA SE UM DATAFRAME (df) EXISTIR ---
if df is not None:
    st.sidebar.header("🎯 Configuração do Modelo")
    target_column = st.sidebar.selectbox("1. Selecione sua variável alvo:", options=[None] + df.columns.tolist())
    user_context = st.sidebar.text_area("2. Descreva o objetivo:", placeholder="Ex: Prever o risco de inadimplência...")
    run_llm_analysis = st.sidebar.button("Gerar Plano de Ação com IA 🧠")

    st.divider()
    st.header("2. Análise e Ação")
    st.subheader(f"Visão Geral do Dataset: `{st.session_state.current_source.split('/')[-1]}`")
    st.dataframe(df.head())
    st.write(f"Shape atual: **{df.shape[0]} linhas** e **{df.shape[1]} colunas**.")

    if st.session_state.get('data_modified', False):
        report_text = gerar_texto_relatorio(st.session_state.current_source, st.session_state.original_shape, df.shape, target_column, user_context, st.session_state.applied_actions)
        zip_data = criar_pacote_zip(df, report_text, st.session_state.current_source)
        st.download_button(
           label="Baixar Pacote de Entrega (CSV + Relatório) 📦",
           data=zip_data,
           file_name=f"entrega_{st.session_state.current_source.split('/')[-1].split('.')[0]}.zip",
           mime='application/zip',
           use_container_width=True
        )

    achados_estruturados = []
    resultado_nulos = analisar_nulos(df)
    if resultado_nulos: achados_estruturados.append(resultado_nulos)
    features_df = df.drop(columns=[target_column]) if target_column and target_column in df.columns else df
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
        if not LLM_OK: st.error("A API do LLM não está configurada. Verifique seu arquivo .streamlit/secrets.toml.")
        elif not target_column or not user_context: st.error("Por favor, selecione a variável alvo e descreva o objetivo do modelo na barra lateral.")
        else:
            with st.spinner("🧠 O especialista de IA está analisando o dossiê e preparando seu plano de ação..."):
                texto_dos_achados = formatar_achados_para_prompt(achados_estruturados)
                plano_de_acao_json_str = gerar_plano_de_acao_com_llm(target_column, user_context, texto_dos_achados)
                try:
                    plano = json.loads(plano_de_acao_json_str)
                    if "error" in plano: st.error(f"A API retornou um erro: {plano['error']}")
                    st.session_state.action_plan = plano.get("action_plan", [])
                except json.JSONDecodeError:
                    st.error("A IA retornou uma resposta em um formato JSON inválido."); st.code(plano_de_acao_json_str)
                    st.session_state.action_plan = []
                st.rerun()

    if 'action_plan' in st.session_state and st.session_state.action_plan:
        st.subheader("✅ Plano de Ação Interativo", divider='rainbow')
        for i, action in enumerate(list(st.session_state.action_plan)):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Ação:** `{action.get('action_code', 'N/A')}` na coluna **`{action.get('column', 'N/A')}`**")
                st.info(f"**Justificativa da IA:** {action.get('justification', 'N/A')}")
            with col2:
                if st.button("Aplicar esta Ação ✨", key=f"apply_{i}", use_container_width=True):
                    action_func = ACTION_MAP.get(action['action_code'])
                    if action_func:
                        st.session_state.applied_actions.append(action)
                        st.session_state.df = action_func(st.session_state.df, action['column'])
                        st.session_state.data_modified = True
                        st.session_state.action_plan.pop(i)
                        st.rerun()
                    else:
                        st.error(f"Ação '{action['action_code']}' não implementada.")

else:
    st.info("Aguardando a seleção de um dataset ou o upload de um arquivo para iniciar a análise.")