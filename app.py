# app.py - Versão Final com Sistema Híbrido (Algoritmo + LLM)
# Autor: Gemini (com base nas suas ideias)
# Data: 03/10/2025

import streamlit as st
import pandas as pd
import numpy as np
from thefuzz import process
import openpyxl
import google.generativeai as genai
import json

# --- 1. CONFIGURAÇÃO INICIAL ---
st.set_page_config(layout="wide", page_title="Consultor de Dados IA", page_icon="🧠")

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    LLM_OK = True
except (KeyError, AttributeError):
    LLM_OK = False

# --- 2. FUNÇÕES DE ANÁLISE - AGORA RETORNAM DADOS ESTRUTURADOS ---
# --- MUDANÇA CRÍTICA ---

def analisar_nulos(df):
    nulos = df.isnull().sum()[lambda x: x > 0]
    if not nulos.empty:
        return {"tipo": "Valores Nulos", "detalhes": nulos.to_dict()}
    return None

def analisar_inconsistencias(series):
    unique_values = series.dropna().unique().tolist()
    if len(unique_values) < 2: return None
    # Lógica de encontrar inconsistências... (simplificada para o exemplo)
    grupos = [] # Substitua pela sua lógica real, ex: encontrar_inconsistencias_categoricas
    if grupos:
        return {"tipo": "Inconsistência Categórica", "coluna": series.name, "detalhes": {"grupos_sugeridos": grupos}}
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
                    "exemplos": outliers.head(3).tolist(), "limites": (limite_inferior, limite_superior)
                }
            }
    return None

# --- 3. FUNÇÕES DE SUPORTE AO LLM ---

def formatar_achados_para_prompt(findings):
    """Transforma a lista de dicionários de achados em um texto formatado."""
    if not findings:
        return "Nenhum problema algorítmico significativo foi detectado."
    
    prompt_text = "Aqui estão os resultados detalhados da análise algorítmica:\n\n"
    for finding in findings:
        prompt_text += f"- **Tipo de Achado:** {finding.get('tipo', 'N/A')}\n"
        if 'coluna' in finding:
            prompt_text += f"  - **Coluna Afetada:** {finding['coluna']}\n"
        prompt_text += f"  - **Detalhes Quantitativos:**\n```json\n{json.dumps(finding['detalhes'], indent=2)}\n```\n\n"
    return prompt_text

def gerar_plano_de_acao_com_llm(target, user_prompt, findings_text):
    """Monta o prompt avançado e chama o LLM."""
    model = genai.GenerativeModel('gemini-2.0-flash')

    master_prompt = f"""
    **Persona:** Você é um Cientista de Dados Sênior e consultor especialista em preparação de dados para Machine Learning.

    **Tarefa:** Sua tarefa é analisar um dossiê de "evidências" gerado por algoritmos e, combinando isso com o objetivo de negócio do usuário, criar um plano de ação priorizado. Você deve agir como um filtro inteligente, focando apenas nos problemas mais críticos.

    **Contexto do Problema (fornecido pelo usuário):**
    - **Variável Alvo a ser prevista:** `{target}`
    - **Objetivo do Modelo:** "{user_prompt}"

    **Dossiê de Evidências (Resultados da Análise Algorítmica):**
    {findings_text}

    **Suas Instruções:**
    1.  **Analise o Dossiê:** Revise todas as evidências quantitativas.
    2.  **Priorize:** Com base no **objetivo do modelo** do usuário, identifique os 2 ou 3 problemas mais críticos que terão o maior impacto negativo se não forem tratados. Ignore os problemas de baixo impacto.
    3.  **Crie um Plano de Ação:** Para cada problema crítico, forneça uma recomendação clara e acionável.
    4.  **Justifique com Dados:** Explique **por que** cada recomendação é importante, usando os dados do dossiê e conectando-os diretamente ao objetivo do usuário. Por exemplo, "Os outliers na coluna 'preço' são críticos porque seu objetivo é ter um modelo preciso...".
    5.  **Formato:** Apresente a resposta em Markdown como um "Plano de Ação de Pré-processamento", com títulos claros para cada recomendação.
    """

    try:
        response = model.generate_content(master_prompt)
        return response.text
    except Exception as e:
        return f"Ocorreu um erro ao chamar a API do LLM: {e}"


st.title("🧠 Consultor de Dados IA")
st.write("Um sistema híbrido que combina análise algorítmica com a inteligência de um LLM para criar seu plano de tratamento de dados.")

uploaded_file = st.file_uploader("Escolha um arquivo (.csv ou .xlsx)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl') if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        
        st.sidebar.header("🎯 Configuração do Modelo")
        target_column = st.sidebar.selectbox("1. Selecione sua variável alvo:", options=[None] + df.columns.tolist())
        user_context = st.sidebar.text_area("2. Descreva o objetivo do seu modelo:", placeholder="Ex: Prever o risco de inadimplência. O modelo precisa ser justo e explicável.")
        run_llm_analysis = st.sidebar.button("Gerar Plano de Ação com IA 🧠")

        st.divider()
        st.subheader("Visualização dos Dados Carregados")
        st.dataframe(df.head())

       
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
            st.info("Os seguintes pontos foram identificados e serão enviados ao especialista de IA para análise contextual:")
            st.json(achados_estruturados)
        
        if run_llm_analysis:
            if not LLM_OK:
                st.error("A API do LLM não está configurada.")
            elif not target_column or not user_context:
                st.error("Por favor, selecione a variável alvo e descreva o objetivo do modelo na barra lateral.")
            else:
                with st.spinner("🧠 O especialista de IA está analisando o dossiê e preparando seu plano de ação..."):
                    texto_dos_achados = formatar_achados_para_prompt(achados_estruturados)
                    plano_de_acao = gerar_plano_de_acao_com_llm(
                        target=target_column,
                        user_prompt=user_context,
                        findings_text=texto_dos_achados
                    )
                    st.subheader("✅ Plano de Ação de Pré-processamento", divider='rainbow')
                    st.markdown(plano_de_acao)

    except Exception as e:
        st.error(f"Ocorreu um erro geral: {e}")