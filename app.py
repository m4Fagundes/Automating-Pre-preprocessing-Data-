import streamlit as st
import pandas as pd
from thefuzz import process
import openpyxl
import numpy as np


def encontrar_inconsistencias_categoricas(series, threshold=85):
    """Encontra e agrupa valores categoricos similares."""
    unique_values = series.dropna().unique().tolist()
    if len(unique_values) < 2: return []
    processed_values = set()
    matches = []
    for val in unique_values:
        if val not in processed_values:
            similares = process.extract(val, [o for o in unique_values if o not in processed_values and o != val], limit=5)
            high_similarity = [s[0] for s in similares if s[1] > threshold]
            if high_similarity:
                grupo = tuple(sorted([val] + high_similarity))
                matches.append(grupo)
                processed_values.update(grupo)
    return matches

def encontrar_colunas_numericas_como_texto(df):
    """Identifica colunas de texto que poderiam ser numéricas."""
    colunas_problema = []
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].dropna().empty: continue
        try:
            numeric_series = pd.to_numeric(df[col].dropna(), errors='coerce')
            if numeric_series.notna().sum() / df[col].dropna().count() > 0.8:
                colunas_problema.append(col)
        except Exception: continue
    return colunas_problema

# NOVA ANÁLISE: Outliers
def encontrar_outliers_numericos(series):
    """Encontra outliers usando o método IQR."""
    if pd.api.types.is_numeric_dtype(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        outliers = series[(series < limite_inferior) | (series > limite_superior)]
        return outliers.tolist()
    return []

def encontrar_necessidade_de_scaling(df):
    """Verifica se features numéricas têm escalas muito diferentes."""
    numeric_cols = df.select_dtypes(include=np.number)
    if numeric_cols.shape[1] < 2: return None
    ranges = numeric_cols.max() - numeric_cols.min()
    if ranges.max() / ranges.min() > 100:
        return ranges.to_dict()
    return None

def encontrar_distribuicao_assimetrica(df, threshold=1.0):
    """Encontra features numéricas com alta assimetria (skewness)."""
    numeric_cols = df.select_dtypes(include=np.number)
    skewed_cols = numeric_cols.skew().filter(like='').loc[lambda x: abs(x) > threshold]
    return skewed_cols.to_dict()


st.set_page_config(layout="wide")
st.title("🤖 Assistente de Pré-processamento para Machine Learning")
st.write("Faça o upload do seu dataset (CSV ou Excel) para obter sugestões de tratamento para modelos de Regressão e Classificação.")

uploaded_file = st.file_uploader("Escolha um arquivo", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl') if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        st.success("Dataset carregado com sucesso!")
        st.dataframe(df.head())
        st.header("🔍 Análise de Pré-processamento para ML")

        
        with st.expander("1. Tratamento de Valores Nulos", expanded=True):
            nulos = df.isnull().sum()[lambda x: x > 0]
            if not nulos.empty:
                st.warning("Colunas com valores nulos foram encontradas!")
                st.dataframe(nulos.to_frame(name='Quantidade de Nulos'))
                st.markdown("""
                **Impacto em ML:** Muitos algoritmos (como Regressão Linear, SVM) não aceitam valores nulos e quebrarão durante o treinamento (`model.fit()`).
                **Sugestão:** Use técnicas de imputação (média, mediana para dados numéricos; moda para categóricos) ou, se a quantidade for pequena, remova as linhas.
                """)
            else: st.success("Nenhum valor nulo encontrado!")
        
        with st.expander("2. Padronização de Features Categóricas", expanded=True):
            colunas_texto = df.select_dtypes(include=['object']).columns
            alguma_inconsistencia = False
            for col in colunas_texto:
                grupos = encontrar_inconsistencias_categoricas(df[col])
                if grupos:
                    alguma_inconsistencia = True
                    st.warning(f"Inconsistências encontradas na feature **'{col}'**:")
                    for g in grupos: st.write(f"- Sugestão de Agrupamento: `{', '.join(g)}`")
            if alguma_inconsistencia:
                st.markdown("""
                **Impacto em ML:** O modelo tratará 'SP' e 'São Paulo' como duas cidades distintas, criando features desnecessárias (*feature explosion*) e piorando a performance. A padronização é essencial antes de aplicar *One-Hot Encoding*.
                **Sugestão:** Padronize os valores para um formato único.
                """)
            else: st.success("Nenhuma inconsistência categórica óbvia encontrada!")

        with st.expander("3. Conversão de Tipos de Dados", expanded=True):
            cols_converter = encontrar_colunas_numericas_como_texto(df)
            if cols_converter:
                st.warning("Features de texto que deveriam ser numéricas foram encontradas!")
                for col in cols_converter: st.write(f"- Feature **'{col}'**")
                st.markdown("""
                **Impacto em ML:** Modelos de regressão e classificação exigem *features* numéricas para funcionar. A conversão é um passo obrigatório.
                **Sugestão:** Converta essas colunas para um tipo numérico (inteiro ou float).
                """)
            else: st.success("Todos os tipos de dados parecem consistentes.")

        with st.expander("4. Análise de Outliers", expanded=True):
            algum_outlier = False
            for col in df.select_dtypes(include=np.number).columns:
                outliers = encontrar_outliers_numericos(df[col])
                if outliers:
                    algum_outlier = True
                    st.warning(f"Potenciais outliers encontrados na feature **'{col}'**:")
                    st.write(f"`{outliers[:5]}`" + ('...' if len(outliers) > 5 else ''))
            if algum_outlier:
                st.markdown("""
                **Impacto em ML:** Outliers podem distorcer a escala dos dados e impactar negativamente modelos sensíveis a eles, como Regressão Linear e algoritmos baseados em distância (KNN, SVM), "puxando" a linha de regressão ou a fronteira de decisão.
                **Sugestão:** Investigue os outliers. Se forem erros, corrija-os. Se não, considere removê-los ou usar algoritmos mais robustos a outliers (ex: RandomForest).
                """)
            else: st.success("Nenhum outlier óbvio detectado pelo método IQR.")

        with st.expander("5. Análise de Escala de Features (Feature Scaling)", expanded=True):
            ranges = encontrar_necessidade_de_scaling(df)
            if ranges:
                st.warning("Features numéricas com escalas muito diferentes foram detectadas!")
                st.json({k: f'Range: {v:.2f}' for k, v in ranges.items()})
                st.markdown("""
                **Impacto em ML:** Algoritmos como SVM, Regressão Logística, Redes Neurais e KNN são sensíveis à escala das features. Uma feature com escala maior (ex: salário) pode dominar o modelo, fazendo com que outras features (ex: idade) sejam ignoradas.
                **Sugestão:** Aplique técnicas de normalização (`MinMaxScaler`) ou padronização (`StandardScaler`) em todas as features numéricas.
                """)
            else: st.success("As escalas das features numéricas parecem estar consistentes.")

        with st.expander("6. Análise de Assimetria de Dados (Skewness)", expanded=True):
            skewed = encontrar_distribuicao_assimetrica(df)
            if skewed:
                st.warning("Features com distribuição de dados muito assimétrica foram encontradas!")
                st.json({k: f'Skew: {v:.2f}' for k, v in skewed.items()})
                st.markdown("""
                **Impacto em ML:** Alguns modelos, como a Regressão Linear, têm melhor performance quando as features (e os erros) seguem uma distribuição normal. Alta assimetria pode violar essa premissa.
                **Sugestão:** Considere aplicar transformações matemáticas (como logarítmica, `np.log1p`, ou Box-Cox) para normalizar a distribuição dos dados.
                """)
            else: st.success("As distribuições das features numéricas não apresentam alta assimetria.")


    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")