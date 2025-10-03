# 🤖 Assistente de Pré-processamento para Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)

Uma ferramenta interativa construída com Streamlit que analisa seu dataset e sugere os passos de pré-processamento essenciais para otimizar modelos de Regressão e Classificação.

Este projeto foi desenvolvido como parte de um hackathon para acelerar o fluxo de trabalho de engenheiros de dados e cientistas de dados.

---

### Demo da Aplicação

> **Nota:** É altamente recomendável que você tire um print (screenshot) da sua aplicação funcionando e substitua o link abaixo. Isso torna o projeto muito mais atrativo!

![Demo da Aplicação](URL_DA_SUA_IMAGEM_AQUI.png)

---

### ✨ Features Principais

O assistente analisa seu dataset em busca dos seguintes problemas, fornecendo justificativas focadas no impacto em Machine Learning:

* **Análise de Valores Nulos:** Identifica dados faltantes que podem quebrar o treinamento de modelos.
* **Padronização de Features Categóricas:** Encontra e agrupa categorias inconsistentes (ex: "SP" vs "São Paulo") para evitar a criação de features desnecessárias.
* **Conversão de Tipos de Dados:** Detecta colunas numéricas armazenadas como texto, um passo obrigatório para a modelagem.
* **Detecção de Outliers:** Sinaliza valores extremos que podem distorcer modelos sensíveis a eles.
* **Análise de Escala (Feature Scaling):** Verifica se as features numéricas possuem escalas muito diferentes, o que pode prejudicar algoritmos baseados em distância.
* **Análise de Assimetria (Skewness):** Identifica distribuições de dados muito assimétricas que podem ser normalizadas para melhorar a performance do modelo.

### 🚀 Tecnologias Utilizadas

* **Python 3.9+**
* **Streamlit:** Para a criação da interface web interativa.
* **Pandas:** Para a manipulação e análise dos dados.
* **thefuzz:** Para a análise de similaridade de strings.

---

### ⚙️ Como Rodar o Projeto Localmente

Siga os passos abaixo para executar a aplicação no seu ambiente.

#### Pré-requisitos

* [Python 3.9](https://www.python.org/downloads/) ou superior
* [Git](https://git-scm.com/downloads)

#### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git](https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git)
    cd SEU-REPOSITORIO
    ```

2.  **Crie e ative um ambiente virtual:**
    * **macOS / Linux:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    * **Windows:**
        ```bash
        python -m venv .venv
        .venv\Scripts\activate
        ```

3.  **Instale as dependências:**
    (Certifique-se de que você tem um arquivo `requirements.txt` com as bibliotecas. Se não tiver, crie-o e adicione `streamlit`, `pandas`, `thefuzz`, `python-levenshtein`, `openpyxl`).
    ```bash
    python3 -m pip install -r requirements.txt
    ```

4.  **Execute a aplicação:**
    ```bash
    streamlit run app.py
    ```

A aplicação estará disponível no seu navegador no endereço `http://localhost:8501`.

### 📝 Como Usar

1.  Acesse a aplicação no seu navegador.
2.  Clique no botão "Browse files" e faça o upload de um arquivo de dados (`.csv` ou `.xlsx`).
3.  Aguarde a análise ser concluída.
4.  Navegue pelas seções de análise para ver os problemas encontrados e as sugestões de tratamento com justificativas focadas em Machine Learning.

### 🔮 Próximos Passos (Roadmap)

- [ ] Gerar automaticamente o código Python (com Scikit-Learn) para aplicar as correções sugeridas.
- [ ] Adicionar suporte para conexão direta com bancos de dados.
- [ ] Implementar mais análises (ex: detecção de dados sensíveis - PII).
- [ ] Permitir o download de um relatório de "saúde dos dados".

---

### 👤 Autor

Feito com ❤️ por **[Seu Nome Aqui]**.

### 📜 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.