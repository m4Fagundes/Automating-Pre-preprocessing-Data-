# 🧠 Consultor de Dados IA: Um Assistente Híbrido para Pré-processamento de ML

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)
![AI](https://img.shields.io/badge/AI-Google%20Gemini-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Este não é apenas um validador de dados. É um **consultor de IA híbrido** que combina a velocidade da análise algorítmica com o raciocínio contextual de um Large Language Model (LLM) para criar um plano de ação de pré-processamento sob medida para o seu projeto de Machine Learning.

---

### ✨ O Conceito: Inteligência Híbrida

Nossa ferramenta opera em duas fases para oferecer o melhor dos dois mundos:

1.  **Análise Algorítmica Rápida:** Primeiro, algoritmos estatísticos e determinísticos escaneiam seu dataset em segundos para gerar um **"dossiê de evidências"** — um relatório quantitativo e preciso sobre problemas como outliers, valores nulos, inconsistências e mais.
2.  **Inteligência Contextual com LLM:** Em seguida, o LLM (Google Gemini) atua como um **cientista de dados sênior**. Ele recebe o dossiê, analisa o seu objetivo de negócio (descrito por você) e a sua variável alvo, e então **prioriza inteligentemente** quais problemas são mais críticos, explicando o porquê e como resolvê-los.

### 📸 Demo da Aplicação

> **Nota:** Um bom screenshot é fundamental! Tente capturar a tela mostrando o "Plano de Ação" gerado pela IA.

![Demo da Aplicação](URL_DA_SUA_IMAGEM_AQUI.png)

---

### 🚀 Features Principais

* **Análise Algorítmica Abrangente:** Detecção rápida de nulos, outliers (IQR), inconsistências categóricas, tipos de dados, escala e assimetria.
* **Contextualização via Prompt:** Permite que você defina sua variável alvo e o objetivo do seu modelo, garantindo que as recomendações sejam relevantes.
* **Geração de Plano de Ação por IA:** O LLM atua como um consultor, analisando as evidências e gerando um plano de ação priorizado, com justificativas claras e focadas no seu objetivo.
* **Interface Interativa:** Construído com Streamlit para uma experiência de usuário fluida e intuitiva.

### 💻 Tecnologias Utilizadas

* **Python 3.9+**
* **Streamlit:** Para a interface web.
* **Pandas:** Para a manipulação dos dados.
* **google-generativeai:** Para a integração com o LLM Gemini.
* **thefuzz:** Para a análise de similaridade de strings.

---

### ⚙️ Como Rodar o Projeto Localmente

Siga os passos abaixo para executar a aplicação no seu ambiente.

#### Pré-requisitos

* [Python 3.9](https://www.python.org/downloads/) ou superior
* [Git](https://git-scm.com/downloads)

#### 1. Instalação

```bash
# Clone o repositório
git clone [https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git](https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git)
cd SEU-REPOSITORIO

# Crie e ative um ambiente virtual
python3 -m venv .venv
source .venv/bin/activate
# No Windows: .venv\Scripts\activate

# Instale as dependências
python3 -m pip install -r requirements.txt
```
*(Certifique-se de que seu `requirements.txt` contém `streamlit`, `pandas`, `thefuzz`, `python-levenshtein`, `openpyxl` e `google-generativeai`)*

#### 2. Configurar a Chave de API

Este projeto precisa de uma chave de API do Google Gemini para funcionar.

1.  Crie uma pasta chamada `.streamlit` na raiz do seu projeto.
2.  Dentro dela, crie um arquivo chamado `secrets.toml`.
3.  Adicione sua chave de API ao arquivo da seguinte forma:
    ```toml
    # .streamlit/secrets.toml
    GEMINI_API_KEY = "SUA_CHAVE_DE_API_AQUI"
    ```

#### 3. Execução

```bash
streamlit run app.py
```
A aplicação estará disponível no seu navegador no endereço `http://localhost:8501`.

### 📝 Como Usar

1.  Acesse a aplicação no seu navegador.
2.  Faça o upload de um arquivo de dados (`.csv` ou `.xlsx`).
3.  Na barra lateral, **selecione sua variável alvo** e **descreva o objetivo do seu modelo**.
4.  Clique no botão **"Gerar Plano de Ação com IA 🧠"**.
5.  Analise as recomendações priorizadas e justificadas fornecidas pelo consultor de IA.

---

### 🔮 Próximos Passos (Roadmap)

- [ ] Gerar automaticamente o código Python (com Scikit-Learn) para aplicar as correções sugeridas.
- [ ] Adicionar suporte para conexão direta com bancos de dados.
- [ ] Permitir que o usuário ajuste os parâmetros dos algoritmos (ex: threshold de outlier).

---

### 👤 Autor

Feito com ❤️ por **Matheus Fagundes Araujo**.

### 📜 Licença

Este projeto está sob a licença MIT.