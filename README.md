# 🚀 Consultor de Dados IA

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)
![AI](https://img.shields.io/badge/AI-Google%20Gemini-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Uma ferramenta de IA híbrida que atua como seu co-piloto na preparação de dados para Machine Learning. O sistema combina a velocidade da análise algorítmica com o raciocínio contextual de um Large Language Model (LLM) para criar e aplicar um plano de tratamento de dados personalizado.

Para uma demonstração imediata, o assistente já vem com datasets clássicos do Kaggle (Titanic e Preços de Casas) prontos para análise!

---

### ✨ O Conceito: Inteligência Híbrida

1.  **Análise Algorítmica Rápida:** Algoritmos precisos escaneiam seu dataset para gerar um **"dossiê de evidências"** quantitativo sobre problemas como outliers, valores nulos e inconsistências.
2.  **Inteligência Contextual com LLM:** Um LLM (Google Gemini) atua como um **cientista de dados sênior**. Ele recebe o dossiê, analisa o seu objetivo de negócio e sua variável alvo, e então **prioriza inteligentemente** quais problemas são mais críticos, explicando o porquê e como resolvê-los em um plano de ação interativo.

---

### 🚀 Features Principais

- **Datasets de Exemplo Inclusos:** Comece a usar a ferramenta instantaneamente com os datasets clássicos do Kaggle.
- **Análise Algorítmica Abrangente:** Detecção rápida de nulos, outliers, inconsistências, tipos de dados, e mais.
- **Contextualização via Prompt:** Permite que você defina sua variável alvo e o objetivo do seu modelo para garantir recomendações relevantes.
- **Plano de Ação Interativo por IA:** Receba um plano priorizado e aplique as transformações sugeridas com um clique.
- **Pacote de Entrega Automatizado:** Baixe um arquivo `.zip` contendo os dados limpos (`.csv`) e um relatório completo (`.md`) das transformações aplicadas.

### 💻 Tecnologias Utilizadas

- **Python 3.9+**
- **Streamlit**
- **Pandas**
- **Google Generative AI (Gemini)**
- **Scikit-learn**, **thefuzz**

---

### ⚙️ Como Rodar o Projeto Localmente

Siga os passos abaixo para executar a aplicação no seu ambiente.

#### 1. (Obrigatório) Prepare os Datasets de Exemplo
Para que a funcionalidade de datasets de exemplo funcione, você precisa:
1.  Criar uma pasta chamada `sample_datasets` na raiz do projeto.
2.  Baixar os arquivos `train.csv` do **Titanic** e **House Prices** do Kaggle e salvá-los dentro desta pasta com os nomes `titanic_train.csv` e `house_prices_train.csv`.

#### 2. Instalação
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
*(Certifique-se de que seu `requirements.txt` está atualizado).*

#### 3. Configure a Chave de API
O projeto precisa de uma chave de API do Google Gemini:
1.  Crie uma pasta chamada `.streamlit` na raiz do seu projeto.
2.  Dentro dela, crie um arquivo chamado `secrets.toml`.
3.  Adicione sua chave de API ao arquivo:
    ```toml
    # .streamlit/secrets.toml
    GEMINI_API_KEY = "SUA_CHAVE_DE_API_AQUI"
    ```

#### 4. Execução
```bash
streamlit run app.py
```

---

### 📝 Como Usar a Ferramenta

1.  **Selecione uma Fonte de Dados:** Ao iniciar, escolha um dos datasets de exemplo no menu principal ou selecione a opção para fazer o upload do seu próprio arquivo.
2.  **Configure a Análise:** Na barra lateral, selecione sua **variável alvo** e **descreva o objetivo** do seu modelo.
3.  **Gere o Plano de Ação:** Clique no botão "Gerar Plano de Ação com IA 🧠".
4.  **Aplique as Ações:** Revise as recomendações priorizadas pela IA e clique nos botões "Aplicar" para executar as transformações.
5.  **Baixe o Pacote:** Após aplicar as ações, um botão de download aparecerá. Clique nele para baixar um arquivo `.zip` com o dataset limpo e um relatório detalhado.

---

### 👤 Autor

Feito por **Matheus Fagundes Araujo**.

### 📜 Licença

Este projeto está sob a licença MIT.
