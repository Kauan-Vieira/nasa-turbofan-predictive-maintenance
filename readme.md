# ✈️ NASA Turbofan Predictive Maintenance System (End-to-End)

Um sistema completo de Manutenção Preditiva (PdM) capaz de estimar a Vida Útil Restante (RUL) de turbinas de avião em tempo real, utilizando arquitetura moderna de microsserviços.



## 🧠 O Problema
Falhas inesperadas em turbinas aeronáuticas geram custos milionários e riscos críticos de segurança. O objetivo deste projeto é antecipar falhas (Manutenção Preditiva) analisando dados de sensores em tempo real.

O sistema processa dados brutos do dataset **NASA C-MAPSS**, cria features complexas (médias móveis e tendências) e utiliza um modelo de Machine Learning para prever exatamente quantos ciclos de voo restam antes da quebra.

## 🛠️ Tech Stack & Arquitetura
O projeto foi construído seguindo as melhores práticas de MLOps e Engenharia de Dados:

* **Linguagem:** Python 3.11
* **ETL & Engine de Dados:** Polars (Alta performance para manipulação de dados)
* **Machine Learning:** XGBoost Regressor (Otimizado com RMSE < 20 ciclos)
* **Backend / API:** FastAPI (Servindo o modelo via REST)
* **Frontend / Dashboard:** Streamlit (Visualização em tempo real)
* **Infraestrutura:** Docker & Docker Compose (Containerização completa)

## 📊 Performance do Modelo
O modelo final alcançou uma performance de nível competitivo:
* **RMSE (Erro Médio):** ~19.67 ciclos
* **Técnicas Usadas:** RUL Clipping (Piecewise Linear Regression), Feature Engineering com Rolling Windows e Cálculo de Derivadas (Tendência).

---

## 🐳 Como Rodar (Via Docker - Recomendado)
A maneira mais simples e robusta de executar este projeto. Garante que todo o ambiente (API, Dashboard e Dependências) funcione perfeitamente isolado.

### Pré-requisitos
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e rodando.

### Passo a Passo
1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/SEU_USUARIO/nasa-turbofan-predictive-maintenance.git](https://github.com/SEU_USUARIO/nasa-turbofan-predictive-maintenance.git)
    cd nasa-turbofan-predictive-maintenance
    ```

2.  **Suba a aplicação com um comando:**
    ```bash
    docker-compose up --build
    ```
    *Isso irá construir as imagens, instalar as dependências e iniciar os serviços.*

3.  **Acesse no navegador:**
    * ✈️ **Dashboard:** http://localhost:8501
    * 📡 **Documentação da API:** http://localhost:8000/docs

4.  **Para parar:**
    * Pressione `Ctrl+C` no terminal.
    * Para remover os containers: `docker-compose down`

---

## 💻 Como Rodar (Manualmente / Local)
Caso prefira rodar sem Docker, você precisará de Python 3.11 instalado.

1.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Treine o modelo (Gera o arquivo `model_v1.pkl`):**
    ```bash
    python train_model.py
    ```

3.  **Inicie a API (Terminal 1):**
    ```bash
    python api.py
    ```

4.  **Inicie o Dashboard (Terminal 2):**
    ```bash
    streamlit run dashboard.py
    ```

## 📂 Estrutura do Projeto
```text
├── 🐳 Dockerfile            # Receita da imagem Docker
├── 🐳 docker-compose.yaml   # Orquestração dos serviços (API + Dash)
├── 📜 requirements.txt      # Lista de bibliotecas
├── 🧠 train_model.py        # Pipeline de ETL e Treinamento (Polars + XGBoost)
├── 📡 api.py                # Servidor Backend (FastAPI)
├── 📊 dashboard.py          # Frontend Interativo (Streamlit)
├── 💾 model_v1.pkl          # Modelo treinado serializado
└── 📄 README.md             # Documentação