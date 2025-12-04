# Usa a versão ESTÁVEL (Bookworm) para evitar erros de pacote não encontrado
FROM python:3.11-slim-bookworm

# Define o diretório de trabalho
WORKDIR /app

# Instala apenas o essencial (compiladores C++ para o XGBoost/Polars)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala as bibliotecas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código
COPY . .

# Expõe as portas
EXPOSE 8000
EXPOSE 8501

# Comando padrão
CMD ["python", "api.py"]