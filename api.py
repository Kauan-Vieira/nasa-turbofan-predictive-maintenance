from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os
import uvicorn

# 1. Configuração Inicial
app = FastAPI(title="Turbine Failure Prediction API", version="1.0")

# Garante o diretório correto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_v1.pkl")

# 2. Carregar o Modelo na Memória (apenas uma vez, na inicialização)
print(f"🔄 Carregando modelo de: {MODEL_PATH}")
try:
    with open(MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)
        model = artifacts["model"]
        features_esperadas = artifacts["features"]
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ CRÍTICO: Não foi possível carregar o modelo. Erro: {e}")
    raise e

# 3. Definir o formato dos dados de entrada
# O modelo espera um dicionário com os valores das features
class SensorData(BaseModel):
    data: dict

@app.get("/")
def home():
    return {"status": "API Online", "model_version": "v1"}

@app.post("/predict")
def predict(input_data: SensorData):
    try:
        # Converte o dicionário recebido para DataFrame (que o modelo entende)
        payload = input_data.data
        df_input = pd.DataFrame([payload])
        
        # Garante que as colunas estão na ordem exata que o modelo aprendeu
        # (Isso evita erros se o JSON vier bagunçado)
        try:
            df_input = df_input[features_esperadas]
        except KeyError as e:
            missing = set(features_esperadas) - set(df_input.columns)
            raise HTTPException(status_code=400, detail=f"Faltam features no envio: {missing}")

        # Faz a predição
        prediction = model.predict(df_input)[0]
        
        # Lógica de Negócio: Status do Risco
        status = "Normal"
        if prediction < 50:
            status = "Alerta: Manutenção Próxima"
        if prediction < 20:
            status = "PERIGO: FALHA IMINENTE"

        return {
            "rul_predito": float(prediction),
            "status": status,
            "detalhes": "Predição realizada com sucesso"
        }

    except Exception as e:
        return {"erro": str(e)}

# Bloco para rodar direto pelo Python (opcional, mas útil para debug)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)