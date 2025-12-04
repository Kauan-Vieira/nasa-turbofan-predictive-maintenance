import polars as pl
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import os

# --- Garante o diretório correto ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def feature_engineering(df):
    """
    Versão Avançada: Médias Móveis + Tendência (Diff)
    """
    # Sensores chave
    sensores_chave = ["s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"]
    
    # Casting para float
    df = df.with_columns([pl.col(c).cast(pl.Float64) for c in sensores_chave])

    # Feature Engineering
    ops = []
    for col in sensores_chave:
        # 1. Média Móvel (Suaviza o ruído)
        ops.append(pl.col(col).rolling_mean(window_size=5).over("unit_nr").alias(f"{col}_rolling_5"))
        ops.append(pl.col(col).rolling_mean(window_size=10).over("unit_nr").alias(f"{col}_rolling_10"))
        
        # 2. Tendência/Derivada (Velocidade da mudança) - NOVO!
        # Isso mostra para o modelo se a temperatura está subindo RÁPIDO
        ops.append(pl.col(col).diff().over("unit_nr").alias(f"{col}_diff"))
    
    df_feat = df.with_columns(ops)
    
    # Remove linhas nulas geradas pelo rolling/diff
    # (Usa a coluna de rolling_10 como referência)
    df_feat = df_feat.drop_nulls(subset=[f"{sensores_chave[0]}_rolling_10"])
    
    return df_feat, sensores_chave

def train():
    print("🚀 Iniciando Pipeline de Treinamento v3.0...")
    
    # 1. Carregar Dados
    cols = ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"] + [f"s_{i}" for i in range(1, 22)]
    
    try:
        df = pl.read_csv("train_FD001.txt", separator=" ", has_header=False, new_columns=cols, null_values=[""], truncate_ragged_lines=True)
    except:
        df = pl.read_csv("train_FD001.txt", separator=" ", has_header=False, new_columns=cols, null_values=[""], ignore_errors=True)

    df = df.select(cols)
    print(f"   Dados brutos carregados: {df.shape}")

    # 2. Criar Target (RUL) com CLIPPING (O Segredo!)
    # Define um teto. Se RUL > 125, vira 125.
    MAX_RUL = 125
    
    df = df.with_columns([
        (pl.col("time_cycles").max().over("unit_nr") - pl.col("time_cycles")).alias("RUL_raw")
    ])
    
    df = df.with_columns([
        pl.when(pl.col("RUL_raw") > MAX_RUL)
        .then(MAX_RUL)
        .otherwise(pl.col("RUL_raw"))
        .alias("RUL")
    ])
    
    print("🛠️ Gerando Features Avançadas (Rolling + Diff)...")
    # AQUI ESTAVA O ERRO: Esta linha precisa existir!
    df_proc, sensores_base = feature_engineering(df)
    
    # Define features
    features_cols = [c for c in df_proc.columns if "s_" in c] 
    target_col = "RUL"
    
    # 3. Split Treino/Validação
    df_train = df_proc.filter(pl.col("unit_nr") <= 80)
    df_val = df_proc.filter(pl.col("unit_nr") > 80)
    
    X_train = df_train.select(features_cols).to_pandas()
    y_train = df_train.select(target_col).to_pandas()
    X_val = df_val.select(features_cols).to_pandas()
    y_val = df_val.select(target_col).to_pandas()
    
    print(f"📊 Dados de Treino: {X_train.shape} | Validação: {X_val.shape}")
    
    if X_train.shape[0] == 0:
        raise ValueError("ERRO: Treino vazio.")

    # 4. Treinar Modelo (XGBoost Otimizado)
    print("🔥 Treinando XGBoost Turbinado...")
    model = xgb.XGBRegressor(
        n_estimators=500,       # Mais árvores
        learning_rate=0.05,     # Aprende mais devagar e com mais detalhe
        max_depth=6,            # Um pouco mais complexo
        objective='reg:squarederror',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 5. Avaliar
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"✅ Treino Finalizado! RMSE na Validação: {rmse:.2f} ciclos")
    
    # 6. Salvar
    artifacts = {
        "model": model,
        "features": features_cols,
        "sensores_base": sensores_base
    }
    
    with open("model_v1.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    
    print("💾 Modelo salvo como 'model_v1.pkl'.")

if __name__ == "__main__":
    train()