import streamlit as st
import polars as pl
import requests
import pandas as pd
import plotly.graph_objects as go
import time
import os

# --- Configurações da Página ---
st.set_page_config(
    page_title="Monitoramento de Turbinas NASA",
    page_icon="✈️",
    layout="wide"
)

# Garante diretório correto
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# URL Inteligente (Funciona no Docker e no Local)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# --- Funções Auxiliares ---
def load_and_process_test_data():
    """
    Carrega os dados de teste e aplica a MESMA engenharia de features do treino.
    ATUALIZADO: Agora inclui Rolling Means + TENDÊNCIA (Diff).
    """
    cols = ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"] + [f"s_{i}" for i in range(1, 22)]
    
    try:
        # Tenta ler com tratamento de erro
        try:
            df = pl.read_csv("test_FD001.txt", separator=" ", has_header=False, new_columns=cols, null_values=[""], truncate_ragged_lines=True)
        except:
            df = pl.read_csv("test_FD001.txt", separator=" ", has_header=False, new_columns=cols, null_values=[""], ignore_errors=True)
            
        # Remove coluna fantasma se existir
        df = df.select(cols)
    except Exception as e:
        st.error(f"Erro ao ler test_FD001.txt: {e}")
        return None

    # Cast para float
    sensores_chave = ["s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"]
    df = df.with_columns([pl.col(c).cast(pl.Float64) for c in sensores_chave])

    # --- AQUI ESTÁ A ATUALIZAÇÃO MÁGICA ---
    ops = []
    for col in sensores_chave:
        # 1. Média Móvel
        ops.append(pl.col(col).rolling_mean(window_size=5).over("unit_nr").alias(f"{col}_rolling_5"))
        ops.append(pl.col(col).rolling_mean(window_size=10).over("unit_nr").alias(f"{col}_rolling_10"))
        
        # 2. Tendência (Diff) - O modelo novo exige isso!
        ops.append(pl.col(col).diff().over("unit_nr").alias(f"{col}_diff"))
    
    df_feat = df.with_columns(ops)
    
    # Preencher os nulos iniciais (do rolling/diff) com backward fill para não quebrar a simulação
    df_feat = df_feat.fill_null(strategy="backward")
    
    # Preenche qualquer nulo restante com 0
    df_feat = df_feat.fill_null(0)
    
    return df_feat

# --- Interface do Streamlit ---
st.title("✈️ Sistema de Manutenção Preditiva em Tempo Real")
st.markdown(f"""
**Status do Sistema:** Conectado à API ({API_URL}).  
Monitorando Turbinas da Frota FD001.
""")

# Sidebar
st.sidebar.header("Painel de Controle")
turbina_id = st.sidebar.number_input("ID da Turbina", min_value=1, max_value=100, value=1)
speed = st.sidebar.slider("Velocidade (segundos)", 0.1, 2.0, 0.2)
start_btn = st.sidebar.button("▶️ INICIAR MONITORAMENTO", type="primary")

# Layout
col1, col2, col3 = st.columns(3)
metric_ciclo = col1.empty()
metric_rul = col2.empty()
metric_status = col3.empty()
chart_placeholder = st.empty()

if start_btn:
    with st.spinner("Calibrando sensores..."):
        df_all = load_and_process_test_data()
    
    if df_all is not None:
        df_turbina = df_all.filter(pl.col("unit_nr") == turbina_id)
        dados_pandas = df_turbina.to_pandas()
        
        st.success(f"Conexão estabelecida com Turbina #{turbina_id}")
        
        history_rul = []
        history_cycles = []
        
        progress_bar = st.progress(0)
        
        for index, row in dados_pandas.iterrows():
            payload = {"data": row.to_dict()}
            
            try:
                response = requests.post(API_URL, json=payload, timeout=2)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Verifica se a API retornou erro interno
                    if "erro" in result:
                        st.error(f"Erro interno da API: {result['erro']}")
                        break
                        
                    pred_rul = result["rul_predito"]
                    status_text = result["status"]
                    
                    # Atualiza Métricas Visualmente
                    metric_ciclo.metric("Ciclo de Voo", int(row["time_cycles"]))
                    metric_rul.metric("RUL Estimado", f"{pred_rul:.1f} ciclos")
                    
                    if "PERIGO" in status_text:
                        metric_status.error(status_text)
                    elif "Alerta" in status_text:
                        metric_status.warning(status_text)
                    else:
                        metric_status.success(status_text)
                    
                    # Gráfico
                    history_cycles.append(row["time_cycles"])
                    history_rul.append(pred_rul)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=history_cycles, y=history_rul,
                        mode='lines', name='Previsão',
                        line=dict(color='#00CC96', width=3),
                        fill='tozeroy'
                    ))
                    
                    fig.add_hline(y=20, line_dash="dot", line_color="red", annotation_text="Falha Iminente")
                    
                    fig.update_layout(
                        xaxis_title="Ciclos", yaxis_title="Vida Útil Restante",
                        height=350, margin=dict(l=0, r=0, t=30, b=0)
                    )
                    
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"Erro HTTP {response.status_code}: {response.text}")
                    break
                    
            except Exception as e:
                st.error(f"Perda de conexão com a API: {e}")
                break
            
            time.sleep(speed)
            
        st.info("Fim da transmissão de dados desta turbina.")