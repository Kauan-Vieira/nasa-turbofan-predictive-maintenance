import polars as pl

# Nomes das colunas padrão do dataset NASA C-MAPSS
col_names = ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"] + [f"s_{i}" for i in range(1, 22)]

def verify_data():
    try:
        # Tenta carregar o treino (ajuste o caminho se necessário)
        # O separador costuma ser espaço " " e não tem header
        df_train = pl.read_csv("train_FD001.txt", separator=" ", has_header=False, new_columns=col_names, null_values=[""])
        
        # Remove colunas vazias que as vezes aparecem no final por causa de espaços duplos
        df_train = df_train.select(col_names)

        print("✅ Arquivo de TREINO carregado com sucesso!")
        print(f"   Shape: {df_train.shape} (Deve ser aprox 20k linhas x 26 colunas)")
        print(f"   Exemplo de Sensores: {df_train.select(['s_2', 's_14']).head(2)}")
        
    except Exception as e:
        print(f"❌ Erro ao ler train_FD001.txt: {e}")

    try:
        # Tenta carregar o RUL (o do seu print)
        df_rul = pl.read_csv("RUL_FD001.txt", has_header=False, new_columns=["RUL_Real"])
        print("\n✅ Arquivo de RUL (Gabarito) carregado!")
        print(f"   Shape: {df_rul.shape} (Deve ter 100 linhas - uma pra cada turbina de teste)")
        print(f"   Primeiros valores: {df_rul.head(3).to_series().to_list()}") # Deve bater com 112, 98, 69...
        
    except Exception as e:
        print(f"❌ Erro ao ler RUL_FD001.txt: {e}")

if __name__ == "__main__":
    verify_data()