import pandas as pd

# Carrega o arquivo (o sep=r'\s+' lida com os múltiplos espaços entre as colunas)
df = pd.read_csv('training_data.txt', sep=r'\s+')

# Descarta as primeiras 14 linhas (ignorando a fase de aquecimento)
df_clean = df.iloc[14:].copy()

# Sobrescreve a coluna 'step' com uma nova sequência começando de 1
df_clean['step'] = range(1, len(df_clean) + 1)

# Reseta o índice interno da tabela
df_clean.reset_index(drop=True, inplace=True)

# # Salva o resultado em um novo arquivo pronto para o treinamento
# df_clean.to_csv('training_data.txt', sep=' ', index=False)