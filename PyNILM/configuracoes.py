import os

# Controle de aleatoriedade
SEED = 33

# Controle de tam. de conjunto de teste 
FRACAO_TESTE = 0.25

# Numero de folds (validacao cruzada)
K_FOLDS = 10

# Metricas de avaliacao (sklearn)
SCORE_METRICS = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Constantes do dominio
TAXA_AMOSTRAGEM = 3 # Frequencia do sinal (LF)

# Window size (5 minutes, after converted to seconds and divided by sample rate), 
# resulting in unit split of each chunk (window)
TAMANHO_JANELA = int((1.5 * 60) / TAXA_AMOSTRAGEM) # (minutes * 60) / sample rate = 1 hora de amostras