import os
import gc
import copy
import random as rn
from collections import Counter

import cv2
import numpy as np
import tensorflow as tf
from pyts.image import RecurrencePlot, GramianAngularField

# Constantes fundamentais dos experimentos
SEED = 33
FRACAO_TESTE = 0.25
EPOCAS = 100
TAMANHO_LOTE = 32
VERBOSIDADE = 2

# Parametros RP (verificado empiricamente)
PARAMETROS_RP = {
    "dimension": 1,
    "time_delay": 1,
    "threshold": None,
    "percentage": 10
}

TAMANHO_IMAGEM = (32,32,1) # Apenas 1 canal
TAMANHO_IMAGEM_RP = (32,32)
TAMANHO_IMAGEM_DLAFE = (224,224,3) # Apenas 1 canal
TIPO_DADOS = np.float32

# Garantindo reprodutibilidade
# Travar Seed's
np.random.seed(SEED)
rn.seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)

def mlp(
    input_shape=TAMANHO_IMAGEM_DLAFE,
    metrics=[
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ], 
    output_bias=None
):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
        
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, activation='relu', input_shape=input_shape),
      # tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
    ])

    model.compile(
      optimizer=tf.keras.optimizers.Adam(lr=0.001),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=metrics
    )

    return model

def class_weight(y, debug=False):
    
    # Classes distribution
    neg, pos = np.bincount(y)
    total = neg + pos

    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    w_0 = (1 / neg)*(total)/2.0 
    w_1 = (1 / pos)*(total)/2.0

    class_weight = {0: w_0, 1: w_1}

    print('Weight for class 0: {:.2f}'.format(w_0))
    print('Weight for class 1: {:.2f}'.format(w_1))
    
    return class_weight

def pos_weight(y):
    try:
        counter = Counter(y)
        return counter[0]/counter[1]
    except:
        return 1

def carregar_dados_aparelho(janelas, instancia, aparelho, taxa, tamanho_janela, 
    split_teste=None, eliminar_janelas_vazias=False, debug=False):

    # Extrair series divididas em janelas para cada medidor
    janelas.preparar(
        taxa_amostral=taxa, 
        intervalo_medicao=tamanho_janela,
        # debug=debug
    )
    print()

    # Pprearando dados (Serie / Estado)
    # X
    dados_medidores = janelas.filtrar(filtros=janelas.listar_medidores())
    
    dados_aparelho = janelas.filtrar(filtros=[(instancia, aparelho)])[0]
    
    # Validar tamanho dos dados de medidores (podem ter mais registros que os aparelhos)
    janela_media_medidores = int(np.sum([len(d["janelas"])for d in dados_medidores])/len(dados_medidores))
    janela_media_aparelho = len(dados_aparelho["janelas"])#int(np.sum([len(d["janelas"])for d in dados_aparelho])/len(dados_aparelho))

    # Ajustando para medidores terem o mesmo shape de janelas dos aparelhos 
    if janela_media_medidores > janela_media_aparelho:
        diferenca = janela_media_medidores-janela_media_aparelho
        #if debug: print("  -> Diferenca encontrada entre medidores/aparelhos:", diferenca, ", ajustando..")
        for i in range(len(dados_medidores)):
            removidos = 0
            while removidos < diferenca:
                # Remover ultima janela
                dados_medidores[i]["janelas"] = dados_medidores[i]["janelas"][:-1,:]
                removidos += 1
    
    # Estruturando dados modelagem (X e y)
    try:
        X = sum([dm["janelas"] for dm in dados_medidores])
    except Exception as e:
        print("Erro ao combinar medidores:", str(e))
        X = dados_medidores[0]["janelas"]

    # Selecionando apenas janelas VALIDAS (ocorrencia de ao menos 1 carga)
    # TODO: Implementar na biblioteca esta rotina de validacao
    if eliminar_janelas_vazias:
        idx_janelas_validas = np.where(np.sum(X, axis=1)>0)[0]
        X = X[idx_janelas_validas]
        #for i in range(len(dados_aparelhos)):
        dados_aparelho["janelas"] = dados_aparelho["janelas"][idx_janelas_validas]
        rotulos = copy.deepcopy(dados_aparelho["rotulos"])
        dados_aparelho["rotulos"]["estado"] = rotulos["estado"][idx_janelas_validas]
        dados_aparelho["rotulos"]["media"]  = rotulos["media"][idx_janelas_validas]
        dados_aparelho["rotulos"]["total"]  = rotulos["total"][idx_janelas_validas]
        if debug:
            print("   - `{}-{}`: {} => {}".format(
                dados_aparelho["carga"].upper(), 
                dados_aparelho["instancia"],
                Counter(rotulos["estado"]),
                Counter(dados_aparelho["rotulos"]["estado"])
            ))

    # y
    y = dados_aparelho["rotulos"]["estado"]

    # <<< Limpando memoria >>>
    dados_cargas = None
    del dados_cargas
    dados_medidores = None
    del dados_medidores
    dados_aparelho = None
    del dados_aparelho
    gc.collect()
    # <<< Limpando memoria >>>

    # Fazendo split dos dados (treino/teste)
    if split_teste is None:
        return X, y
    else:
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y, 
            test_size=split_teste,
            stratify=y,
            random_state=SEED
        )
        print()

        return X_treino, X_teste, y_treino, y_teste        
        


def instancia_aparelho_residencia(aparelho, residencia, base = None):
    """Função para coletar o id/instancia do aparelho na residencia,
    permitindo executar os testes independente da residencia"""
    instancia = []
    #for e in base.buildings[residencia].elec.all_meters():
    for e_i in range(1, len(base.buildings[residencia].elec.all_meters())):

        # Selecionando canal/aparelho
        e = base.buildings[residencia].elec[e_i]
        
        if not hasattr(e,'meters'):
            if e.label().lower().replace(" ","_") == aparelho:
                instancia.append( e.instance() )
        else:
            for e_ in e.meters:
                if e_.label().lower().replace(" ","_") == aparelho:
                    instancia.append( e_.instance() )
    return instancia

def serie_para_imagem(serie, params_rp = PARAMETROS_RP, tam_imagem=TAMANHO_IMAGEM_RP, 
                      normalizar=False, padronizar=False, tipo_dados=TIPO_DADOS):
    """
    Funcao responsavel por gerar e tratar a imagem RP (baseado estudo #17).
    """
    # Gerando imagem RP/redimensiona_prndo
    imagem = RecurrencePlot(**params_rp).fit_transform([serie])[0]
    imagem = cv2.resize(
            imagem, 
            dsize=tam_imagem[:2], 
            interpolation=cv2.INTER_CUBIC
        ).astype(tipo_dados)
    
    if np.sum(imagem) > 0:
        # Normalizar
        if normalizar:
                imagem = (imagem - imagem.min()) / (imagem.max() - imagem.min()) # MinMax (0,1)
            #imagem = (imagem - imagem.mean()) / np.max([imagem.std(), 1e-4])

    #     # centralizar
    #     if centralizar:
    #         imagem -= imagem.mean()

        # Padronizar
        elif padronizar:
            imagem = (imagem - imagem.mean())/imagem.std()#tf.image.per_image_standardization(imagem).numpy()

    # N canais
    imagem = np.stack([imagem for i in range(tam_imagem[-1])],axis=-1).astype(tipo_dados)     
    
    return imagem

def preparar_amostras(
    X, y, 
    params_rp=PARAMETROS_RP, 
    tam_imagem=TAMANHO_IMAGEM_RP, 
    normalizar=False, padronizar=False
    ):
    
    X_imagem = np.empty((len(X), *tam_imagem))
    for i, x in enumerate(X):
        X_imagem[i,] = serie_para_imagem(
            x, 
            params_rp=params_rp, 
            tam_imagem=tam_imagem,
            normalizar=normalizar,
            padronizar=padronizar,
        )
    return X_imagem, y


def centralizar_dados(X):
    return np.array([x - x.mean() for x in X], dtype=TIPO_DADOS)

def normalizar_dados(X):
    X_ = np.empty(np.asarray(X).shape)
    for i, x in enumerate(X):
        if len(np.unique(x))>1:
            X_[i] = (x - x.min()) / (x.max() - x.min())
        elif x.max()>0:
            X_[i] = x / x.max()
        else:
            X_[i] = x
    return X_.astype(TIPO_DADOS)

def padronizar_dados(X):
    """
    Calcular z-score por amostra.
    Ref.: https://datascience.stackexchange.com/questions/16034/dtw-dynamic-time-warping-requires-prior-normalization    
    """
    from scipy import stats
    
    return np.array([stats.zscore(x) for x in X], dtype=TIPO_DADOS)


def preparar_amostra_tfdata(amostra, rotulo):
    """
    Preparação da amostra/rótulo para o modelo.
    """
    # Convertendo serie para imagem
    amostra = tf.numpy_function(serie_para_imagem, [amostra], TIPO_DADOS)
    amostra = tf.reshape(amostra, TAMANHO_IMAGEM)
    return amostra, rotulo