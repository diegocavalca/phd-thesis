import os

from matplotlib import pyplot as plt

plt.style.use('ggplot')
from tqdm import tqdm_notebook
from collections import Counter
import seaborn as sns
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from scipy import interpolate
import numpy as np
import pandas as pd


# from .metricas import *

def avaliar_threshold(pos_probs, threshold):
    """Verificar se a probabilidade é igual ou maior que um limiar para classe positiva"""
    return (pos_probs >= threshold).astype('int')


def plotar_previsto_esperado(test, predicted):
    # import matplotlib.pyplot as plt
    plt.plot(predicted.flatten(), label='pred')
    plt.plot(test.flatten(), label='Y')
    plt.show();
    return


def plotar_matriz_confusao(y_teste, y_pred, labels, caminho_persistencia=None):
    matriz = confusion_matrix(y_teste, y_pred)

    sns.set(color_codes=True)
    fig = plt.figure(1, figsize=(9, 6))

    plt.title("Matriz de Confusão")

    sns.set(font_scale=1.3)
    ax = sns.heatmap(matriz, annot=True, cmap='Reds', fmt='g')

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set(ylabel="Teste", xlabel="Previsto")

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    if caminho_persistencia:
        fig.savefig(os.path.join(caminho_persistencia, 'matriz_confusao.png'))
        plt.close(fig)
    else:
        plt.show()

def plotar_curva_roc(modelo, X_teste, y_teste, caminho_persistencia=None):
    probas = modelo.predict_proba(X_teste)
    tx_fp, tx_vp, threshold = roc_curve(y_teste, probas)
    roc_auc = auc(tx_fp, tx_vp)

    fig = plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--')  # , label='Modelo Descalibrado')
    plt.plot(tx_fp, tx_vp, marker='.', label='AUC = %0.2f' % roc_auc)

    plt.xlabel('Taxa Falso Positivo (Especificidade)')
    plt.ylabel('Taxa Verdad. Positivo (Sensibilidade)')
    plt.legend()
    plt.title('Curva ROC do Classificador')

    if caminho_persistencia:
        #fig = sns_plot.fig
        plt.savefig(os.path.join(caminho_persistencia, 'curva_roc.png'))
        plt.close(fig)
    else:
        plt.show()

def plotar_curva_pr(precisao, recall, thresholds, y_teste, best_thres_ix=None):
    fig = plt.figure(figsize=(8, 8))
    no_skill = len(y_teste[y_teste == 1]) / len(y_teste)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Modelo Descalibrado')
    plt.plot(recall, precisao, marker='.', label='Resultados Classificador')
    if best_thres_ix is not None:
        plt.scatter(
            recall[best_thres_ix], precisao[best_thres_ix],
            marker='o', color='black',
            label='Melhor Threshold'
        )
    plt.xlabel('Recall')
    plt.ylabel('Precisão')
    plt.legend()
    plt.title('Curva PR (Precisão x Recall)')
    # show the plot
    plt.show()