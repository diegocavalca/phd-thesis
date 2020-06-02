import os

from sklearn.metrics import *
import numpy as np
import pandas as pd
from IPython.display import display
from collections import Counter

from .metricas import *
from .graficos import *


def relatorio_classificacao(y_test, y_pred, labels=None):
    final_performance = []

    for i in range(y_test.shape[1]):
        test = y_test[:, i]
        predicted = y_pred[:, i]
        # acc, prec, rec, f1, f1m, hl, supp = metrics(y_test[:, i], y_pred[:, i])
        acc = accuracy_score(test, predicted)
        prec = precision_score(test, predicted)
        rec = recall_score(test, predicted)
        f1 = f1_score(test, predicted)
        f1m = f1_score(test, predicted, average='macro')
        hl = hamming_loss(test, predicted)
        auc_ = roc_auc_score(test, predicted)
        y_i = y_test[:, i]
        supp_0 = y_i[y_i == 0].shape[0]
        supp_1 = y_i[y_i == 1].shape[0]

        final_performance.append([
            labels[i] if labels is not None else i,
            round(acc * 100, 2),
            round(prec * 100, 2),
            round(rec * 100, 2),
            round(f1 * 100, 2),
            round(f1m * 100, 2),
            round(hl, 2),
            round(auc_, 2),
            supp_0,
            supp_1
        ])

    print("* Desempenho do Modelo Multi-rótulo por Aparelho:")
    df_metrics = pd.DataFrame(
        data=final_performance,
        columns=["Aparelho", "Acurácia", "Precisão", "Recall", "F1-score", "F1-macro", "Hamming Loss", "AUC",
                 "Suporte (y=0)", "Suporte (y=1)"]
    )
    display(df_metrics)

    print("")
    print("* Desempenho Médio Geral:")
    display(df_metrics.describe().round(2).loc[['mean', 'max', 'min']])

    print("")
    print("* Matriz de Confusão (Estados 0/1 - OFF/ON), por Aparelho:")

    cms = multilabel_confusion_matrix(y_test, y_pred)
    for i, a in enumerate(labels):
        print("")
        print(" - {}:".format(a))
        print(cms[i])
    # print(, labels= appliance_labels)


def relatorio_classificacao_aparelho(modelo, X_teste, y_teste, label=None, caminho_persistencia=None):
    final_performance = []

    label = ''.join([label[0].upper(), label[-(len(label) - 1):].lower()])
    y_pred = modelo.predict(X_teste).round()

    test = y_teste
    predicted = y_pred
    # acc, prec, rec, f1, f1m, hl, supp = metrics(y_teste[:, i], y_pred[:, i])
    acc = accuracy_score(test, predicted)
    prec = precision_score(test, predicted)
    rec = recall_score(test, predicted)
    f1 = f1_score(test, predicted)
    f1m = f1_score(test, predicted, average='macro')
    hl = hamming_loss(test, predicted)
    auc_ = roc_auc_score(test, predicted) if len(np.unique(test)) > 1 else 0
    y_i = y_teste
    supp_0 = y_i[y_i == 0].shape[0]
    supp_1 = y_i[y_i == 1].shape[0]

    final_performance = [[
        #         label,
        round(acc * 100, 2),
        round(prec * 100, 2),
        round(rec * 100, 2),
        round(f1 * 100, 2),
        round(f1m * 100, 2),
        round(hl, 2),
        round(auc_, 2),
        supp_0,
        supp_1
    ]]

    print("* Desempenho do Classificador `{}`:".format(label))
    df_metrics = pd.DataFrame(
        data=final_performance,
        index=[label],
        columns=["Acurácia", "Precisão", "Recall", "F1-score", "F1-macro", "Hamming Loss", "AUC",
                 "Suporte (y=0)", "Suporte (y=1)"]
    )
    display(df_metrics.transpose())

    # Salvar graficos, caso tenha path de persistencia
    if caminho_persistencia is not None:
        if not os.path.isdir(caminho_persistencia):
            os.makedirs(caminho_persistencia)

    print("")
    print("* Matriz de Confusão (Estados 0/1 - OFF/ON):")
    plotar_matriz_confusao(y_teste, y_pred,
                           labels=np.unique(y_teste),
                           caminho_persistencia=caminho_persistencia)

    print("")
    print("* Análise Curva ROC:")
    plotar_curva_roc(modelo, X_teste, y_teste, caminho_persistencia)
    # if caminho_persistencia:
    #     plt.savefig(os.path.join(caminho_persistencia, 'curva_roc.png'))
    #     plt.close(fig)

    print("")
    print("* Análise de Convergência do Modelo:")

    # accuracy
    historico = modelo.history
    fig = plt.figure(figsize=(15, 5))
    plt.plot(historico.history['accuracy'])
    plt.plot(historico.history['val_accuracy'])
    plt.title('Análise de Convergência do Modelo - Acurácia')
    plt.ylabel('Score')
    plt.xlabel('Época')
    plt.legend(['Treino', 'Teste'], loc='upper left')
    if caminho_persistencia:
        plt.savefig(os.path.join(caminho_persistencia, 'convergencia_acuracia.png'))
        plt.close(fig)
    else:
        plt.show()

    # loss
    fig = plt.figure(figsize=(15, 5))
    plt.plot(historico.history['loss'])
    plt.plot(historico.history['val_loss'])
    plt.title('Análise de Convergência do Modelo - Loss')
    plt.ylabel('Score')
    plt.xlabel('Época')
    plt.legend(['Treino', 'Teste'], loc='upper left')
    if caminho_persistencia:
        plt.savefig(os.path.join(caminho_persistencia, 'convergencia_loss.png'))
        plt.close(fig)
    else:
        plt.show()


def avaliar_modelo_binario_cv(modelo, X, y, threshold=None, folds=10, exibir_relatorio=False, tamanho_teste=0.3,
                              seed=42):
    """
    Função de avaliação para labels binários, com suporte a ROC.
    """
    # Métricas de avaliação
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    f1_micros = []
    roc_aucs = []
    # Métricas classe minoritaria
    accuracies_min = []
    precisions_min = []
    recalls_min = []
    f1s_min = []
    f1_micros_min = []
    # roc_aucs_min = []

    # Variaveis auxiliares para Plot ROC-Folds
    # VSF FDP
    tprs = []
    fprs = []
    aucs = []
    base_fpr = np.linspace(0, 1, 101)

    # Selecionando a classe minoritaria
    classe_minoritaria = Counter(y).most_common()[-1][0]

    # Splits estratificados
    fig = plt.figure(figsize=(8, 8))
    print("Validando usando K-folds (K={})...".format(folds))
    skf = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
    for i, (treino_ix, teste_ix) in tqdm_notebook(enumerate(skf.split(X, y)), total=folds):

        # Preparar dados
        # Teste
        if type(X) == pd.core.frame.DataFrame:
            X_treino = X.loc[treino_ix, :]
        else:
            if len(X.shape) > 1:
                X_treino = X[treino_ix, :]
            else:
                X_treino = X[treino_ix]
        y_treino = pd.factorize(y[treino_ix])[0]
        # Teste
        if type(X) == pd.core.frame.DataFrame:
            X_teste = X.loc[teste_ix, :]
        else:
            if len(X.shape) > 1:
                X_teste = X[teste_ix, :]
            else:
                X_teste = X[teste_ix]
        y_teste = pd.factorize(y[teste_ix])[0]

        # Ajustando modelo
        pipeline = modelo
        pipeline.fit(X_treino, y_treino)

        # Inferindo no conjunto de teste
        y_pred = pipeline.predict(X_teste)
        y_proba = pipeline.predict_proba(X_teste)[:, 1]

        # Verificar uso de threshold, p/ transformar em rotulo discreto
        if threshold is not None:
            y_pred = avaliar_threshold(y_proba, threshold)

        # Avaliando previsoes
        accuracies.append(accuracy_score(y_teste, y_pred))
        precisions.append(precision_score(y_teste, y_pred, average='macro'))
        recalls.append(recall_score(y_teste, y_pred, average='macro'))
        f1s.append(f1_score(y_teste, y_pred, average='macro'))
        f1_micros.append(f1_score(y_teste, y_pred, average='micro'))
        roc_aucs.append(roc_auc_score(y_teste, y_proba, average='macro'))

        # Registros da classe minoritaria (indices)
        i_min = np.where(y_teste == classe_minoritaria)
        # Avaliando previsoes
        accuracies_min.append(accuracy_score(y_teste[i_min], y_pred[i_min]))
        precisions_min.append(precision_score(y_teste[i_min], y_pred[i_min], average='macro'))
        recalls_min.append(recall_score(y_teste[i_min], y_pred[i_min], average='macro'))
        f1_micros_min.append(f1_score(y_teste[i_min], y_pred[i_min], average='micro'))
        f1s_min.append(f1_score(y_teste[i_min], y_pred[i_min], average='macro'))
        # roc_aucs_min.append(roc_auc_score(y_teste[i_min], y_pred[i_min], average='macro'))

        # Plotando curva ROC FOLD #i
        fpr, tpr, thresholds = roc_curve(y_teste, y_proba)
        fold_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr, 'b', alpha=0.15,
            label=r'ROC fold {} (AUC = {:.2f})'.format(i, fold_auc)
        )
        tpr = interpolate(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
        aucs.append(fold_auc)

    # Plotar Media ROCs K-folds
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    plt.plot(base_fpr, mean_tprs, 'b',
             label=r'ROC Média (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_auc))
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.33)
    plt.plot([0, 1], [0, 1], 'r--', label='Modelo Descalibrado')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.title('Curvas ROC | K-Folds')
    plt.ylabel('Taxa de Verdadeiro Positivo (Sensibilidade)')
    plt.xlabel('Taxa de Falso Positivo (Especificidade)')
    plt.legend()
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

    # Relatorio de classificacao para um fold aleatorio
    if exibir_relatorio:

        print("Score médio do pipeline ({}-Folds):".format(folds))
        print("- - - - - - - - - - -")
        print("Acurácia:", np.mean(accuracies).round(2))
        print("Precisão:", np.mean(precisions).round(2))
        print("Recall:", np.mean(recalls).round(2))
        print("F1-macro:", np.mean(f1s).round(2))
        print("F1-micro:", np.mean(f1_micros).round(2))
        print("AUC ROC:", np.mean(roc_aucs).round(2))
        print("- - - - - -")
        print("   Métricas da Classe minoritária ({}):".format(classe_minoritaria))
        print()
        print("   -> accuracy:", np.mean(accuracies_min).round(2))
        print("   -> precision_macro:", np.mean(precisions_min).round(2))
        print("   -> recall_macro:", np.mean(recalls_min).round(2))
        print("   -> f1_macro:", np.mean(f1s_min).round(2))
        print("   -> f1_micro:", np.mean(f1_micros_min).round(2))
        # print("   -> roc_auc:", np.mean(roc_aucs_min).round(2))

        # Split estratificado
        print()
        print("=" * 100)
        print()
        print("Validando no conjunto de Treino/Teste (teste = {})...".format(tamanho_teste))
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y,
            test_size=tamanho_teste,
            stratify=y,
            random_state=seed
        )

        # Isntanciando pipeline
        pipeline = modelo
        pipeline.fit(X_treino, y_treino)

        # Inferindo no conjunto de teste
        y_pred = pipeline.predict(X_teste)
        y_proba = pipeline.predict_proba(X_teste)[:, 1]

        # Verificar uso de threshold, p/ transformar em rotulo discreto
        if threshold is not None:
            y_pred = avaliar_threshold(y_proba, threshold)

        # Calcular e exibir curva ROC / thresholds
        tfp, tvp, thresholds = roc_curve(y_teste, y_proba)
        plotar_curva_roc(tfp, tvp, thresholds)

        # Calcular e execiver curva PR / thresholds
        ps, rs, thresholds = precision_recall_curve(y_teste, y_proba)
        plotar_curva_pr(ps, rs, thresholds, y_teste)

        # Métricas do modelo para o conjunto de teste
        print(classification_report(y_teste, y_pred))

    return (precisions, recalls, f1s, roc_aucs)


def avaliar_modelo_multiclasse(modelo, X, y, threshold=None, folds=10, exibir_relatorio=False, tamanho_teste=0.3,
                               seed=42):
    """
    Função de avaliação para conjuntos de dados multiclasse, sem suporte a ROC.
    """
    # Métricas de avaliação
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    f1_micros = []

    # Métricas classe minoritaria
    accuracies_min = []
    precisions_min = []
    recalls_min = []
    f1s_min = []
    f1_micros_min = []

    # Selecionando a classe minoritaria
    classe_minoritaria = Counter(y).most_common()[-1][0]

    # Splits estratificados
    print("Validando usando K-folds (K={})...".format(folds))
    skf = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
    for i, (treino_ix, teste_ix) in tqdm_notebook(enumerate(skf.split(X, y)), total=folds):

        # Preparar dados
        # Teste
        if type(X) == pd.core.frame.DataFrame:
            X_treino = X.loc[treino_ix, :]
        else:
            if len(X.shape) > 1:
                X_treino = X[treino_ix, :]
            else:
                X_treino = X[treino_ix]
        y_treino = pd.factorize(y[treino_ix])[0]
        # Teste
        if type(X) == pd.core.frame.DataFrame:
            X_teste = X.loc[teste_ix, :]
        else:
            if len(X.shape) > 1:
                X_teste = X[teste_ix, :]
            else:
                X_teste = X[teste_ix]
        y_teste = pd.factorize(y[teste_ix])[0]

        # Ajustando modelo
        pipeline = modelo
        pipeline.fit(X_treino, y_treino)

        # Inferindo no conjunto de teste
        y_pred = pipeline.predict(X_teste)
        y_proba = pipeline.predict_proba(X_teste)[:, 1]

        # Verificar uso de threshold, p/ transformar em rotulo discreto
        if threshold is not None:
            y_pred = avaliar_threshold(y_proba, threshold)

        # Avaliando previsoes
        accuracies.append(accuracy_score(y_teste, y_pred))
        precisions.append(precision_score(y_teste, y_pred, average='macro'))
        recalls.append(recall_score(y_teste, y_pred, average='macro'))
        f1s.append(f1_score(y_teste, y_pred, average='macro'))
        f1_micros.append(f1_score(y_teste, y_pred, average='micro'))

        # Registros da classe minoritaria (indices)
        i_min = np.where(y_teste == classe_minoritaria)
        # Avaliando previsoes
        accuracies_min.append(accuracy_score(y_teste[i_min], y_pred[i_min]))
        precisions_min.append(precision_score(y_teste[i_min], y_pred[i_min], average='macro'))
        recalls_min.append(recall_score(y_teste[i_min], y_pred[i_min], average='macro'))
        f1_micros_min.append(f1_score(y_teste[i_min], y_pred[i_min], average='micro'))
        f1s_min.append(f1_score(y_teste[i_min], y_pred[i_min], average='macro'))

    # Relatorio de classificacao para um fold aleatorio
    if exibir_relatorio:

        print("Score médio do pipeline ({}-Folds):".format(folds))
        print("- - - - - - - - - - -")
        print("Acurácia:", np.mean(accuracies).round(2))
        print("Precisão:", np.mean(precisions).round(2))
        print("Recall:", np.mean(recalls).round(2))
        print("F1-macro:", np.mean(f1s).round(2))
        print("F1-micro:", np.mean(f1_micros).round(2))

        print("- - - - - -")
        print("   Métricas da Classe minoritária ({}):".format(classe_minoritaria))
        print()
        print("   -> accuracy:", np.mean(accuracies_min).round(2))
        print("   -> precision_macro:", np.mean(precisions_min).round(2))
        print("   -> recall_macro:", np.mean(recalls_min).round(2))
        print("   -> f1_macro:", np.mean(f1s_min).round(2))
        print("   -> f1_micro:", np.mean(f1_micros_min).round(2))
        # print("   -> roc_auc:", np.mean(roc_aucs_min).round(2))

        # Split estratificado
        print()
        print("=" * 100)
        print()
        print("Validando no conjunto de Treino/Teste (teste = {})...".format(tamanho_teste))
        print()
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y,
            test_size=tamanho_teste,
            stratify=y,
            random_state=seed
        )

        # Isntanciando pipeline
        pipeline = modelo
        pipeline.fit(X_treino, y_treino)

        # Inferindo no conjunto de teste
        y_pred = pipeline.predict(X_teste)
        y_proba = pipeline.predict_proba(X_teste)[:, 1]

        # Verificar uso de threshold, p/ transformar em rotulo discreto
        if threshold is not None:
            y_pred = avaliar_threshold(y_proba, threshold)

        # Métricas do modelo para o conjunto de teste
        print(classification_report(y_teste, y_pred))

    return (precisions, recalls, f1s)


def avaliar_modelo_multirotulo(modelo, X, y, rotulos=None, threshold=None, folds=10, exibir_relatorio=False,
                               tamanho_teste=0.3, seed=42):
    """
    Função de avaliação para labels binários, com suporte a ROC.
    """
    # Métricas de avaliação
    scores = cross_validate(
        modelo,
        X, y,
        scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_micro'],
        cv=folds,
    )
    accuracies = scores["test_accuracy"]
    precisions = scores["test_precision_macro"]
    recalls = scores["test_recall_macro"]
    f1s = scores["test_f1_macro"]
    f1_micros = scores["test_f1_micro"]

    # Relatorio de classificacao para um fold aleatorio
    if exibir_relatorio:

        print("Score médio do pipeline ({}-Folds):".format(folds))
        print("- - - - - - - - - - -")
        print("Acurácia:", np.mean(accuracies).round(2))
        print("Precisão:", np.mean(precisions).round(2))
        print("Recall:", np.mean(recalls).round(2))
        print("F1-macro:", np.mean(f1s).round(2))
        print("F1-micro:", np.mean(f1_micros).round(2))

        # Split
        print()
        print("=" * 58)
        print()
        print("Validando no conjunto de Treino/Teste (teste = {})...".format(tamanho_teste))
        print()
        print("-> Relatório de classificação:")
        print()
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y,
            test_size=tamanho_teste,
            stratify=y,
            random_state=seed
        )

        # Instanciando pipeline
        pipeline = modelo
        pipeline.fit(X_treino, y_treino)

        # Inferindo no conjunto de teste
        y_pred = pipeline.predict(X_teste)
        y_proba = pipeline.predict_proba(X_teste)[:, 1]

        # Verificar uso de threshold, p/ transformar em rotulo discreto
        if threshold is not None:
            y_pred = avaliar_threshold(y_proba, threshold)

        # Métricas do modelo para o conjunto de teste
        print(classification_report(y_teste, y_pred, labels=rotulos))

        # Erros/acertos do modelo para o conjunto de teste
        print("-" * 58)
        print()
        print("-> Matriz de confusão multilabel:")
        print()
        print(multilabel_confusion_matrix(y_teste, y_pred, labels=rotulos))

    return (precisions, recalls, f1s)


