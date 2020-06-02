from tensorflow.keras import backend as K


def recall_macro(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precisao_macro(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    precision = precisao_macro(y_true, y_pred)
    recall = recall_macro(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def avaliar_threshold(pos_probs, threshold):
    """Verificar se a probabilidade é igual ou maior que um limiar para classe positiva"""
    return (pos_probs >= threshold).astype('int')