import gc 

import tensorflow as tf
from tensorflow.keras import applications as transfer_learning
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.base import clone

def start_tf_session(memory_limit=int(1024*4), debug=False):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            if debug: print("Inicializando sessão com", len(gpus), "GPUs Físicas /", len(logical_gpus), "GPUs Lógicas")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    else:
        if debug: print("Nenhuma GPU disponível ou encontrada")

# start_tf_session()

# Reset Keras Session
def reset_tf_session(model_name, debug=False):
    
    if debug: print("* Reinicializando sessão tensorflow...")
    tf.keras.backend.clear_session()

    try:
        del globals()[model_name] # this is from global space - change this as you need
    except:
        pass

    # if it's done something you should see a number being outputted (print)
    gc.collect()
    
    try:
        start_tf_session(debug=debug)
    except:
        pass

# reset_tf_session(model_name='dlafe')




def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.

      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)

      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.

    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
               -K.mean((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0))

    return binary_focal_loss_fixed