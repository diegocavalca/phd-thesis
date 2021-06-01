import os
import random as rn

import cv2
import numpy as np
from sklearn.base import clone
from pyts.image import RecurrencePlot

class DLAFE:
    def __init__(self,
                feature_extractor=None, 
                preprocess_input=None,
                classifier=None,
                rp_params={
                    "dimension": 1,
                    "time_delay": 1,
                    "threshold": None,
                    "percentage": 10
                },
                input_shape=(224, 224, 3),
                data_type=np.float32,
                normalize=False, 
                standardize=False, 
                rescale=False,
                seed=33
                ):
        self.model = None
        self.feature_extractor = feature_extractor
        self.preprocess_input = preprocess_input
        self.classifier = classifier
        self.rp_params = rp_params
        self.input_shape = input_shape
        self.data_type = data_type
        self.normalize = normalize
        self.standardize = standardize
        self.rescale = rescale
        self.seed = seed

        # Garantindo reprodutibilidade
        # Travar Seed's
        np.random.seed(self.seed)
        rn.seed(self.seed)
        os.environ['PYTHONHASHSEED']=str(self.seed)


    def seq_to_rp(self, X):
        
        X_rp = np.empty((len(X), *self.input_shape))
        
        for i, x in enumerate(X):
            
            img = RecurrencePlot(**self.rp_params).fit_transform([x])[0]
            img = cv2.resize(
                    img, 
                    dsize=self.input_shape[:2], 
                    interpolation=cv2.INTER_CUBIC
                ).astype(self.data_type)

            if np.sum(img) > 0:
                # TODO: improve fit/predict statistics
                # Normalizar
                if self.normalize:
                        img = (img - img.min()) / (img.max() - img.min()) # MinMax (0,1)
                    #img = (img - img.mean()) / np.max([img.std(), 1e-4])

            #     # centralizar
            #     if centralizar:
            #         img -= img.mean()

                # Padronizar
                elif self.standardize:
                    img = (img - img.mean())/img.std()#tf.image.per_image_standardization(img).numpy()
                    
                elif self.rescale:
                    img = (img - img.min()) / (img.max() - img.min())

            # N canais
            img = np.stack([img for i in range(self.input_shape[-1])],axis=-1).astype(self.data_type)     
            
            X_rp[i,] = img
            
        return X_rp
    
    def preprocessing(self, X):
        # Seq to RP image
        recurrence_plots = self.seq_to_rp(X)
        # Image preprocessing
        return self.preprocess_input(recurrence_plots)
    
    def feature_extraction(self, X):
        # Seq -> RP -> Preproc TL
        images = self.preprocessing(X)
        # Preprocesse Image Feat. Extract.
        return self.feature_extractor.predict(images)#, batch_size=32)
                
    def fit(self,X,y):
        X_features = self.feature_extraction(X) # Transfer-learning
        self.classifier.fit(X_features, y)
    
    def predict(self,X):
        X_features = self.feature_extraction(X) # Transfer-learning
        y = self.classifier.predict(X_features)
        return y