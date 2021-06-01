import numpy as np
import pandas as pd
# from tqdm import tqdm_notebook

from pyts.image import RecurrencePlot, GramianAngularField
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation

class RQA:
    def __init__(
        self,
        classifier,
        appliance_label,
        params={
            "dimension": 1,
            "time_delay": 1,
            "threshold": None,
            "percentage": 10
            },
        rqa_column_names=[
            "Appliance", #"State",
            "Minimum diagonal line length (L_min)",
            "Minimum vertical line length (V_min)",
            "Minimum white vertical line length (W_min)",
            "Recurrence rate (RR)",
            "Determinism (DET)",
            "Average diagonal line length (L)",
            "Longest diagonal line length (L_max)",
            "Divergence (DIV)",
            "Entropy diagonal lines (L_entr)",
            "Laminarity (LAM)",
            "Trapping time (TT)",
            "Longest vertical line length (V_max)",
            "Entropy vertical lines (V_entr)",
            "Average white vertical line length (W)",
            "Longest white vertical line length (W_max)",
            "Longest white vertical line length inverse (W_div)",
            "Entropy white vertical lines (W_entr)",
            "Ratio determinism / recurrence rate (DET/RR)",
            "Ratio laminarity / determinism (LAM/DET)"
            ],
        columns_model=[
            "Recurrence rate (RR)",
            "Determinism (DET)"
            ]
        ):
        self.classifier = classifier
        self.appliance_label = appliance_label
        self.params = params
        self.rqa_column_names = rqa_column_names
        self.columns_model = columns_model

    def serie_to_rqa(self, X):
        
        rqa_data = []

        # for x in tqdm_notebook(X, total=X.shape[0]):
        for x in X:

            # Calculating RQA attributes
            time_series = TimeSeries(x,
                        embedding_dimension=self.params["dimension"],
                        time_delay=self.params["time_delay"])
            settings = Settings(time_series,
                                analysis_type=Classic,
                                neighbourhood=FixedRadius(self.params["percentage"]/100), 
                                # PS.: Utilizando percentage ao inves de threshold 
                                # devido a semanticas distintas entre libs (pyts e pyrqa)
                                # bem como distincao entre RPs (cnn) e RQAs (supervisionado).
                                similarity_measure=EuclideanMetric)
            computation = RQAComputation.create(settings, verbose=False)
            rqa_result = computation.run()

            rqa_data.append( 
                [self.appliance_label]  + list(np.nan_to_num(rqa_result.to_array())) 
                )

        # Numpy to Pandas 
        df_rqa = pd.DataFrame(
            data=rqa_data,
            columns=self.rqa_column_names
        )
        # TODO: persist file (y needed in this case)

        # Select only specific attributes
        X_rqa = df_rqa[self.columns_model].values

        return X_rqa

    def feature_extraction(self, X):
        X_rqa = self.serie_to_rqa(X)
        return X_rqa

    def fit(self, X, y):
        X_features = self.feature_extraction(X) # RQA attr.
        self.classifier.fit(X_features, y)

    def predict(self,X):
        X_features = self.feature_extraction(X) # RQA attr.
        y = self.classifier.predict(X_features)
        return y