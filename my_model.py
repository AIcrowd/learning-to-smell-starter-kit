#!/usr/bin/env python
import random

from evaluator import aicrowd_helpers
from evaluator.learning_to_smell import L2SPredictor
from feature_extractor import FeatureExtractorCGR
from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json
import pandas as pd
import numpy as np

class MyModel(L2SPredictor):

    """
    Below paths will be preloaded for you, you can read them as you like.
    """
    training_data_path = None
    test_data_path = None
    vocabulary_path = None

    """
    You can do any preprocessing required for your codebase here like loading up models into memory, etc.
    """
    def predict_setup(self):
       json_file = open('model.json', 'r')
       loaded_model_json = json_file.read()
       json_file.close()
       self.model = model_from_json(loaded_model_json)
       self.model.load_weights("model.h5")
       self.voc = pd.read_csv(self.vocabulary_path, sep='\n', header=None)
       self.preprocessing_setup()
       pass

    def preprocessing_setup(self):
        self.df_test = pd.read_csv(self.test_data_path)
        fe = FeatureExtractorCGR(self.df_test)
        self.df_data = fe.return_data() 

    def return_data_for_smile(self, smile_string):
        try:
            QUERY = 'SMILES == "{0}"'.format(smile_string)
            df = self.df_data.query(QUERY)
            df = df.drop(['SMILES'], axis=1)
            return df 
        except:
            size = len(self.df_data.columns)
            columns = self.df_data.columns
            df = pd.DataFrame(data=[np.zeros(size)], columns=columns)
            df = df.drop(['SMILES'], axis=1)
            return df

    def create_label_list(self):
        s = set()
        for item in self.voc.values:
            s.add(item[0])
        return sorted(list(s))
    
    def sorted_predictions_with_labels(self, predictions):
        sorted_predictions = []
        LABEL_LIST = self.create_label_list()
        for prediction in predictions:
            _z = zip(LABEL_LIST, prediction)
            sorted_z = sorted(_z, key=lambda tup: tup[1], reverse=True)
            sorted_predictions.append(sorted_z)
        return sorted_predictions

    def return_sentences(self, sorted_predictions):
        sentences = []
        tmp_list = []
        pred = sorted_predictions[0]
        for i in range(0, 15, 3):
            for j in range(3):
                tmp_list.append(pred[j+i][0])
            sentences.append(tmp_list)
            tmp_list = []

        return sentences
        
    """
    This function will be called for all the smiles string one by one during the evaluation.
    The return need to be a list of list. Example: [["burnt", "oily", "rose"], ["fresh", "citrus", "green"]]
    NOTE: In case you want to load your model, please do so in `predict_setup` function.
    """
    def predict(self, smile_string):
        df = self.return_data_for_smile(smile_string)
        predictions = self.model.predict(df)
        sorted_predictions = self.sorted_predictions_with_labels(predictions)
        sentences = self.return_sentences(sorted_predictions)
        return sentences

    """
    (optional)
    Add your training code in this function, this will be called only during training locally.
    This will not be executed during the evaluation, remember to save your models/pkl/artifacts properly.
    You can upload them via git (git-lfs in case of large files)
    """
    def train(self):
        raise NotImplementedError()

if __name__ == "__main__":
    submission = MyModel()
    submission.run()
    print("Successfully generated predictions!")
