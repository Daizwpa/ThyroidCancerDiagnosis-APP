from Model.base.featuresExtractor import RadiomicsExtractor
import pandas as pd
import joblib
import os
import tensorflow as tf
import six
from Model.preprocessing.datasetManager import DatasetManger


class Classifier:

    def __init__(self, wieghts_path, pipeline_sclaer_path, logPathRadimoics, SettingsPathRadimoics):
        assert os.path.exists(pipeline_sclaer_path) == True
        assert os.path.exists(wieghts_path) == True
        self.model = model = tf.keras.models.load_model(wieghts_path)
        self.pre_processor = joblib.load(filename=pipeline_sclaer_path)
        self.radiomic_extractor = RadiomicsExtractor(
            logPath=logPathRadimoics, settingsPath=SettingsPathRadimoics)

    def __extract_radiomics_features(self, image, mask):
        """
        parameter:
            image: 2D Array image.
            mask: 2D Array image.
        Return: Dictionary of features for region of interest of image.
        Raise:
            Exception And None.
        """
        try:

            extracted_features = self.radiomic_extractor.extract(
                image=image, mask=mask)
            data_specific_row = {}
            for k, v in six.iteritems(extracted_features):
                if "diagnostics" not in k:
                    data_specific_row[k] = [v]
            return data_specific_row
        except:
            raise

    def __concatinate_radiomics_with_clinical_data(self, clnical_df: pd.DataFrame, dicOfFeatures: dict):
        """ 
            Concatinate Data frame with dictonary.
            Parameter:
                clinical_df: pd.DataFrame
                dicOfFeatures: dict
            Return:
                DataFrame
        """
        try:
            features_pd = pd.DataFrame(dicOfFeatures)
            clnical_df = clnical_df.reset_index()
            result = clnical_df.merge(features_pd, how="cross")
            return result
        except:
            raise

    def __perform_preprocessing(self, raw_input: pd.DataFrame):
        try:
            return self.pre_processor.transform(raw_input)
        except:
            raise

    def __convert_data_to_tensor(self, data: pd.DataFrame):
        try:
            managerdataset = DatasetManger(data, None, batch_size=1)
            result = managerdataset.convert_df_to_dataset_x(data)
            return result

        except:
            raise

    def Classify(self, image, mask, clinical_data: pd.DataFrame):
        # extract radimoic features
        radiomics_features = self.__extract_radiomics_features(image, mask)
        # Concatinate radiomic features
        raw_input = self.__concatinate_radiomics_with_clinical_data(
            clinical_data, radiomics_features)
        # pre-processing
        ready_input = self.__perform_preprocessing(raw_input)
        tensor_form = self.__convert_data_to_tensor(ready_input)
        result = self.model.predict(tensor_form)
        result = tf.squeeze(result).numpy()
        return result
