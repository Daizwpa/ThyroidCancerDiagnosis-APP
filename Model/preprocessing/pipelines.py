
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd


class Data_Process:

    def __init__(self, dataframe, categorical_columns, numerical_columns, Ti_rads_columns):
        self.dataframe = dataframe
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.Ti_rads_columns = Ti_rads_columns
        self.scaled_data = None

        self.categorical_pipeline = Pipeline([
            ('OneHotEncoder', OneHotEncoder())

        ])

        self.numerical_pipeline = Pipeline([
            ('zscoreNormlization', StandardScaler())
        ])

        self.Ti_rads_pipeline = Pipeline([
            ('OrdinalEncoder', OrdinalEncoder()),
            ('minmax', MinMaxScaler(feature_range=(-1, 1)))
        ])

        self.pre_processing = ColumnTransformer([
            ('categorical', self.categorical_pipeline, self.categorical_columns),
            ('num', self.numerical_pipeline, self.numerical_columns),
            ('tirads', self.Ti_rads_pipeline, self.Ti_rads_columns)
        ])

    def fit_transform(self):
        if self.scaled_data is None:
            self.scaled_data = pd.DataFrame(self.pre_processing.fit_transform(
                self.dataframe), columns=self.pre_processing.get_feature_names_out(), index=self.dataframe.index)
        return self.scaled_data

    def transform(self, df):
        return pd.DataFrame(self.pre_processing.transform(
            df), columns=self.pre_processing.get_feature_names_out(), index=df.index)
