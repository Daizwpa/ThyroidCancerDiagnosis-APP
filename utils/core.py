from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np
import os
import joblib


def Load_data(path, **kwargs):
    """
        load dataset 
    """
    if not os.path.exists(path):
        raise FileExistsError(path)
    data = pd.read_csv(path, **kwargs)
    # We have only one NA value which in 125_ or 109. #
    data = data.fillna(0.9)
    # TODO: Missed value
    return data


def Get_conflicted_data(dataset, target, exclude):
    data = dataset.drop(exclude, axis=1)
    return np.where((dataset.iloc[:, 1:-1].nunique(axis=1,) != 1), "Yes", "No")


def Save_models(models: [], save_path: str, model_number):
    checkpoint_path = "{0}model{1}.keras"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    for index_model in range(len(models)):
        path = os.path.join(
            save_path,
            checkpoint_path.format(model_number, index_model)
        )
        models[index_model].save(path)


def Save_pipeline(pipeline, path):
    joblib.dump(pipeline, filename=path)
