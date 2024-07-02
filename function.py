from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle  # Pour sauvegarder le modèle entraîné

def categorical(df, column):
    #retire les lignes avec données manquantes
    df = df.dropna()
    #liste les items et les trans
    liste_ = list(df[column].value_counts().index)
    df[column] = df[column].apply(lambda x: liste_.index(x))
    return df