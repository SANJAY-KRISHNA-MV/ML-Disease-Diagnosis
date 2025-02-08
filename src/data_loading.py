import pandas as pd

def load_data():
    df = pd.read_csv(r'data\Disease_Prediction_Dataset.csv')
    return df

def load_traindata():
    df = pd.read_csv(r'data\Training.csv')
    return df

def load_testdata():
    df = pd.read_csv(r'data\Testing.csv')
    return df

