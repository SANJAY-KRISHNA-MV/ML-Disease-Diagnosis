import pandas as pd

def load_data():
    df = pd.read_csv(r'D:\DATA_SCIENCE\PROJECTS\ML Disease Diagnosis\data\Disease_Prediction_Dataset.csv')
    return df

def load_traindata():
    df = pd.read_csv(r'D:\DATA_SCIENCE\PROJECTS\ML Disease Diagnosis\data\Training.csv')
    return df

def load_testdata():
    df = pd.read_csv(r'D:\DATA_SCIENCE\PROJECTS\ML Disease Diagnosis\data\Testing.csv')
    df.dropna(axis=1)
    return df

