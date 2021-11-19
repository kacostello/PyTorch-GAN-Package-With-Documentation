import SuperTrainer
import SimpleGANTrainer
import pandas as pd
import numpy as np

def wineData():
    wine_file = 'winequality-white.csv'
    wine_data = pd.read_csv(wine_file, sep=";")
    return wine_data

def testWineData(data):
    print(np.shape(data))
    print(data)
    data = data.to_numpy()
    print(data)
    print(data[0])
    # gan = SimpleGANTrainer.SimpleGANTrainer()
    # gan.train()
