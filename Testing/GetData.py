import SuperTrainer
import SimpleGANTrainer
import pandas as pd
import numpy as np

def wineData():
    wine_file = 'winequality-white.csv'
    file_data = pd.read_csv(wine_file, sep=";")
    file_data = file_data.to_numpy()
    labels = file_data[:, 11].astype(int)
    file_data = (file_data - file_data.mean()) / file_data.std()
    data = file_data[:, [1, 2, 3, 4, 7, 9, 10]]

    return data, labels

def useWineData(data):
    # print(np.shape(data))
    # print(data)
    # data = data.to_numpy()
    print(data)
    # print(data[0])
    # gan = SimpleGANTrainer.SimpleGANTrainer()
    # gan.train()

wine_data, wine_labels = wineData()
number_of_rows = wine_data.shape[0]
random_indices = np.random.choice(number_of_rows, size=16, replace=False)
random_rows = wine_data[random_indices, :]

# print(random_rows)
