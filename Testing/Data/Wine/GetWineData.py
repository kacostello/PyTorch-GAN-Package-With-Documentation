import pandas as pd


# function to test winequality-white data set
def wine_data():
    wine_file = 'winequality-white.csv'
    file_data = pd.read_csv(wine_file, sep=";")
    file_data = file_data.to_numpy()
    labels = file_data[:, 11].astype(int)
    file_data = (file_data - file_data.mean()) / file_data.std()
    data = file_data[:, [1, 2, 3, 4, 7, 9, 10]]

    return data, labels
