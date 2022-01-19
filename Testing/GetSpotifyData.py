import pandas as pd
import os


# function to test spotify data set
def spotify_data():
    cwd = os.getcwd()
    spotify_file = cwd + '\\pytorch_GAN_Package\\Testing\\spotify.csv'
    file_data = pd.read_csv(spotify_file)
    file_data = file_data.to_numpy()
    labels = file_data[:, 19].astype(int)
    data = file_data[:, [0, 2, 3, 4, 9, 10, 15, 16, 17]]
    maxDataValues = data.max(axis=0)
    data = data / maxDataValues
    return data, labels, maxDataValues
