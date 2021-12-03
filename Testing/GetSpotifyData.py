import SuperTrainer
import SimpleGANTrainer
import pandas as pd
import numpy as np

def spotifyData():
    spotify_file = 'spotify.csv'
    file_data = pd.read_csv(spotify_file)
    file_data = file_data.to_numpy()
    labels = file_data[:, 19].astype(int)
    data = file_data[:, [0, 2, 3, 4, 9, 10, 15, 16, 17]]
    data = data / data.max(axis=0)
    #data = (data - data.mean()) / data.std()
    return data, labels

def useSpotifyData(data):
    # print(np.shape(data))
    # print(data)
    # data = data.to_numpy()
    print(data)
    # print(data[0])
    # gan = SimpleGANTrainer.SimpleGANTrainer()
    # gan.train()

# spotify_data, spotify_labels = spotifyData()
# print(spotify_data)
# print(spotify_labels)
# number_of_rows = spotify_data.shape[0]
# random_indices = np.random.choice(number_of_rows, size=16, replace=False)
# random_rows = spotify_data[random_indices, :]
#
# print(random_rows)
