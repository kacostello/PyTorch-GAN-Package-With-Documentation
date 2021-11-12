import SuperTrainer
import SimpleGANTrainer
import pandas as pd
import numpy as np


def runGan():
    data = pd.read_csv('C:\\Users\\Ryana\\Documents\\WPI\\Senior\\MQP\\pytorch_GAN_Package\\winequality-white.csv', sep=";")
    print(np.shape(data))
    print(data)
    data = data.to_numpy()
    print(data)
    print(data[0])
    # gan = SimpleGANTrainer.SimpleGANTrainer()
    # gan.train()


if __name__ == "__main__":
    runGan()
