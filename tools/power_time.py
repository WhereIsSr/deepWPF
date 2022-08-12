import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from __init__ import DATAPATH, POSPATH
import os


def power_time(data_path, save_path):
    df_data = pd.read_csv(data_path)
    df_data.replace(to_replace=np.nan, value=0, inplace=True)
    col_data = df_data[[df_data.columns[1], df_data.columns[2], df_data.columns[-1]]]
    group_data = col_data.groupby([col_data.columns[0], col_data.columns[1]]).sum()
    patVs = np.array(group_data[group_data.columns[-1]])
    plt.plot(range(patVs.shape[0]), patVs)
    plt.show()



if __name__ == "__main__":
    power_time(DATAPATH, "./tools/output")