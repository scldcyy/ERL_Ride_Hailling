import os
import pickle

import numpy as np
import pandas as pd

files=os.listdir("data/data/cd")

for file in files:
    file_name=file.split(".")[0]
    tmp=None
    if file.endswith(".pkl"):
        tmp=pickle.load(open("data/data/cd/"+file,"rb"))
    if file.endswith(".npz") or file.endswith(".npy"):
        tmp=np.load("data/data/cd/"+file)
    if file.endswith('csv'):
        tmp=pd.read_csv("data/data/cd/"+file)
    pass
