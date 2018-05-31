import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

sns.factorplot("type", col="color", col_wrap=4, data=train, kind="count", size=2.4, aspect=.8)
plt.show()