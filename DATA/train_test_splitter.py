import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('ai4i2020.csv')
X_train, X_test, y_train, y_test = train_test_split(df, df['Machine failure'], test_size=0.33, random_state=84)

X_train.to_csv('train.csv', index=False)
X_test.to_csv('test.csv', index=False)
