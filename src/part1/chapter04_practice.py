import pandas as pd
import os
from sklearn import tree
import pickle

base_dir = os.path.dirname(os.path.abspath(__file__))

#q4-1
columns = ['データベースの試験得点', 'ネットワークの試験得点']
data = [
  [70, 80],
  [72, 85],
  [75, 79],
  [80, 92]
]
df = pd.DataFrame(data, columns=columns)
print(df)

#q4-2
index = ['一郎', '次郎', '三郎', '太郎']
df = pd.DataFrame(data, columns=columns, index=index)
print(df)

#q4-3
path_for_ex1 = os.path.join(base_dir, '../../csv/ex1.csv')
df = pd.read_csv(path_for_ex1)
print(df)

# q4-4
print(df.index)

# q4-5

target_columns = ['x0', 'x1']
df = df[target_columns]
print(df)