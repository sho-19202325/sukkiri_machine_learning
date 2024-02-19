import pandas as pd
import os
from sklearn import tree
import pickle

base_dir = os.path.dirname(os.path.abspath(__file__))

data = {
    'マツダの労働時間': [160, 160],
    'アサギの労働時間': [162, 175]
}

# pd.DateFrame(data, index, columns)
df = pd.DataFrame(data)

# indexの変更
df.index = ['4月', '5月']

# columnの変更
df.columns = ['松田の労働(h)', '浅木の労働(h)']

# 初期化時に指定する場合
# pd.DateFrame(data, index = ['4月', '5月'], columns = ['松田の労働', '浅木の労働'])
print(df)
print(type(df))
print(df.shape)

path_for_kvst = os.path.join(base_dir, '../../csv/KvsT.csv')
df = pd.read_csv(path_for_kvst)
print(df.head(3))

columns = ['身長', '体重']
print(df[columns])

xcol = ['身長', '体重', '年代']
x = df[xcol]

# print(x) 

t = df['派閥']
# print(t)

# モデルの準備
model = tree.DecisionTreeClassifier(random_state = 0)

# 学習の実行
model.fit(x, t)

taro = pd.DataFrame([[170, 70, 20]], columns=xcol)

# 予測
print(model.predict(taro))

matsuda = (172, 65, 20)
asagi = (158, 48, 20)
new_data = pd.DataFrame([matsuda, asagi], columns=xcol)

print(model.predict(new_data))

# 評価
print(model.score(x, t))
path_for_kt_pkl = os.path.join(base_dir, '../../pkls/KinokoTakenoko.pkl')

# with open (path_for_kt_pkl, 'wb') as f:
#     pickle.dump(model, f)

with open (path_for_kt_pkl, 'rb') as f:
  model2 = pickle.load(f)

suzuki = pd.DataFrame([[180, 75, 30]], columns=xcol)
print(model2.predict(suzuki))
