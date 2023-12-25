import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from mlp import model




train_label = pd.read_csv('/Users/hongseoklee/VSCodeWorkspace/AIs/NCSoft/final_data/train/train_label.csv')
print(train_label)


train_activity = pd.read_csv('/Users/hongseoklee/VSCodeWorkspace/AIs/NCSoft/final_data/train/train_activity.csv')
print(train_activity)


df = pd.merge(left = train_label , right = train_activity, how = "inner", on = "acc_id")
print(df)


# 원하는 라벨 순서 정의
label_mapping = {'week':0, 'month':1, '2month':2, 'retained':3}

# 데이터프레임 열에 매핑 적용
df['label'] = df['label'].map(label_mapping)


categorical_features = ["acc_id"]
# for문을 사용하여 Label Encoding 적용
label_encoder = LabelEncoder()
for column in categorical_features:
    df[column] = label_encoder.fit_transform(df[column])


# 결과 출력
print(df)


train_x = df.drop(labels = ["label", "acc_id"], axis = 1)
train_y = df["label"]


columns = train_x.columns

print(train_x)

print(train_x.shape)

selected_features = ['item_hongmun', 'party_chat', 'cnt_dt', 'npc_hongmun', 'whisper_chat', 'wk', 'cnt_use_buffitem', 'get_money', 'quest_hongmun', 'guild_chat', 'game_combat_time', 'play_time']
train_x = train_x[selected_features]


#검증 데이터셋 추출
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)


model.fit(train_x, train_y, epochs=1, batch_size=32, validation_split=0.2)

# 훈련된 모델을 사용하여 검증 세트 예측
y_pred = model.predict(val_x)

y_pred = list(map(int, y_pred))

print(y_pred)
# 분류 보고서 출력
print('Classification Report:')
print(classification_report(val_y, y_pred))