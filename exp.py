import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import keras
from keras import backend as K


df = pd.read_csv('./CMod_HackForFusion_v2.csv')
df = df[df['intentional_disruption'] != 1.0]

df_drop = df.drop(['Mirnov', 'Te_width', 'intentional_disruption', 'z_error', 'z_times_v_z', 'zcur', 'v_z'], axis=1)
df_drop['time_until_disrupt'] = df_drop.time_until_disrupt.apply(lambda x: x if not math.isnan(x) else 100.0)

df_clean = df_drop.dropna(how='any', axis=0)
test = df_clean.groupby('shot')['Greenwald_fraction', 'p_icrf'].agg('count')
test = test[test.p_icrf>40]
valid = test.index.values
df_clean = df_clean[df_clean.shot.apply(lambda x: x in valid)]
print(len(df_clean))

thres = 0.10

def create_dataset(data, look_back=1):
    dataX, dataY, time, shot = [], [], [], []
    for dataset in data:
        dataset = np.array(dataset)
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i+look_back), :-3]
            dataX.append(a)
            y = dataset[i + look_back - 1, -1]
            if math.isnan(y):
                dataY.append(0.0)
            elif y >=  thres:
                dataY.append(0.0)
            else:
                dataY.append(1.0)
            time.append(dataset[i + look_back - 1, -2])
            shot.append(dataset[i + look_back - 1, -3])
    return np.array(dataX), np.array(dataY), np.array(time), np.array(shot)

def prep_df(df):
    df_list = []
    for s in df['shot'].unique():
        df_list.append(df[df.shot == s])
    return df_list

# fix random seed for reproducibility
# fix random seed for reproducibility
np.random.seed(7)


# fix random seed for reproducibility
np.random.seed(7)


# split into train and test sets
train_size = int(len(df_clean) * 0.67)
test_size = len(df_clean) - train_size
train, test = pd.DataFrame(df_clean[0:train_size]), pd.DataFrame(df_clean[train_size:len(df_clean)])
t_time = train['time_until_disrupt']
test_time = test['time_until_disrupt']
train_s = train['shot']
test_s = test['shot']
t_t = train['time']
test_t = test['time']
train = train.drop(['time_until_disrupt', 'time', 'shot'], axis=1)
test = test.drop(['time_until_disrupt', 'time', 'shot'], axis=1)
m, std = train.mean(), train.std()
train = (train - m)/std
test = (test - m)/std
train['shot'] = train_s
test['shot'] = test_s
train['time'] = t_t
test['time'] = test_t
train['time_until_disrupt'] = t_time
test['time_until_disrupt'] = test_time
train = prep_df(train)
test = prep_df(test)

look_back = 10
trainX, trainY, timeTrain, shotTrain = create_dataset(train, look_back)
testX, testY, timeTest, shotTest = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(10, input_shape=(look_back, 26), return_sequences=True))
model.add(LSTM(5))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer= keras.optimizers.Adam(lr=0.0001))
model.fit(trainX, trainY, epochs=2, batch_size=100, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

import sklearn
train_pred = [0.0 if x <= 0.5 else 1.0 for x in trainPredict[:,0]]
test_pred = [0.0 if x <= 0.5 else 1.0 for x in testPredict[:,0]]
print(sklearn.metrics.accuracy_score(trainY, train_pred))
print(sklearn.metrics.accuracy_score(testY, test_pred))

get_2nd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[1].output])
train_features = get_2nd_layer_output([trainX])[0]
test_features = get_2nd_layer_output([testX])[0]

results = {}
for s,t,p in zip(shotTrain, timeTrain, train_pred):
    if p == 1.0:
        if s not in results:
            results[s] = t + 0.10

for s,t,p in zip(shotTest, timeTest, test_pred):
    if p == 1.0:
        if s not in results:
            results[s] = t + 0.10

for s in set(df['shot'].unique()) - results.keys():
    results[s] = None

import csv

with open('discharge_prediction.csv', mode='w') as file:
    prediction_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    prediction_writer.writerow(['Shot_Id', 'T', 'March'])
    for shot_id, pred_time in results.items():
        prediction_writer.writerow([shot_id, pred_time])