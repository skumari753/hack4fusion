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


df = pd.read_csv('./CMod_HackForFusion_v2.csv')
df = df[df['intentional_disruption'] != 1.0]

df_drop = df.drop(['Mirnov', 'Te_width', 'intentional_disruption', 'z_error', 'z_times_v_z', 'zcur', 'time', 'v_z'], axis=1)
df_drop['time_until_disrupt'] = df_drop.time_until_disrupt.apply(lambda x: x if not math.isnan(x) else 100.0)

df_clean = df_drop.dropna(how='any', axis=0)
test = df_clean.groupby('shot')['Greenwald_fraction', 'p_icrf'].agg('count')
test = test[test.p_icrf>40]
valid = test.index.values
df_clean = df_clean[df_clean.shot.apply(lambda x: x in valid)]
print(len(df_clean))

thres = 0.5

def create_dataset(data, look_back=1):
    dataX, dataY = [], []
    for dataset in data:
        dataset = np.array(dataset)
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i+look_back), :-1]
            dataX.append(a)
            y = dataset[i + look_back - 1, -1]
            if math.isnan(y):
                dataY.append(False)
            elif y >=  thres:
                dataY.append(False)
            else:
                dataY.append(True)
    return np.array(dataX), np.array(dataY)

def prep_df(df):
    df_list = []
    for s in df['shot'].unique():
        df_list.append(df[df.shot == s])
    return df_list

# convert an array of values into a dataset matrix
thres = 0.25

def create_dataset(data, look_back=1):
    dataX, dataY = [], []
    for dataset in data:
        dataset = np.array(dataset)
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i+look_back), :-1]
            dataX.append(a)
            y = dataset[i + look_back - 1, -1]
            if math.isnan(y):
                dataY.append(False)
            elif y >=  thres:
                dataY.append(False)
            else:
                dataY.append(True)
    return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
np.random.seed(7)
df_norm = df_clean - df_clean.mean()/ df_clean.std()

# split into train and test sets
train_size = int(len(df_norm) * 0.67)
test_size = len(df_norm) - train_size
train, test = pd.DataFrame(df_norm[0:train_size]), pd.DataFrame(df_norm[train_size:len(df_norm)])
train = prep_df(train)
test = prep_df(test)

# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 27), return_sequences=True))
model.add(LSTM(4, input_shape=(look_back, 27)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=100, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()