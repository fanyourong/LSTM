import 
# import numpy
# import matplotlib.pyplot as plt
# from pandas import read_csv
# import math
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# import  pandas as pd
# import  os
# from keras.models import Sequential, load_model
# # load the dataset
# dataframe = read_csv('./international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
#
# dataset = dataframe.values
# train_size = int(len(dataset) * 0.65)
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
# trainlist = dataset[:train_size,:]
# testlist = dataset[train_size:,:]
#
#
#
# # X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
#
# # convert an array of values into a dataset matrix
# def create_dataset(dataset, look_back):
# #这里的look_back与timestep相同
#     dataX, dataY = [], []
#     for i in range(len(dataset)-look_back-1):
#         a = dataset[i:(i+look_back)]
#         dataX.append(a)
#         dataY.append(dataset[i+look_back])
#
#     return numpy.array(dataX),numpy.array(dataY)
#
# look_back = 1
# trainX,trainY  = create_dataset(trainlist,look_back)
# testX,testY = create_dataset(testlist,look_back)


# # reshape input to be [samples, time steps, features]
# trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1] ,1 ))
# print(trainX.shape,trainY.shape)
# print(testX.shape,testY.shape)
# #
# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(4, input_shape=(1, look_back)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# model.save(os.path.join("DATA","TJ15New" + ".h5"))
# # make predictions
# model = load_model(os.path.join("DATA","TJ15New" + ".h5"))
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
#
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform(trainY)
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform(testY)
#
# plt.plot(trainY)
# plt.plot(trainPredict[1:])
# plt.show()
# plt.plot(testY)
# plt.plot(testPredict[1:])
# plt.show()
# #
