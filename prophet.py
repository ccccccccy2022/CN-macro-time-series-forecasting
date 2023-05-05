from pmdarima.arima import auto_arima
import pickle
import numpy as np
import pandas as pd
with open('D:/model0629/ttd.pkl','rb') as pk_file:
    train_total_dataload = pickle.load(pk_file)
for l in range(len(train_total_dataload)):
    if l==0:
        trainx=train_total_dataload[0][0]
        trainy=train_total_dataload[0][1]
    else:
        trainx=np.vstack([train_total_dataload[l][0],trainx])
        trainy=np.hstack([train_total_dataload[l][1],trainy])


#
model = auto_arima(X=trainx[,:],y=trainy.reshape(-1, 1), trace=True, error_action='ignore', suppress_warnings=True)
model.fit(trainy)
with open('D:/model0629/ttd.pkl','rb') as pk_file:
    train_total_dataload = pickle.load(pk_file)

with open('D:/model0629/cyd'+str(0)+'.pkl','rb') as pk_file:
    cur_y_dataset = pickle.load(pk_file)
forecast = model.predict(n_periods=1)
# forecast = pd.DataFrame(forecast, index=valid.index, columns=['Prediction'])
#
# # plot the predictions for validation set
# plt.plot(train, label='Train')
# plt.plot(valid, label='Valid')
# plt.plot(forecast, label='Prediction')
