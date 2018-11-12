import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(style='darkgrid')
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from keras.utils import to_categorical


data = pd.read_csv("hour.csv").values

x = data[:,0]
y = data[:,11]

#-----------------Visualization---------------------#
y1 = data[:,1].reshape(17379,1)
y2 = data[:,2].reshape(17379,1)
y3 = data[:,3].reshape(17379,1)
y4 = data[:,4].reshape(17379,1)
y5 = data[:,5].reshape(17379,1)
y6 = data[:,6].reshape(17379,1)
y7 = data[:,7].reshape(17379,1)
y8 = data[:,8].reshape(17379,1)
y9 = data[:,9].reshape(17379,1)
y10 = data[:,10].reshape(17379,1)
y11 = data[:,11].reshape(17379,1)
y12 = data[:,12].reshape(17379,1)
y13 = data[:,13].reshape(17379,1)
y14 = data[:,14].reshape(17379,1)
y15 = data[:,15].reshape(17379,1)
y16 = data[:,16].reshape(17379,1)

# plt.plot(x,y2,x,y3)
# plt.show()
# plt.close()

# plt.plot(x,y4)
# plt.show()
# plt.close()

# plt.plot(x,y9)
# plt.show()
# plt.close()

# plt.plot(x,y6,x,y7)
# plt.show()
# plt.close()

# plt.plot(x,y10,x,y11)
# plt.show()
# plt.close()
###----------------------------------------------------###

##-------- Studying correlation between variables--------##

# print(pearsonr(y2,y3))
# # -0.0107
# print(pearsonr(y6,y7))
# #-0.102
# print(pearsonr(y7,y8))
# # 0.035
# print(pearsonr(y8,y9))
# #0.04
# print(pearsonr(y9,y10))
# # -0.1
# print(pearsonr(y10,y11))
# 0.98 : Highly correlated 
# print(pearsonr(y10,y12))
# # -0.06
# print(pearsonr(y16,y15))
# casual, registered and total all highly correlated with each other
# Possible relation: Total = Linear combination of casual and registered.

# Model Selection and Training

y2 = y2-1
y4 = y4-1
# Converting categorical data into one-hot representation
y2 = to_categorical(y2)#,num_classes=4)
y4 = to_categorical(y4)#,num_classes=12)
y5 = to_categorical(y5)#,num_classes=24)
y7 = to_categorical(y7)#,num_classes=7)
y9 = to_categorical(y9)#,num_classes=4)


#print(y2.shape)

X1 = np.concatenate((y2,y3,y4,y5,y6,y7,y8,y9,y11,y12,y13),axis=1)
y = y16#data[:,16]

reg = LinearRegression()
reg.fit(X1,y)
print(reg.coef_)
r_square = reg.score(X1,y)
#print(X2.shape[1])
n = X1.shape[0]
p = X1.shape[1]
adjusted_r_square = 1-(1 - r_square)*(n-1)/(n-p-1)
print(r_square)
print(adjusted_r_square)

###-----------------------------------------------###

# Time series approach

from statsmodels.tsa.arima_model import ARIMA
#from statsmodels.tsa.statespace import SARIMAX

model = ARIMA(y16,order=(4,0,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot()
# plt.show()
# residuals.plot(kind='kde')
# plt.show()
# print(residuals.describe())

# mod = SARIMAX(y16,order=(1,0,1),seasonal_order=(1,1,1,4))
# results = mod.fit()
# print(results.summary())

#plt.plot(x[0:1808],y16[0:1808])
#plt.show()
#plt.close()