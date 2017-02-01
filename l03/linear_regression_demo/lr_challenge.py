import  numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# read data
dataframe = pd.read_csv('challenge_dataset.txt', names=['x','y'])
dataframe = dataframe.sort_values(by=['x'])

x_values = dataframe[['x']]
y_values = dataframe[['y']]

# train model on data
lr = linear_model.LinearRegression()
lr.fit(x_values, y_values)

# compute error using the train data again
y_fit = lr.predict(x_values)
err = np.sum(np.square(y_fit - y_values))/len(y_values)
print('Error =%.2f' % err)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, y_fit)
plt.show()
