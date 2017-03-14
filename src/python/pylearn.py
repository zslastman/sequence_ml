print("pylearn script running")


class super:
	def hello(self):
		self.data1='spam'

class sub(super):
	def hola(self):
		self.data2 = 'eggs'




# -------------------------------------------------------------------------------
# --------Pandas tutorial
# -------------------------------------------------------------------------------




di = {'a':3,'b':300,'c':2}

s1 = pd.Series((1,10,100,1000))
s2 = pd.Series((3,1/3,1/30,1/300))

s1[[1,2]]-s2[1:4]



# fake up some data
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low), 0)

# basic plot
plt.boxplot(data)
plt.show()

df=pd.DataFrame({'a':pd.Series(np.random.rand(100)*range(100)), 'b':pd.Series(np.random.rand(100)*range(100))})



# -------------------------------------------------------------------------------
# --------Keras tutorial
# -------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#i'll use urllib, and pandas to get the pima indians dataset
import urllib.request
local_filename, headers = urllib.request.urlretrieve('http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data')
#now read in the data
pimadata = pd.read_csv(open(local_filename),header=None)
X,Y = np.array(pimadata.loc[:,0:7]),np.array(pimadata.loc[:,8])

import sklearn.linear_model


fit = sklearn.linear_model.LogisticRegression()
fit = fit.fit(X[:,:],Y)
fit.score(X[:,:],Y)


model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10)

#
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
