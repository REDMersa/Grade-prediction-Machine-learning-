import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle
from sklearn.utils import shuffle
data = pd.read_csv('student-mat.csv', sep=';')
print(data.head())
data = data[['G1', 'G2','absences','studytime','failures','G3']]
print(data.head())
predict = 'G3'
x = np.array(data.drop([predict],1))
y = np.array(data[predict])
x_train, x_test, y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size =0.1)
model = linear_model.LinearRegression()
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)
with open('studentsmodel.pickle', 'wb') as f:
    pickle.dump(model, f)
pickle_in = open('studentsmodel.pickle', 'rb')
model = pickle.load(pickle_in)
print('cofficent: \n', model.coef_)
print('intercept: \n', model.intercept_)
prediction = model.predict(x_test)
for x in range(len(prediction)):
    print(prediction[x],x_test[x], y_test[x])
p = 'G1'
Final_Grade = 'G3'
style = 'ggplot'
pyplot.scatter(data[p],data[Final_Grade])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()