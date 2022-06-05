import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

#take existing labels and encode them into corresponding integer values
label_encode = preprocessing.LabelEncoder()
buying = label_encode.fit_transform(list(data['buying']))
door = label_encode.fit_transform(list(data['door']))
maint = label_encode.fit_transform(list(data['maint']))
persons = label_encode.fit_transform(list(data['persons']))
lug_boot = label_encode.fit_transform(list(data['lug_boot']))
safety = label_encode.fit_transform(list(data['safety']))
cls = label_encode.fit_transform(list(data['class']))

predict = 'class'

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print("model accuracy is ", accuracy)

predicted = model.predict(x_test)
names = ['unaccurate', 'accurate', 'good', 'verygood']

"""return accuracy stats of our predictions, as well as the neighbors of a given point"""
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data:", x_test[x], "Actual: ", y_test[x])
    neighbors = model.kneighbors([x_test[x]], 5, True)
    print(neighbors)

