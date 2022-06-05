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

