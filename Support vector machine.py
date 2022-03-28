
from sklearn import svm, datasets

import sklearn
from sklearn.metrics import accuracy_score


model = svm.SVC(kernel='linear', C=1, gamma=1)
X, y = datasets.load_iris(return_X_y=True)

print(f"iris: X.shape:{X.shape}, y.shape:{y.shape}")
# print(f'y:{y}')
tr_data, ts_data, tr_label, ts_label = sklearn.model_selection.train_test_split(X,
                                                                                y,
                                                                                random_state=1,
                                                                                train_size=0.6,
                                                                                test_size=0.4)

model.fit(tr_data, tr_label)

print(f"train accuray：{model.score(tr_data, tr_label)}")
print(f"test accuray：{model.score(ts_data, ts_label)}")
tr_predicted = model.predict(tr_data)
ts_predicted = model.predict(ts_data)
print(f"train accuray：{accuracy_score(tr_label, tr_predicted)}")
print(f"test accuray：{accuracy_score(ts_label, ts_predicted)}")
