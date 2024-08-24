import numpy as np
import matplotlib.pyplot as plt
from utilities import prepare_data, split_data
from sklearn import svm

data = np.load('data.npy')
labels = np.load('labels.npy')
split = .5
train_data, train_labels, test_data, test_labels = split_data(data, labels, split)
#
num = 2
clf = svm.SVC(random_state = 1337, kernel = 'linear')
train_data_reshaped = train_data.reshape((train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
clf.fit(train_data_reshaped, train_labels)
test_data_reshaped = test_data.reshape((test_data.shape[0], test_data.shape[1] * test_data.shape[2]))
pred_euclid = clf.predict(test_data_reshaped)
acc_euclid = np.count_nonzero(test_labels == pred_euclid)/len(test_labels)
print(acc_euclid)
acc_rcdt = list()
for i in range(num):
    clf = svm.SVC(random_state = 1337, kernel = 'linear', C = 1*len(train_data))
    # clf = svm.SVC(random_state = 0, kernel = 'linear', C = 1e10)
    train_data_transformed = prepare_data(train_data, num_angles=i+1)
    clf.fit(train_data_transformed, train_labels)
    test_data_transformed = prepare_data(test_data, num_angles=i+1)
    pred_rcdt = clf.predict(test_data_transformed)
    acc_rcdt.append(np.count_nonzero(test_labels == pred_rcdt)/len(test_labels))
    print(acc_rcdt)
# np.save('data_acc_rcdt.npy',acc_rcdt)
# np.save('data_acc_euclid.npy',acc_euclid)

X = np.arange(1, num+1, 1)
plt.figure()
plt.plot(X, acc_rcdt, color='red', label='R-CDT space')
plt.plot(X, acc_euclid*np.ones(num), color='blue', label='Euclidean space')
plt.xticks(np.arange(1, num+1, 1))
plt.yticks(np.arange(0, 1+0.1, 0.1))
plt.ylim(-0.05, 1.05)
plt.xlabel('Number of angles')
plt.ylabel('Accuracy')
plt.legend(loc="lower right")
plt.show()