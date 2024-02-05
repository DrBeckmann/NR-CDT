import numpy as np
import odl
import matplotlib.pyplot as plt
from utilities import gen_dataset, shuffle_data, split_data, prepare_data, cdt
from sklearn import svm


class Radon:
    def __init__(self, im_shape, num_angles):
        self.im_shape = im_shape
        self.num_angles = num_angles
        apart = odl.uniform_partition(-np.pi, np.pi, num_angles)
        dpart = odl.uniform_partition(-int(np.ceil(np.sqrt(2 * im_shape[0] ** 2) / 2)),
                                      int(np.ceil(np.sqrt(2 * im_shape[0] ** 2) / 2)),
                                      int(np.ceil(np.sqrt(2 * im_shape[0] ** 2))))
        geometry = odl.tomo.geometry.parallel.Parallel2dGeometry(apart, dpart)
        reco_space = odl.uniform_discr(min_pt= [-im_shape[0] / 2,-im_shape[0] / 2],
                                       max_pt= [im_shape[1] / 2, im_shape[1] / 2], shape=im_shape, dtype='float64')
        self.forward = odl.tomo.operators.ray_trafo.RayTransform(reco_space, geometry, impl='astra_cpu')
        self.inverse = odl.tomo.analytic.filtered_back_projection.fbp_op(self.forward)


def rcdt(ref, tar, num_angles):
    radon = Radon(np.shape(tar), num_angles).forward
    tar_radon = radon(tar)
    if len(np.unique(ref)) == 1:
        ref_radon = np.ones(tar_radon.shape)
    else:
        radon_ref = Radon(np.shape(ref), num_angles).forward
        ref_radon = radon_ref(ref)
    tar_rcdt = np.zeros(ref_radon.shape)
    x_ref = np.linspace(0, 1, np.shape(ref_radon)[1])
    x_tar = np.linspace(0, 1, np.shape(tar_radon)[1])
    for i in range(np.shape(ref_radon)[0]):
        tar_rcdt[i, :] = cdt(x_ref, ref_radon[i, :], x_tar, tar_radon[i, :])

    return tar_rcdt, tar_radon


templates = np.load('templates.npy')
label = np.arange(len(templates))
templates = np.array([templates[0], templates[1]])
label = np.array([label[0], label[1]])
#
templates_ext = np.zeros((len(templates),256,256))
for i in range(len(templates)):
    image = np.zeros((256,256))
    a = round((256 - templates[i].shape[0])/2)
    b = round((256 - templates[i].shape[1])/2)
    image[a:templates[i].shape[0]+a,b:templates[i].shape[1]+b] = templates[i]
    templates_ext[i] = image
templates = templates_ext
plt.gray()
fig, ax = plt.subplots(1, 2, figsize=(5,10))
for i in range(len(templates)):
    ax[i].imshow(templates[i])
    ax[i].set_axis_off()
fig.tight_layout()
plt.show()
#
random_seed = 42 #42, 1337, 0, 2024
size = 100
parameters = [(0.75,1.25),(-180.,180.),(-0.5,0.5),(-10,10),(-10,10)] #noise: (4,20,2,5), (10,40,2,5)
data, labels =  gen_dataset(templates, label, size, parameters, random_seed)
#
# np.random.seed(42)
sel = np.random.choice(len(data), 9, replace=False)
plt.gray()
fig, ax = plt.subplots(3, 3, figsize=(15,15))
for i in range(3):
    for j in range(3):
        ax[i, j].imshow(data[sel[3*i+j]])
        ax[i, j].set_axis_off()
fig.tight_layout()
plt.show()
#
num_angles = 36
ref = np.ones(data[0].shape)
templates_rcdt = list()
for i in range(len(templates)):
    templates_rcdt.append(rcdt(ref, templates[i], num_angles)[0].transpose())
templates_rcdt = np.asarray(templates_rcdt)
data_rcdt = list()
for i in range(len(data)):
    data_rcdt.append(rcdt(ref, data[i], num_angles)[0].transpose())
data_rcdt = np.asarray(data_rcdt)
#
templates_rcdt_normalized = (templates_rcdt - np.mean(templates_rcdt, axis=1, keepdims=True))/np.sqrt(np.var(templates_rcdt, axis=1, keepdims=True))
templates_rcdt_normalized = np.max(templates_rcdt_normalized, axis=2)
# templates_rcdt_normalized = templates_rcdt_normalized[:,:,0]
data_rcdt_normalized = (data_rcdt - np.mean(data_rcdt, axis=1, keepdims=True))/np.sqrt(np.var(data_rcdt, axis=1, keepdims=True))
data_rcdt_normalized = np.max(data_rcdt_normalized, axis=2)
# data_rcdt_normalized = data_rcdt_normalized[:,:,0]
#
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
for i in range(len(data)):
    if labels[i] == label[0]:
        ax[0, 0].plot(data_rcdt[i,:,0])
        ax[0, 1].plot(data_rcdt_normalized[i,:])
    elif labels[i] == label[1]:
        ax[1, 0].plot(data_rcdt[i,:,0])
        ax[1, 1].plot(data_rcdt_normalized[i,:])
ax[0, 0].plot(templates_rcdt[0,:,0], 'k')
ax[0, 1].plot(templates_rcdt_normalized[0,:], 'k')
ax[1, 0].plot(templates_rcdt[1,:,0], 'k')
ax[1, 1].plot(templates_rcdt_normalized[1,:], 'k')
plt.show()
#
data, labels = shuffle_data(data, labels, random_seed)
split = .05
train_data, train_labels, test_data, test_labels = split_data(data, labels, split)
#
clf_euclid = svm.SVC(kernel = 'linear')
train_data_reshaped = train_data.reshape((train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
clf_euclid.fit(train_data_reshaped, train_labels)
test_data_reshaped = test_data.reshape((test_data.shape[0], test_data.shape[1] * test_data.shape[2]))
pred_euclid = clf_euclid.predict(test_data_reshaped)
acc_euclid = np.count_nonzero(test_labels == pred_euclid)/len(test_labels)
print(acc_euclid)
#
num = 5
acc_rcdt = list()
for i in range(num):
    clf_rcdt = svm.SVC(kernel = 'linear', C = 1e10)
    train_data_transformed = prepare_data(train_data, num_angles=i+1)
    clf_rcdt.fit(train_data_transformed, train_labels)
    test_data_transformed = prepare_data(test_data, num_angles=i+1)
    pred_rcdt = clf_rcdt.predict(test_data_transformed)
    acc_rcdt.append(np.count_nonzero(test_labels == pred_rcdt)/len(test_labels))
    print(acc_rcdt)
#
num_angles = 180
clf_rcdt_normalized = svm.SVC(kernel = 'linear', C = 1e10)
ref = np.ones(train_data[0].shape)
train_data_rcdt = list()
for i in range(len(train_data)):
    train_data_rcdt.append(rcdt(ref, train_data[i], num_angles)[0].transpose())
train_data_rcdt = np.asarray(train_data_rcdt)
train_data_rcdt_normalized = (train_data_rcdt - np.mean(train_data_rcdt, axis=1, keepdims=True))/np.sqrt(np.var(train_data_rcdt, axis=1, keepdims=True))
train_data_rcdt_normalized = np.max(train_data_rcdt_normalized, axis=2)
clf_rcdt_normalized.fit(train_data_rcdt_normalized, train_labels)
test_data_rcdt = list()
for i in range(len(test_data)):
    test_data_rcdt.append(rcdt(ref, test_data[i], num_angles)[0].transpose())
test_data_rcdt = np.asarray(test_data_rcdt)
test_data_rcdt_normalized = (test_data_rcdt - np.mean(test_data_rcdt, axis=1, keepdims=True))/np.sqrt(np.var(test_data_rcdt, axis=1, keepdims=True))
test_data_rcdt_normalized = np.max(test_data_rcdt_normalized, axis=2)
pred_rcdt_normalized = clf_rcdt_normalized.predict(test_data_rcdt_normalized)
acc_rcdt_normalized = np.count_nonzero(test_labels == pred_rcdt_normalized)/len(test_labels)
print(acc_rcdt_normalized)
# np.save('data_acc_rcdt.npy',acc_rcdt)
# np.save('data_acc_rcdt_normalized.npy',acc_rcdt_normalized)
# np.save('data_acc_euclid.npy',acc_euclid)
#
X = np.arange(1, num+1, 1)
plt.figure()
plt.plot(X, acc_rcdt, color='red', label='R-CDT')
plt.plot(X, acc_rcdt_normalized*np.ones(num), color='green', label='Normalized R-CDT')
plt.plot(X, acc_euclid*np.ones(num), color='blue', label='Euclidean')
plt.xticks(np.arange(1, num+1, 1))
plt.yticks(np.arange(0, 1+0.1, 0.1))
plt.ylim(-0.05, 1.05)
plt.xlabel('Number of angles')
plt.ylabel('Accuracy')
plt.legend(loc="lower right")
plt.show()