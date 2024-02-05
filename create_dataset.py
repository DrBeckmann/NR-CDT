import numpy as np
from utilities import gen_dataset


templates = np.load('templates.npy')
label = np.arange(len(templates))
#
templates = np.array([templates[1], templates[7]])
label = np.array([label[1], label[7]])
#
size = 100
random_seed = 42
# Translation
# parameters = [(1,1),(-0.,0.),(-0.,0.),(-10,10),(-25,25),(4,20,2,5)] # with noise
# parameters = [(1,1),(-0.,0.),(-0.,0.),(-10,10),(-25,25)] # without noise
# Scaling, translation
# parameters = [(0.75,1.1),(-0.,0.),(-0.,0.),(-10,10),(-25,25),(4,20,2,5)] # with noise
# parameters = [(0.75,1.1),(-0.,0.),(-0.,0.),(-10,10),(-25,25)] # without noise
# Scaling, rotation, translation
# parameters = [(0.5,1),(-5.,5.),(-0.,0.),(-10,10),(-10,10),(10,25,3,9)] # with noise
parameters = [(0.5,1),(-5.,5.),(-0.,0.),(-10,10),(-10,10)] # without noise
# Scaling, rotation, shear, translation
# parameters = [(0.5,1),(-5.,5.),(-0.25,0.25),(-10,10),(-10,10),(10,25,3,9)] # with noise
# parameters = [(0.5,1),(-5.,5.),(-0.25,0.25),(-10,10),(-10,10)] # without noise
dataset, labels =  gen_dataset(templates, label, size, parameters, random_seed)
np.save('data.npy',dataset)
np.save('labels.npy',labels)