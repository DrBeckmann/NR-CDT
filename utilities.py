import numpy as np
import odl
from scipy import ndimage
from skimage import transform


def random_mask(shape, num_pos_min, num_pos_max, width, length_max, seed):
    np.random.seed(seed)
    mask = np.zeros(shape)
    num_pos = np.random.randint(num_pos_min, num_pos_max + 1)
    for i in range(num_pos):
        row_pos = np.random.randint(0, shape[0])
        col_pos = np.random.randint(0, shape[1])
        length = np.random.randint(1, length_max + 1)
        case = np.random.randint(0, 4)
        if case == 0:
            # vertical
            mask[row_pos: row_pos + length, col_pos: col_pos + width] = 1
        elif case == 1:
            # horizontal
            mask[row_pos: row_pos + width, col_pos: col_pos + length] = 1
        elif case == 2:
            # diagonal down
            for i in range(length): 
                try:
                    mask[row_pos + i, col_pos + i*int(width/2): col_pos + i*int(width/2) + width] = 1
                except:
                    break
        elif case == 3:
            # diagonal up
            for i in range(length): 
                try:
                    mask[row_pos - i, col_pos + i*int(width/2): col_pos + i*int(width/2) + width] = 1
                except:
                    break
    
    return mask


def random_image_distortion(image, scale_bounds, angle_bounds, shear_bounds, shift_bounds_x, shift_bounds_y,
                           noise_bounds, seed):
    np.random.seed(seed)
    # Scale
    if scale_bounds[1] > scale_bounds[0]:
        scale = round(np.random.uniform(scale_bounds[0], scale_bounds[1]), 2)
        img = transform.rescale(image, scale, preserve_range=True)
        if scale <= 1.0:
            image = np.zeros((image.shape[0],image.shape[0]))
            a = round((image.shape[0] - img.shape[0])/2)
            image[a:img.shape[0]+a,a:img.shape[1]+a] = img
        else:
            image = np.zeros((image.shape[0], image.shape[0]))
            x, y = np.asarray(img.shape)//2
            a = image.shape[0]//2
            image[:,:] = img[x -a:x+a, y-a:y+a]
    # Rotate
    if angle_bounds[1] > angle_bounds[0]:
        angle = round(np.random.uniform(angle_bounds[0], angle_bounds[1]))
        image = ndimage.rotate(image, angle, reshape=False, mode='nearest', prefilter=False)
    # Shear
    if shear_bounds[1] > shear_bounds[0]:
        shear_tf = transform.AffineTransform(shear=np.random.uniform(shear_bounds[0],shear_bounds[1]))
        image = transform.warp(image, inverse_map=shear_tf)
    # Shift
    if shift_bounds_x[1] > shift_bounds_x[0]:
        shift_x = np.random.randint(shift_bounds_x[0], shift_bounds_x[1])
    else:
        shift_x = 0
    if shift_bounds_y[1] > shift_bounds_y[0]:
        shift_y = np.random.randint(shift_bounds_y[0], shift_bounds_y[1])
    else:
        shift_y = 0
    if shift_x != 0 or shift_y != 0:
        image = ndimage.shift(image, (shift_x, shift_y))
    # Binarize
    image[image < 64] = 0
    image[image >= 64] = 255
    # Noise
    if noise_bounds != 0:
        mask = random_mask(np.shape(image), noise_bounds[0], noise_bounds[1], noise_bounds[2], noise_bounds[3], seed)
        image[mask==1] = 255
    
    return image


def gen_dataset(template, label, size, parameters, seed):
    scale_bounds = parameters[0]
    angle_bounds = parameters[1]
    shear_bounds = parameters[2]
    shift_bounds_x = parameters[3]
    shift_bounds_y = parameters[4]
    try:
        noise_bounds = parameters[5]
    except:
        noise_bounds = 0
    dataset = list()
    labels = list()
    for i in range(size):
        for j in range(len(template)):
            img = random_image_distortion(template[j], scale_bounds, angle_bounds, shear_bounds,
                                          shift_bounds_x, shift_bounds_y, noise_bounds, seed+i)
            dataset.append(img)
            labels.append(label[j])
    dataset = np.asarray(dataset)
    labels = np.asarray(labels)
    
    return dataset, labels


def signal_to_pdf(input, epsilon=1e-7):
    if epsilon <= 0:
        raise ValueError('epsilon must be > 0.')
    pdf = input/sum(input)
    pdf += epsilon
    pdf /= sum(pdf)

    return pdf


def cdt(x0, s0, x1, s1):
    s0 = signal_to_pdf(s0)
    s1 = signal_to_pdf(s1)
    r = np.min(abs(x0 - np.roll(x0, 1)))
    cum0 = np.cumsum(s0)
    cum1 = np.cumsum(s1)
    if len(np.unique(s0)) == 1:
        s_hat = np.interp(x0, cum1, x1)
    else:
        s_hat = np.interp(r * cum0, r * cum1, x1)

    return s_hat


class Radon:
    def __init__(self, im_shape, num_angles):
        self.im_shape = im_shape
        self.num_angles = num_angles
        apart = odl.uniform_partition(0, np.pi, num_angles)
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


def prepare_data(data, num_angles):
    ref = np.ones(data[0].shape)
    data_transformed = list()
    for i in range(len(data)):
        data_transformed.append(rcdt(ref, data[i], num_angles)[0])
    data_transformed = np.asarray(data_transformed)
    nsamples, nx, ny = data_transformed.shape
    data_transformed = data_transformed.reshape((nsamples, nx * ny))

    return data_transformed


def shuffle_data(data, labels, random_seed):
    rng = np.random.default_rng(random_seed)
    ind = np.arange(data.shape[0])
    rng.shuffle(ind)
    data = data[ind]
    labels = labels[ind]
    
    return data, labels


def split_data(data, labels, split):
    lab = np.unique(labels)
    train_ind = list()
    test_ind = list()
    for l in lab:
        ind = np.nonzero(labels == l)[0].tolist()
        ind_split = round(len(ind)*split)
        train_ind += ind[:ind_split]
        test_ind += ind[ind_split:]
    train_ind.sort()
    test_ind.sort()
    data_train = data[train_ind]
    labels_train = labels[train_ind]
    data_test = data[test_ind]
    labels_test = labels[test_ind]

    return data_train, labels_train, data_test, labels_test