"""Script to create validation data."""

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow
import matplotlib.pyplot as plt

from mayo_util_batch_valid import FileLoader, VALID_DATA_FOLDER
from tomo_problem import get_operators

import os

# Seed randomness for reproducability in validation
np.random.seed(0)

# Set up save-path
save_path = 'Give save path'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Create ODL data structures
size = 512
space = odl.uniform_discr([-128, -128], [128, 128], [size, size],
                          dtype='float32', weighting=1.0)

angle_partition = odl.uniform_partition(0, 2 * np.pi, 1000)
detector_partition = odl.uniform_partition(-360, 360, 1000)
geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                    src_radius=500, det_radius=500)


# Get operators
(odl_op_layer, odl_op_layer_adjoint, odl_grad0_layer,
 odl_grad0_layer_adjoint, odl_grad1_layer, odl_grad1_layer_adjoint, operator,
 pseudoinverse) = get_operators(space, geometry)


# -----------------------------------------------------------------------------
# User selected paramters
# -----------------------------------------------------------------------------
num_valid_phantoms = 100  # there are 210 slices for L286
# -----------------------------------------------------------------------------

batch_valid_loader = FileLoader(VALID_DATA_FOLDER)

def generate_data():
    """Generate a set of random data."""
    n_iter = num_valid_phantoms

    x_arr = np.empty((n_iter, space.shape[0], space.shape[1], 1), dtype='float32')
    x_true_arr = np.empty((n_iter, space.shape[0], space.shape[1], 1), dtype='float32')
    y_arr = np.empty((n_iter, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')

    for i in range(n_iter):
        fi = batch_valid_loader.next_file()

        data = np.load(fi)

        phantom = space.element(np.rot90(data, -1))
        phantom /= 1000.0  # convert go g/cm^3

        data = operator(phantom)
        noisy_data = data + odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05

        x_arr[i, ..., 0] = np.zeros_like(phantom)
        x_true_arr[i, ..., 0] = phantom
        y_arr[i, ..., 0] = noisy_data

    return x_arr, y_arr, x_true_arr

# Create the data
x_arr_validate, y_arr_validate, x_true_arr_validate = generate_data()


# Plot three slices
for i in range(3):
    x_current = np.squeeze(x_true_arr_validate[i])
    print('Max pixel value: {}'.format(np.max(x_current)))
    fig = plt.figure()
    plt.imshow(np.rot90(x_current), cmap='bone')
    plt.xticks([])
    plt.yticks([])
    fig.savefig((save_path + 'phantom_{}').format(i),
                transparent=True, bbox_inches='tight', pad_inches=0)

    fig = plt.figure()
    plt.imshow(np.rot90(x_current), cmap='bone', clim=[0.8, 1.2])
    plt.xticks([])
    plt.yticks([])
    fig.savefig((save_path + 'phantom_{}_windowed').format(i),
                transparent=True, bbox_inches='tight', pad_inches=0)

# Save data
np.save(save_path + 'x_arr_validate.npy', x_arr_validate)
np.save(save_path + 'y_arr_validate.npy', y_arr_validate)
np.save(save_path + 'x_true_arr_validate.npy', x_true_arr_validate)
