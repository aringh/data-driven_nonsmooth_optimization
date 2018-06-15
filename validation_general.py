"""Validation producing reconstructions from one slice"""

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow
import matplotlib.pyplot as plt

from util import Logger
from mayo_util import FileLoader, DATA_FOLDER
from tomo_problem import get_operators
from odl.contrib.fom import mean_squared_error, ssim

import datetime
import sys
import os

# Seed randomness for reproducability in validation
np.random.seed(0)

# Set up save-paths and output file
save_path = 'Give save path'
time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
file_name = os.path.splitext(os.path.basename(__file__))[0]
output_filename = (save_path + 'Output_' + file_name + '_' +
                   time_str + '.txt')
sys.stdout = Logger(output_filename)  # Creates logger that writes to file

# Start the tensorflow session
sess = tf.InteractiveSession()


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
alpha = 0.0045
beta = 1.0

n_data = 1

n_iter = [1,2,3,4,5,6,7,8,9,10,11,12]

# -----------------------------------------------------------------------------
file_loader = FileLoader(DATA_FOLDER, exclude='L286')

def generate_data(validation=False):
    """Generate a set of random data."""
    n_iter = 1 if validation else n_data

    x_arr = np.empty((n_iter, space.shape[0], space.shape[1], 1), dtype='float32')
    x_true_arr = np.empty((n_iter, space.shape[0], space.shape[1], 1), dtype='float32')
    y_arr = np.empty((n_iter, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')

    for i in range(n_iter):
        if validation:
            fi = DATA_FOLDER + 'L286_FD_3_1.CT.0002.0201.2015.12.22.18.22.49.651226.358225786.npy'
        else:
            fi = file_loader.next_file()

        data = np.load(fi)

        phantom = space.element(np.rot90(data, -1))
        phantom /= 1000.0  # convert go g/cm^3

        data = operator(phantom)
        noisy_data = data + odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05

        #data = np.exp(-data * mu_water)
        #noisy_data = odl.phantom.poisson_noise(data * photons_per_pixel) / photons_per_pixel

        x_arr[i, ..., 0] = np.zeros_like(phantom) #  pseudoinverse(noisy_data)
        x_true_arr[i, ..., 0] = phantom
        y_arr[i, ..., 0] = noisy_data

    return x_arr, y_arr, x_true_arr


# Define tensorflow place-holders for initial guess, data, and true phantom
with tf.name_scope('place_holders'):
    x_0 = tf.placeholder(tf.float32, shape=[None, size, size, 1],
                         name="x_0")
    x_true = tf.placeholder(tf.float32, shape=[None, size, size, 1],
                            name="x_true")
    g = tf.placeholder(tf.float32, shape=[None, operator.range.shape[0],
                       operator.range.shape[1], 1], name="y")

    imax = tf.placeholder(tf.int32, name="imax")


# -----------------------------------------------------------------------------
# Parameters for defining the optimization solver
# -----------------------------------------------------------------------------
n0 = 1
n = 2
m0 = 1
m = 2

print('n0={}, n={}, m0={}, m={}'.format(n0, n, m0, m))
# -----------------------------------------------------------------------------


# Set the values of the trained variables
with tf.name_scope('variable_definitions'):


    if n == 3 and m == 3:
        A = tf.Variable(tf.constant([[1.86463928, 0.00300482, 0.86318403],
                                     [0.7708801, 0.45403832, -0.14642699],
                                     [-0.32346275, -0.25273058, -0.03792271]],
                                    dtype=tf.float32), name='A')
        B1 = tf.Variable(tf.constant([[1.0921663, -0.37221879, 1.27031374],
                                      [0.7241298, -0.84452474, -0.63437283],
                                      [1.73737991, 0.48538038, 0.12716696]],
                                     dtype=tf.float32), name='B1')
        B2 = tf.Variable(tf.constant([[0.61043346, 0.34826526, 0.41468441],
                                      [0.2862702, 1.08472383, -0.0722857 ],
                                      [1.38410306, 0.51068896, 1.35045254]],
                                     dtype=tf.float32), name='B2')

        C1 = tf.Variable(tf.constant([[1.52877188, -1.11131012, 0.93177617],
                                      [0.83692318, -0.46301287, 0.41224441],
                                      [-0.09206133, -0.37906235, -0.14927569]],
                                     dtype=tf.float32), name='C1')
        C2 = tf.Variable(tf.constant([[1.69956744, 0.41451526, 1.3164885],
                                      [0.63600004, -0.26314148, -0.1640159],
                                      [0.62707186, 0.4018631, -0.29943225]],
                                     dtype=tf.float32), name='C2')
        D = tf.Variable(tf.constant([[-1.03958809, -0.84732455, 0.68730778],
                                     [-0.31452525, 0.11265664, -0.10849705],
                                     [1.17189109, 0.22176336, 0.67002881]],
                                    dtype=tf.float32), name='D')

        # Note that the prox for the TV-term is indep. of the step length sigma_one
        # Therefore, this variable is never initialized
        sigma_two = tf.Variable(tf.constant(1.07939565, dtype=tf.float32),
                                name='sigma_two', dtype=tf.float32)
        #"""


    elif n == 2 and m == 2:
        A = tf.Variable(tf.constant([[1.73995507, 0.97327393],
                                     [-0.1740977, -0.28461453]],
                                    dtype=tf.float32), name='A')
        B1 = tf.Variable(tf.constant([[0.9517206, 0.56561661],
                                      [2.29460692, 1.21984196]],
                                     dtype=tf.float32), name='B1')
        B2 = tf.Variable(tf.constant([[0.64494759, 0.69092709],
                                      [1.31639469, 1.04333568]],
                                     dtype=tf.float32), name='B2')

        C1 = tf.Variable(tf.constant([[1.71106815, 0.66143233],
                                      [-0.23952611, -0.7074939]],
                                     dtype=tf.float32), name='C1')
        C2 = tf.Variable(tf.constant([[1.73355091, 1.03382146],
                                      [0.67602092, -0.38309783]],
                                     dtype=tf.float32), name='C2')
        D = tf.Variable(tf.constant([[-1.33146405, -0.01882807],
                                     [1.21085811, 0.60968941]],
                                    dtype=tf.float32), name='D')

        # Note that the prox for the TV-term is indep. of the step length sigma_one
        # Therefore, this variable is never initialized
        sigma_two = tf.Variable(tf.constant(0.93228781, dtype=tf.float32),
                                name='sigma_two', dtype=tf.float32)

    else:
        raise Exception('Only implemented for n = m = 2 or = 3')

# Setting up functionals
def two_norm_sq(x):
    with tf.name_scope('l2_norm_sq'):
        return tf.reduce_sum(x**2, axis=(1, 2))


def data_discrepancy(x):  # The squared l2-norm of mismatch
    with tf.name_scope('data_disc_func'):
        return beta * tf.reduce_mean(two_norm_sq(odl_op_layer(x) - g))


def regularizer(x):  # The 2-1 norm of the gradient
    with tf.name_scope('reg_func'):
        two_norm_grad_x = tf.reduce_sum(tf.sqrt(
            odl_grad0_layer(x)**2 + odl_grad1_layer(x)**2 + 1e-10),
            axis=(1, 2))
        return alpha * tf.reduce_mean(two_norm_grad_x)


def two_norm_sq_cc_prox(x, sigma):
    with tf.name_scope('l2_norm_sq_cc_prox'):
        return (x - sigma*g) / (1.0 + sigma/(2.0*beta))


def one_norm_cc_prox(x0, x1):
    with tf.name_scope('l1_norm_cc_prox'):
        tmp = tf.sqrt(x0**2 + x1**2 + 1e-10)
        d = tf.maximum(tmp, alpha)
        return alpha * x0 / d, alpha * x1 / d


def proxf(x):
    return x


def matmul(x, v):
    return tf.nn.conv2d(x, v[None, None, ...], strides=[1, 1, 1, 1], padding='SAME')


# This is the optimization algorithm. The initialization is to make it CP
x = x_0
x_vec = tf.concat([x_0] + [x_0] * (n-1), axis=-1)

v_1_0 = tf.concat([tf.zeros_like(x_0)] * m, axis=-1)
v_1_1 = tf.concat([tf.zeros_like(x_0)] * m, axis=-1)

v_2 = tf.concat([tf.zeros_like(g)]*m, axis=-1)


def cond(i, x_loop, x_loop_vec, v_1_0, v_1_1, v_2):
    return i < imax


def body(i, x_loop, x_loop_vec, v_1_0, v_1_1, v_2):

    v_1_0 = tf.concat([odl_grad0_layer(x_loop_vec[..., 0:m0]),  v_1_0[..., m0:]], axis=-1)
    v_1_1 = tf.concat([odl_grad1_layer(x_loop_vec[..., 0:m0]),  v_1_1[..., m0:]], axis=-1)
    v_2 = tf.concat([odl_op_layer(x_loop_vec[..., 0:m0]),  v_2[..., m0:]], axis=-1)

    v_1_0 = matmul(v_1_0, B1)
    v_1_1 = matmul(v_1_1, B1)
    v_2 = matmul(v_2, B2)

    tmp0, tmp1 = one_norm_cc_prox(v_1_0[..., 0:m0], v_1_1[..., 0:m0])
    v_1_0 = tf.concat([tmp0,  v_1_0[..., m0:]], axis=-1)
    v_1_1 = tf.concat([tmp1,  v_1_1[..., m0:]], axis=-1)

    v_2 = tf.concat([two_norm_sq_cc_prox(v_2[..., 0:m0], sigma_two),  v_2[..., m0:]], axis=-1)

    v_1_0 = matmul(v_1_0, C1)
    v_1_1 = matmul(v_1_1, C1)
    v_2 = matmul(v_2, C2)


    x_loop_vec = tf.concat([odl_grad0_layer_adjoint(v_1_0[..., 0:m0]) +
                       odl_grad1_layer_adjoint(v_1_1[..., 0:m0]) +
                       odl_op_layer_adjoint(v_2[..., 0:m0]),
                       x_loop_vec[..., m0:]],
                  axis=-1)

    x_loop_vec = matmul(x_loop_vec, D)

    x_loop_vec = tf.concat([proxf(x_loop_vec[..., 0:n0]),  x_loop_vec[..., n0:]], axis=-1)

    x_loop_vec = matmul(x_loop_vec, A)

    x_loop = x_loop_vec[..., n-1:n]

    return i + 1, x_loop, x_loop_vec, v_1_0, v_1_1, v_2

i = tf.constant(0, dtype=tf.int32)
i, x, x_vec, v_1_0, v_1_1, v_2 = tf.while_loop(
    cond, body, [i, x, x_vec, v_1_0, v_1_1, v_2])


# Defining loss function
with tf.name_scope('loss'):
    loss = data_discrepancy(x) + regularizer(x)

# Initialize all TF variables
sess.run(tf.global_variables_initializer())


# Generate validation data
x_arr_validate, y_arr_validate, x_true_arr_validate = generate_data(validation=True)

x_true_odl = space.element(np.squeeze(x_true_arr_validate))
x_true_odl.show()

# Iterate over different depths of the solver
for imax_val in n_iter:
    print('------------------------')
    print('Number of iterations={}'.format(imax_val))

    # Validate on validation phantom
    (x_result, loss_result, data_disc_res, reg_res) = sess.run(
            [x, loss, data_discrepancy(x), regularizer(x)],
            feed_dict={x_0: x_arr_validate,
                       x_true: x_true_arr_validate,
                       g: y_arr_validate,
                       imax: imax_val})

    x_rec_odl = space.element(np.squeeze(x_result))
    # x_rec_odl.show(clim=[0, 2.33],
    #                saveto=(save_path + 'general_nm_{}_iter_{}').format(n, imax_val))

    fig = plt.figure()
    plt.imshow(np.rot90(x_rec_odl.asarray()), cmap='bone', clim=[0, 2.33])
    plt.xticks([])
    plt.yticks([])
    fig.savefig((save_path + 'general_nm_{}_iter_{}').format(n, imax_val),
                transparent=True, bbox_inches='tight', pad_inches=0)

    fig = plt.figure()
    plt.imshow(np.rot90(x_rec_odl.asarray()), cmap='bone', clim=[0.8, 1.2])
    plt.xticks([])
    plt.yticks([])
    fig.savefig((save_path + 'general_nm_{}_iter_{}_windowed').format(n, imax_val),
                transparent=True, bbox_inches='tight', pad_inches=0)

    mse_val = mean_squared_error(x_rec_odl, x_true_odl)
    ssim_val = ssim(x_rec_odl, x_true_odl)

    print('validation loss={}, data dis={}, regularizer={}'.format(loss_result,
          data_disc_res, reg_res))

    print('mse={}, ssim={}'.format(mse_val, ssim_val))

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
