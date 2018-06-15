"""Generalization to deconvolution problem."""

import tensorflow as tf
import numpy as np
import scipy
import odl
import odl.contrib.tensorflow
import matplotlib.pyplot as plt

from util import Logger
from deconv_problem import get_operators
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
size_one = 512
size_two = 512
# size_one = 1024
# size_two = 768

space = odl.uniform_discr([-128, -128], [128, 128], [size_one, size_two],
                          dtype='float32', weighting=1.0)

# Get operators
(odl_op_layer, odl_op_layer_adjoint, odl_grad0_layer,
 odl_grad0_layer_adjoint, odl_grad1_layer, odl_grad1_layer_adjoint,
 part_grad_0, part_grad_1, operator,
 pseudoinverse) = get_operators(space)


# -----------------------------------------------------------------------------
# User selected paramters
# -----------------------------------------------------------------------------
alpha = 0.0045
# alpha = 0.003
beta = 1.0

n_data = 1

n_iter = [1,2,3,4,5,6,7,8,9,10,11,12,100,1000]

# -----------------------------------------------------------------------------

def generate_data():
    """Generate a set of random data."""
    n_iter = n_data

    x_arr = np.empty((n_iter, space.shape[0], space.shape[1], 1), dtype='float32')
    x_true_arr = np.empty((n_iter, space.shape[0], space.shape[1], 1), dtype='float32')
    y_arr = np.empty((n_iter, operator.range.shape[0], operator.range.shape[1], 1), dtype='complex64')

    for i in range(n_iter):
        img = np.rot90(scipy.misc.ascent(), -1) / 255
        # img = np.rot90(scipy.misc.face(gray=True), -1) / 255
        phantom = space.element(img)

        data = operator(phantom)
        noisy_data = data + odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05

        x_arr[i, ..., 0] = np.zeros_like(phantom)
        x_true_arr[i, ..., 0] = phantom
        y_arr[i, ..., 0] = noisy_data

    return x_arr, y_arr, x_true_arr


# Define tensorflow place-holders for initial guess, data, and true phantom
with tf.name_scope('place_holders'):
    x_0 = tf.placeholder(tf.float32, shape=[None, size_one, size_two, 1],
                         name="x_0")
    x_true = tf.placeholder(tf.float32, shape=[None, size_one, size_two, 1],
                            name="x_true")
    g = tf.placeholder(tf.float32, shape=[None, operator.range.shape[0],
                       operator.range.shape[1], 1], name="y")

    imax = tf.placeholder(tf.int32, name="imax")


# Set the values of the parameters
with tf.name_scope('variable_definitions'):
    # Parameter values set guided by the following test
    """
    import numpy as np

    np.exp(36)/(1 + np.exp(36)) == 1
    Out[2]: False

    np.exp(37)/(1 + np.exp(37)) == 1
    Out[3]: True
    """

    par_one = tf.Variable(tf.constant(37, dtype=tf.float32),
                          name='par_one')
    par_two = tf.Variable(tf.constant(37, dtype=tf.float32),
                          name='par_two')
    par_three = tf.Variable(tf.constant(0, dtype=tf.float32),
                            name='par_three')


theta = tf.exp(par_one) / (1 + tf.exp(par_one))
sigma = tf.exp(par_two - par_three) / (1 + tf.exp(par_two))
tau = tf.exp(par_two + par_three) / (1 + tf.exp(par_two))


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


# This is the optimization algorithm
x = x_0
x_old = x_0

v_1_0 = tf.zeros_like(x_0)
v_1_1 = tf.zeros_like(x_0)

v_2 = tf.zeros_like(g)


def cond(i, x, x_old, v_1_0, v_1_1, v_2):
    return i < imax


def body(i, x, x_old, v_1_0, v_1_1, v_2):
    y = x + theta * (x - x_old)
    v_1_0, v_1_1 = one_norm_cc_prox(v_1_0 + sigma * odl_grad0_layer(y),
                                    v_1_1 + sigma * odl_grad1_layer(y))

    v_2 = two_norm_sq_cc_prox(v_2 + sigma * odl_op_layer(y),
                              sigma)

    x_old = x
    x = x - tau * (odl_grad0_layer_adjoint(v_1_0) +
                   odl_grad1_layer_adjoint(v_1_1) +
                   odl_op_layer_adjoint(v_2))

    return i + 1, x, x_old, v_1_0, v_1_1, v_2

i = tf.constant(0, dtype=tf.int32)
i, x, x_old, v_1_0, v_1_1, v_2 = tf.while_loop(
    cond, body, [i, x, x_old, v_1_0, v_1_1, v_2])


# Defining loss function
with tf.name_scope('loss'):
    loss = data_discrepancy(x) + regularizer(x)


# Initialize all TF variables
sess.run(tf.global_variables_initializer())

# Generate validation data
x_arr_validate, y_arr_validate, x_true_arr_validate = generate_data()

x_true_odl = space.element(np.squeeze(x_true_arr_validate))
# x_true_odl.show(clim=[0,1], saveto=save_path + 'true_image', colorbar=False)

fig = plt.figure()
plt.imshow(np.rot90(np.squeeze(x_true_arr_validate)), cmap='gray', clim=[0,1])
plt.xticks([])
plt.yticks([])
fig.savefig(save_path + 'true_image', transparent=True,
            bbox_inches='tight', pad_inches=0)

y_odl = space.element(np.squeeze(y_arr_validate))
# y_odl.show(clim=[0,1], saveto=save_path + 'blurred_image')

fig = plt.figure()
plt.imshow(np.rot90(np.squeeze(np.real(y_arr_validate))), cmap='gray', clim=[0,1])
plt.xticks([])
plt.yticks([])
fig.savefig(save_path + 'blurred_image', transparent=True,
            bbox_inches='tight', pad_inches=0)


# Iterate over different depths of the solver
for imax_val in n_iter:
    print('------------------------')
    print('Number of iterations={}'.format(imax_val))

    # Validate on shepp-logan
    (x_result, loss_result, data_disc_res, reg_res, sigma_res, tau_res,
        theta_res) = sess.run(
            [x, loss, data_discrepancy(x), regularizer(x), sigma, tau, theta],
            feed_dict={x_0: x_arr_validate,
                       x_true: x_true_arr_validate,
                       g: y_arr_validate,
                       imax: imax_val})

    x_rec_odl = space.element(np.squeeze(x_result))
    # x_rec_odl.show(clim=[0, 1],
    #                saveto=(save_path + 'sidky_iter_{}').format(imax_val))
    fig = plt.figure()
    plt.imshow(np.rot90(np.squeeze(x_rec_odl.asarray())), cmap='gray', clim=[0,1])
    plt.xticks([])
    plt.yticks([])
    fig.savefig((save_path + 'sidky_iter_{}').format(imax_val),
                transparent=True, bbox_inches='tight', pad_inches=0)

    mse_val = mean_squared_error(x_rec_odl, x_true_odl)
    ssim_val = ssim(x_rec_odl, x_true_odl)

    print('validation loss={}, data dis={}, regularizer={}'.format(loss_result,
          data_disc_res, reg_res))

    print('mse={}, ssim={}'.format(mse_val, ssim_val))

print('sigma={}, tau={}, theta={}, sigma*tau*||L||^2={}'.format(sigma_res,
      tau_res, theta_res, sigma_res*tau_res))

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
