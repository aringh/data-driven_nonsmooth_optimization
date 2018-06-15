"""Validation of trained parameters."""

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow

from util import Logger
from tomo_problem import get_operators

import datetime
import sys
import os

# Seed randomness for reproducability in validation
np.random.seed(0)

# Set up save-path
save_path = 'Give save path'
time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
file_name = os.path.splitext(os.path.basename(__file__))[0]
output_filename = (save_path + 'Output_' + file_name + '_' +
                   time_str + '.txt')
sys.stdout = Logger(output_filename)

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

n_iter = [1,2,3,4,5,6,7,8,9,10,11,12,15,18,21,24,27,30,33,36,39,42,45,50,55,60,70,80,90,100,150,200,250,300, 1000]

num_valid_phantoms = 100

# -----------------------------------------------------------------------------


# Define tensorflow place-holders for initial guess, data, and true phantom
with tf.name_scope('place_holders'):
    x_0 = tf.placeholder(tf.float32, shape=[None, size, size, 1],
                         name="x_0")
    x_true = tf.placeholder(tf.float32, shape=[None, size, size, 1],
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


validation_data_path = 'Give path to where the validation data is saved'
x_arr_validate = np.load(validation_data_path + 'x_arr_validate.npy')
y_arr_validate = np.load(validation_data_path + 'y_arr_validate.npy')
x_true_arr_validate = np.load(validation_data_path + 'x_true_arr_validate.npy')


loss_values = []
# Iterate over different depths of the solver
for imax_val in n_iter:
    print('------------------------')
    print('Number of iterations={}'.format(imax_val))

    total_loss = 0
    for i in range(num_valid_phantoms):
        print('Validation phantom number {}'.format(i))
        # Validate on validation phantom
        loss_result = sess.run(loss,
                               feed_dict={x_0: np.expand_dims(x_arr_validate[i,...], 0),
                                          x_true: np.expand_dims(x_true_arr_validate[i,...], 0),
                                          g: np.expand_dims(y_arr_validate[i,...], 0),
                                          imax: imax_val})

        total_loss += loss_result

    loss_values += [total_loss/num_valid_phantoms]

print('Average loss values over {} phantoms'.format(num_valid_phantoms))
print(loss_values)
print('Number of iterations in solver')
print(n_iter)

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
