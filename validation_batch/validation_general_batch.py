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

n_iter = [1,2,3,4,5,6,7,8,9,10,11,12,15,18,21,24,27,30,33,36,39,42,45,50,55,60,70,80,90,100,150,200,250,300]

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


# To make it easier changing the optimization problem later
def proxf(x):
    return x


# Trick to implement the matrix-tensor multiplication
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
