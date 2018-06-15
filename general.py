"""Training the general scheme."""

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow

from util import Logger
from mayo_util import FileLoader, DATA_FOLDER
from tomo_problem import get_operators

import datetime
import sys
import os

from adler.tensorflow.training import cosine_decay

np.random.seed(0)

# Set up save-paths and output file
save_path = 'Give save path'
time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
file_name = os.path.splitext(os.path.basename(__file__))[0]
output_filename = (save_path + 'Output_' + file_name + '_' +
                   time_str + '.txt')
sys.stdout = Logger(output_filename)  # Creates logger that writes to file


# Start the tensorflow session. Limit GPU memory to 90%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))


# Create ODL data structures
size = 512
space = odl.uniform_discr([-128, -128], [128, 128], [size, size],
                          dtype='float32', weighting=1.0)

angle_partition = odl.uniform_partition(0, 2 * np.pi, 1000)
detector_partition = odl.uniform_partition(-360, 360, 1000)
geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                    src_radius=500, det_radius=500)


# Get operators that define the problem
(odl_op_layer, odl_op_layer_adjoint, odl_grad0_layer,
 odl_grad0_layer_adjoint, odl_grad1_layer, odl_grad1_layer_adjoint, operator,
 pseudoinverse) = get_operators(space, geometry)


# -----------------------------------------------------------------------------
# User selected paramters
# -----------------------------------------------------------------------------
alpha = 0.0045
beta = 1.0

n_data = 1

n_iter = 10
lowest_n_iter = 8
log_normal_std = 1.25
up_lim_iter = 100

n0 = 1  # Number primal variable to evaluate operators on
n = 3  # Total number primal variables
m0 = 1  # Number dual variable to evaluate operators on
m = 3  # Total number dual variables

print('n0={}, n={}, m0={}, m={}'.format(n0, n, m0, m))
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# To generate training data
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

        x_arr[i, ..., 0] = np.zeros_like(phantom)
        x_true_arr[i, ..., 0] = phantom
        y_arr[i, ..., 0] = noisy_data

    return x_arr, y_arr, x_true_arr
# -----------------------------------------------------------------------------


# Define tensorflow place-holders for initial guess, data, and true phantom
with tf.name_scope('place_holders'):
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

    x_0 = tf.placeholder(tf.float32, shape=[None, size, size, 1],
                         name="x_0")
    x_true = tf.placeholder(tf.float32, shape=[None, size, size, 1],
                            name="x_true")
    g = tf.placeholder(tf.float32, shape=[None, operator.range.shape[0],
                       operator.range.shape[1], 1], name="y")

    imax = tf.placeholder(tf.int32, name="imax")


# Define the variables
with tf.name_scope('variable_definitions'):
    # Defining initial values
    std = 1e-8
    theta = 0.5
    sig_tau = 1.0/np.sqrt(2)

    # Note that the prox for the TV-term is indep. of the step length sigma_one
    # Therefore, this variable is never initialized
    sigma_two = tf.Variable(tf.constant(sig_tau, dtype=tf.float32) +
                            tf.truncated_normal([1], stddev=std),
                            name='sigma_two', dtype=tf.float32)

    # Initialization corresponds to Chambolle-Pock algorithm
    if n == 3 and m == 3:
        A = tf.Variable(tf.constant([[1+theta, 0, 1], [0, 0, 0], [-theta, 0, 0]], dtype=tf.float32) +
                        tf.truncated_normal([n, n], stddev=std), name='A')
        B1 = tf.Variable(tf.constant([[sig_tau, 0, 0], [0, 0, 0], [1, 0, 1]], dtype=tf.float32) +
                         tf.truncated_normal([n, n], stddev=std), name='B1')
        B2 = tf.Variable(tf.constant([[sig_tau, 0, 0],  [0, 0, 0], [1, 0, 1]], dtype=tf.float32) +
                         tf.truncated_normal([n, n], stddev=std), name='B2')

        C1 = tf.Variable(tf.constant([[1, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=tf.float32) +
                         tf.truncated_normal([n, n], stddev=std), name='C1')
        C2 = tf.Variable(tf.constant([[1, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=tf.float32) +
                         tf.truncated_normal([n, n], stddev=std), name='C2')
        D = tf.Variable(tf.constant([[-sig_tau, 0, 0], [0, 0, 0], [1, 0, 1]], dtype=tf.float32) +
                        tf.truncated_normal([n, n], stddev=std), name='D')

    elif n == 2 and m == 2:
        A = tf.Variable(tf.constant([[1+theta, 1], [-theta, 0]], dtype=tf.float32) +
                        tf.truncated_normal([n, n], stddev=std), name='A')
        B1 = tf.Variable(tf.constant([[sig_tau, 0], [1, 1]], dtype=tf.float32) +
                         tf.truncated_normal([n, n], stddev=std), name='B1')
        B2 = tf.Variable(tf.constant([[sig_tau, 0], [1, 1]], dtype=tf.float32) +
                         tf.truncated_normal([n, n], stddev=std), name='B2')

        C1 = tf.Variable(tf.constant([[1, 1], [0, 0]], dtype=tf.float32) +
                         tf.truncated_normal([n, n], stddev=std), name='C1')
        C2 = tf.Variable(tf.constant([[1, 1], [0, 0]], dtype=tf.float32) +
                         tf.truncated_normal([n, n], stddev=std), name='C2')
        D = tf.Variable(tf.constant([[-sig_tau, 0], [1, 1]], dtype=tf.float32) +
                        tf.truncated_normal([n, n], stddev=std), name='D')

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


# -----------------------------------------------------------------------------
# This is the optimization algorithm. The initialization is to make it CP
# -----------------------------------------------------------------------------
x = x_0
x_vec = tf.concat([x_0] + [x_0] * (n-1), axis=-1)

v_1_0 = tf.concat([tf.zeros_like(x_0)] * m, axis=-1)
v_1_1 = tf.concat([tf.zeros_like(x_0)] * m, axis=-1)

v_2 = tf.concat([tf.zeros_like(g)]*m, axis=-1)


def cond(i, x, x_vec, v_1_0, v_1_1, v_2):
    return i < imax


def body(i, x, x_vec, v_1_0, v_1_1, v_2):

    v_1_0 = tf.concat([odl_grad0_layer(x_vec[..., 0:m0]),  v_1_0[..., m0:]], axis=-1)
    v_1_1 = tf.concat([odl_grad1_layer(x_vec[..., 0:m0]),  v_1_1[..., m0:]], axis=-1)
    v_2 = tf.concat([odl_op_layer(x_vec[..., 0:m0]),  v_2[..., m0:]], axis=-1)

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


    x_vec = tf.concat([odl_grad0_layer_adjoint(v_1_0[..., 0:m0]) +
                       odl_grad1_layer_adjoint(v_1_1[..., 0:m0]) +
                       odl_op_layer_adjoint(v_2[..., 0:m0]),
                       x_vec[..., m0:]],
                  axis=-1)

    x_vec = matmul(x_vec, D)

    x_vec = tf.concat([proxf(x_vec[..., 0:n0]),  x_vec[..., n0:]], axis=-1)

    x_vec = matmul(x_vec, A)

    x = x_vec[..., n-1:n]

    return i + 1, x, x_vec, v_1_0, v_1_1, v_2

i = tf.constant(0, dtype=tf.int32)
i, x, x_vec, v_1_0, v_1_1, v_2 = tf.while_loop(
    cond, body, [i, x, x_vec, v_1_0, v_1_1, v_2])
# -----------------------------------------------------------------------------


# Defining loss function to train against and the optimization scheme to
# train with
with tf.name_scope('loss'):
    loss = data_discrepancy(x) + regularizer(x)
    data_disc_eval = data_discrepancy(x)
    reg_eval = regularizer(x)

# Setting up the optimization
with tf.name_scope('optimizer'):
    # Sets up learning rateLearning rate - use cosine decay
    global_step = tf.Variable(0, trainable=False)
    maximum_steps = 100001
    starter_learning_rate = 1e-3
    learning_rate = cosine_decay(starter_learning_rate,
                                 global_step,
                                 maximum_steps,
                                 name='learning_rate')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Initialize an optimizer
        train_op = tf.train.AdamOptimizer(learning_rate, beta2=0.99)
        # Returns all variables that can be trained
        tvars = tf.trainable_variables()
        # Compute the gradient of 'loss' w.r.t. these variables, clip to norm 1
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
        # Use the clipped gradients in the optimizer to take a step
        optimizer = train_op.apply_gradients(zip(grads, tvars),
                                             global_step=global_step)


# Summaries
summary_path = save_path + 'tensorboard_' + file_name + '_' + time_str

if not os.path.exists(summary_path):
    os.makedirs(summary_path)

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('psnr', -10 * tf.log(loss) / tf.log(10.0))

    tf.summary.image('x_result', x)
    tf.summary.image('x_true', x_true)

    tf.summary.scalar('data_discrepancy', data_discrepancy(x))
    tf.summary.scalar('regularizer', regularizer(x))

    merged_summary = tf.summary.merge_all()
    test_summary_writer = tf.summary.FileWriter(summary_path + '/test',
                                                sess.graph)
    train_summary_writer = tf.summary.FileWriter(summary_path + '/train')


# Initialize all TF variables
sess.run(tf.global_variables_initializer())

# Add op to save and restore
saver = tf.train.Saver()


# Used to restor sessions, e.g., if crashed
if 0:
    tf_save_path = (save_path + 'tf_variables_' + file_name + '_' +
                    "Give date for run to restor")
    print('Restoring session "{}"'.format(tf_save_path))
    print('Remember to set iteration count manually to start from the restored point')
    input('Press enter to continue.')
    saver.restore(sess, tf_save_path + '/state')

    print('Starting from the following values')
    # Starting from the following values
    Ar, B1r, B2r, C1r, C2r, Dr, sigma_two_res, = sess.run(
        [A, B1, B2, C1, C2, D, sigma_two])

    print('-------------------------------------------')
    print('sigma 2={}'.format(sigma_two_res))

    print('Matrices A, B1, B2, C1, C2, and D')
    print(Ar)
    print(B1r)
    print(B2r)
    print(C1r)
    print(C2r)
    print(Dr)

else:
    tf_save_path = save_path + 'tf_variables_' + file_name + '_' + time_str
    if not os.path.exists(tf_save_path):
        os.makedirs(tf_save_path)


# Generate validation data
x_arr_validate, y_arr_validate, x_true_arr_validate = generate_data(validation=True)

for i in range(0, maximum_steps):

    if i%10 == 0:
        print('Generating new data')
        x_arr, y_arr, x_true_arr = generate_data()

    # Draw number of iterations in the optimization method
    imax_train = lowest_n_iter + int(np.round(np.random.lognormal(
                    mean=np.log(n_iter - lowest_n_iter) - log_normal_std**2/2,
                    sigma=log_normal_std)))
    imax_train = imax_train if imax_train <= up_lim_iter else up_lim_iter

    # Training step
    (_, loss_training, merged_summary_train, global_step_res,
     learning_rate_res) = sess.run(
         [optimizer, loss, merged_summary, global_step, learning_rate],
         feed_dict={x_0: x_arr,
                    x_true: x_true_arr,
                    g: y_arr,
                    imax: imax_train,
                    is_training: True})

    if i%10 == 0:
        print('Current learning rate: {}'.format(learning_rate_res))


    # If the loss for training is nan, break the loop
    if np.isnan(loss_training):
        break

    if i%100 == 99:
        print('iteration {}'.format(i))

        # Extract values and print
        Ar, B1r, B2r, C1r, C2r, Dr, sigma_two_res, = sess.run(
            [A, B1, B2, C1, C2, D, sigma_two])

        print('-------------------------------------------')
        print('sigma 2={}'.format(sigma_two_res))

        print('Matrices A, B1, B2, C1, C2, and D')
        print(Ar)
        print(B1r)
        print(B2r)
        print(C1r)
        print(C2r)
        print(Dr)

        # Validate on validation phantom
        (x_result, loss_result, data_disc_res, reg_res, _,
         merged_summary_validation) = sess.run(
             [x, loss, data_disc_eval, reg_eval, global_step, merged_summary],
             feed_dict={x_0: x_arr_validate,
                        x_true: x_true_arr_validate,
                        g: y_arr_validate,
                        imax: n_iter,
                        is_training: False})

        print('-------------------------------------------')
        print('validation loss={}, data dis={}, regularizer={}'.format(
                loss_result, data_disc_res, reg_res))
        print('-------------------------------------------')

        # If the loss for validation is nan, break the loop
        if np.isnan(loss_result):
            break

        train_summary_writer.add_summary(merged_summary_train, global_step_res)
        test_summary_writer.add_summary(merged_summary_validation, global_step_res)

    if i>0 and i%1000 == 0:
        saver.save(sess, tf_save_path + '/state')

# Extract final values and print
Ar, B1r, B2r, C1r, C2r, Dr, sigma_two_res, = sess.run(
    [A, B1, B2, C1, C2, D, sigma_two])

print('-------------------------------------------')
print('sigma 2={}'.format(sigma_two_res))

print('Matrices A, B1, B2, C1, C2, and D')
print(Ar)
print(B1r)
print(B2r)
print(C1r)
print(C2r)
print(Dr)

# Close logger file and set std.out to terminal
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
