"""Helper function for setting up the deconvolution problem."""

import tensorflow as tf
import numpy as np
import scipy.signal
import odl
import odl.contrib.tensorflow


def get_operators(space):
    # Create the forward operator
    filter_width = 4  # standard deviation of the Gaussian filter
    ft = odl.trafos.FourierTransform(space)
    c = filter_width ** 2 / 4.0 ** 2
    gaussian = ft.range.element(lambda x: np.exp(-(x[0] ** 2 + x[1] ** 2) * c))
    operator = ft.inverse * gaussian * ft

    # Normalize the operator and create pseudo-inverse
    opnorm = odl.power_method_opnorm(operator)
    operator = (1 / opnorm) * operator

    # Do not need good pseudo-inverse, but keep to have same interface.
    pseudoinverse = odl.ZeroOperator(space)

    # Create gradient operator and normalize it
    part_grad_0 = odl.PartialDerivative(space, 0, method='forward',
                                        pad_mode='order0')
    part_grad_1 = odl.PartialDerivative(space, 1, method='forward',
                                        pad_mode='order0')

    grad_norm = odl.power_method_opnorm(
        odl.BroadcastOperator(part_grad_0, part_grad_1),
        xstart=odl.util.testutils.noise_element(space))

    part_grad_0 = (1 / grad_norm) * part_grad_0
    part_grad_1 = (1 / grad_norm) * part_grad_1

    # Create tensorflow layer from odl operator
    with tf.name_scope('odl_layers'):
        odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(
                operator, 'RayTransform')
        odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(
                operator.adjoint, 'RayTransformAdjoint')
        odl_grad0_layer = odl.contrib.tensorflow.as_tensorflow_layer(
                part_grad_0, 'PartialGradientDim0')
        odl_grad0_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(
                part_grad_0.adjoint, 'PartialGradientDim0Adjoint')
        odl_grad1_layer = odl.contrib.tensorflow.as_tensorflow_layer(
                part_grad_1, 'PartialGradientDim1')
        odl_grad1_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(
                part_grad_1.adjoint, 'PartialGradientDim1Adjoint')

    return (odl_op_layer, odl_op_layer_adjoint, odl_grad0_layer,
            odl_grad0_layer_adjoint, odl_grad1_layer, odl_grad1_layer_adjoint,
            part_grad_0, part_grad_1, operator, pseudoinverse)


if __name__ == '__main__':
    # Space used for "face" image
    space = odl.uniform_discr([-128, -128], [128, 128], [1024, 768])

    # Space used for "ascent" image
    # space = odl.uniform_discr([-128, -128], [128, 128], [512, 512])

    (_, _, _, _, _, _, _, _, convolution, _) = get_operators(space)
    kernel = convolution.operator.left.vector

    # Create phantom (the "unknown" solution)
    phantom = np.rot90(scipy.misc.face(gray=True), -1)
    # phantom = np.rot90(scipy.misc.ascent(), -1)/255

    phantom = space.element(phantom)

    # Apply convolution to phantom to create data
    g = convolution(phantom)

    # Display the results using the show method
    kernel.show('kernel')
    phantom.show('phantom')
    g.show('convolved phantom')
