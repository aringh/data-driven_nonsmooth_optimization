"""Helper function for setting up the tomography problem."""

import tensorflow as tf
import odl


def get_operators(space, geometry):
    # Create the forward operator
    operator = odl.tomo.RayTransform(space, geometry)
    pseudoinverse = odl.tomo.fbp_op(operator)

    # Normalize the operator and create pseudo-inverse
    opnorm = odl.power_method_opnorm(operator)
    operator = (1 / opnorm) * operator

    pseudoinverse = pseudoinverse * opnorm

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
            operator, pseudoinverse)
