#!/usr/bin/python3

import functools;
import tensorflow as tf;
from tensorflow.python.eager import context;
from tensorflow.python.ops import nn;
from tensorflow.python.ops import nn_ops;

class WSConv2D(tf.keras.layers.Conv2D):
  def build(self, input_shape):
    super(WSConv2D, self).build(input_shape);
    self.gain = self.add_weight(shape = (tf.shape(self.kernel)[-1],), dtype = tf.float32, initializer = tf.keras.initializers.Ones()); # self.gain.shape = (cout,)
    # Convert Keras formats to TF native formats.
    if self.padding == 'causal':
      tf_padding = 'VALID'  # Causal padding handled in `call`.
    elif isinstance(self.padding, str):
      tf_padding = self.padding.upper()
    else:
      tf_padding = self.padding
    tf_dilations = list(self.dilation_rate)
    tf_strides = list(self.strides)

    tf_op_name = self.__class__.__name__
    if tf_op_name == 'Conv1D':
      tf_op_name = 'conv1d'  # Backwards compat.
    self._convolution_op = functools.partial(
        nn_ops.convolution_v2,
        strides=tf_strides,
        padding=tf_padding,
        dilations=tf_dilations,
        data_format=self._tf_data_format,
        name=tf_op_name)
  def standardize_weight(self):
    # NOTE: kernel.shape = (kh, kw, cin, cout)
    mean = tf.math.reduce_mean(self.kernel, axis = (0,1,2)); # mean.shape = (cout,)
    var = tf.math.reduce_variance(self.kernel, axis = (0,1,2)); # var.shape = (cout,)
    fan_in = tf.cast(tf.math.reduce_prod(tf.shape(self.kernel)[:-1]), dtype = tf.float32); # fan_in.shape = ()
    scale = self.gain * tf.math.rsqrt(tf.math.maximum(var * fan_in, 1e-4)); # scale.shape = (cout,), gain/(sqrt(fan_in) * stdvar)
    w = (self.kernel - mean) * scale;
    return w;
  def call(self, inputs):
    input_shape = inputs.shape
    if self._is_causal:  # Apply causal padding to inputs for Conv1D.
      inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))
    kernel = self.standardize_weight();
    outputs = self._convolution_op(inputs, kernel)
    if self.use_bias:
      output_rank = outputs.shape.rank
      if self.rank == 1 and self._channels_first:
        # nn.bias_add does not accept a 1D input tensor.
        bias = array_ops.reshape(self.bias, (1, self.filters, 1))
        outputs += bias
      else:
        # Handle multiple batch dimensions.
        if output_rank is not None and output_rank > 2 + self.rank:
          def _apply_fn(o):
            return nn.bias_add(o, self.bias, data_format=self._tf_data_format)
          outputs = conv_utils.squeeze_batch_dims(
              outputs, _apply_fn, inner_rank=self.rank + 1)
        else:
          outputs = nn.bias_add(
              outputs, self.bias, data_format=self._tf_data_format)
    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(input_shape)
      outputs.set_shape(out_shape)
    if self.activation is not None:
      return self.activation(outputs)
    return outputs

if __name__ == "__main__":
  import numpy as np;
  a = np.random.normal(size = (4,224,224,3));
  b = WSConv2D(10,(3,3), padding = 'same')(a);
  print(b.shape)
