#!/usr/bin/python3

import functools;
import tensorflow as tf;
import tensorflow_addons as tfa;
from tensorflow.python.eager import context;
from tensorflow.python.ops import nn;
from tensorflow.python.ops import nn_ops;

nfnet_params = {
    'F0': {
        'width': [256, 512, 1536, 1536], 'depth': [1, 2, 6, 3],
        'train_imsize': 192, 'test_imsize': 256,
        'RA_level': '405', 'drop_rate': 0.2},
    'F1': {
        'width': [256, 512, 1536, 1536], 'depth': [2, 4, 12, 6],
        'train_imsize': 224, 'test_imsize': 320,
        'RA_level': '410', 'drop_rate': 0.3},
    'F2': {
        'width': [256, 512, 1536, 1536], 'depth': [3, 6, 18, 9],
        'train_imsize': 256, 'test_imsize': 352,
        'RA_level': '410', 'drop_rate': 0.4},
    'F3': {
        'width': [256, 512, 1536, 1536], 'depth': [4, 8, 24, 12],
        'train_imsize': 320, 'test_imsize': 416,
        'RA_level': '415', 'drop_rate': 0.4},
    'F4': {
        'width': [256, 512, 1536, 1536], 'depth': [5, 10, 30, 15],
        'train_imsize': 384, 'test_imsize': 512,
        'RA_level': '415', 'drop_rate': 0.5},
    'F5': {
        'width': [256, 512, 1536, 1536], 'depth': [6, 12, 36, 18],
        'train_imsize': 416, 'test_imsize': 544,
        'RA_level': '415', 'drop_rate': 0.5},
    'F6': {
        'width': [256, 512, 1536, 1536], 'depth': [7, 14, 42, 21],
        'train_imsize': 448, 'test_imsize': 576,
        'RA_level': '415', 'drop_rate': 0.5},
    'F7': {
        'width': [256, 512, 1536, 1536], 'depth': [8, 16, 48, 24],
        'train_imsize': 480, 'test_imsize': 608,
        'RA_level': '415', 'drop_rate': 0.5},
};

class WSConv2D(tf.keras.layers.Conv2D):
  def build(self, input_shape):
    super(WSConv2D, self).build(input_shape);
    self.gain = self.add_weight(shape = (tf.shape(self.kernel)[-1],), dtype = tf.float32, initializer = tf.keras.initializers.Ones()); # self.gain.shape = (cout,)
  def standardize_weight(self):
    # NOTE: kernel.shape = (kh, kw, cin, cout)
    mean = tf.math.reduce_mean(self.kernel, axis = (0,1,2)); # mean.shape = (cout,)
    var = tf.math.reduce_variance(self.kernel, axis = (0,1,2)); # var.shape = (cout,)
    fan_in = tf.cast(tf.math.reduce_prod(tf.shape(self.kernel)[:-1]), dtype = tf.float32); # fan_in.shape = ()
    scale = self.gain * tf.math.rsqrt(tf.math.maximum(var * fan_in, 1e-4)); # scale.shape = (cout,), gain/(sqrt(fan_in) * stdvar)
    w = (self.kernel - mean) * scale;
    return w;
  def call(self, inputs):
    input_shape = inputs.shape;
    if self._is_causal:  # Apply causal padding to inputs for Conv1D.
      inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs));
    kernel = self.standardize_weight();
    outputs = self.convolution_op(inputs, kernel);
    if self.use_bias:
      output_rank = outputs.shape.rank;
      if self.rank == 1 and self._channels_first:
        # nn.bias_add does not accept a 1D input tensor.
        bias = array_ops.reshape(self.bias, (1, self.filters, 1));
        outputs += bias;
      else:
        # Handle multiple batch dimensions.
        if output_rank is not None and output_rank > 2 + self.rank:
          def _apply_fn(o):
            return nn.bias_add(o, self.bias, data_format=self._tf_data_format);
          outputs = conv_utils.squeeze_batch_dims(
              outputs, _apply_fn, inner_rank=self.rank + 1);
        else:
          outputs = nn.bias_add(
              outputs, self.bias, data_format=self._tf_data_format);
    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(input_shape);
      outputs.set_shape(out_shape);
    if self.activation is not None:
      return self.activation(outputs);
    return outputs;

def NFBlock(in_channel, out_channel, alpha = 0.2, beta = 1.0, stride = 1):
  inputs = tf.keras.Input((None, None, in_channel));
  results = tf.keras.layers.GELU()(inputs);
  results = tf.keras.layers.Lambda(lambda x, b: x * b, arguments = {'b': beta})(results);
  if stride > 1:
    shortcut = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = 'same')(results);

def NFNet(variant = 'F0'):
  assert variant in ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7'];
  inputs = tf.keras.Input((None, None, 3)); # inputs.shape = (batch, height, width, 3)
  results = WSConv2D(16, kernel_size = (3,3), strides = (2,2), padding = 'same', name = 'stem_conv0', activation = tf.keras.activations.gelu)(inputs);
  results = WSConv2D(32, kernel_size = (3,3), strides = (1,1), padding = 'same', name = 'stem_conv1', activation = tf.keras.activations.gelu)(results);
  results = WSConv2D(64, kernel_size = (3,3), strides = (1,1), padding = 'same', name = 'stem_conv2', activation = tf.keras.activations.gelu)(results);
  results = WSConv2D(nfnet_params[variant]['width'][0] // 2, kernel_size = (3,3), strides = (2,2), padding = 'same', name = 'stem_conv3')(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":
  import numpy as np;
  a = np.random.normal(size = (4,224,224,3));
  model = NFNet();
  b = model(a);
  model.save('model.h5');
