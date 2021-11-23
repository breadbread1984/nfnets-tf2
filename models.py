#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_addons as tfa;
from tensorflow.python.eager import context;
from tensorflow.python.ops import nn;
from tensorflow.python.ops import nn_ops;

nfnet_params = {
    'F0': {
        'width': [256, 512, 1536, 1536], 'depth': [1, 2, 6, 3],
        'train_imsize': 192, 'test_imsize': 256,
        'RA_level': '405', 'drop_rate': 0.2,
        'expansion': [0.5] * 4,
        'group_width': [128] * 4,
        'big_width': [True] * 4,
        'stride_pattern': [1, 2, 2, 2],},
    'F1': {
        'width': [256, 512, 1536, 1536], 'depth': [2, 4, 12, 6],
        'train_imsize': 224, 'test_imsize': 320,
        'RA_level': '410', 'drop_rate': 0.3,
        'expansion': [0.5] * 4,
        'group_width': [128] * 4,
        'big_width': [True] * 4,
        'stride_pattern': [1, 2, 2, 2],},
    'F2': {
        'width': [256, 512, 1536, 1536], 'depth': [3, 6, 18, 9],
        'train_imsize': 256, 'test_imsize': 352,
        'RA_level': '410', 'drop_rate': 0.4,
        'expansion': [0.5] * 4,
        'group_width': [128] * 4,
        'big_width': [True] * 4,
        'stride_pattern': [1, 2, 2, 2],},
    'F3': {
        'width': [256, 512, 1536, 1536], 'depth': [4, 8, 24, 12],
        'train_imsize': 320, 'test_imsize': 416,
        'RA_level': '415', 'drop_rate': 0.4,
        'expansion': [0.5] * 4,
        'group_width': [128] * 4,
        'big_width': [True] * 4,
        'stride_pattern': [1, 2, 2, 2],},
    'F4': {
        'width': [256, 512, 1536, 1536], 'depth': [5, 10, 30, 15],
        'train_imsize': 384, 'test_imsize': 512,
        'RA_level': '415', 'drop_rate': 0.5,
        'expansion': [0.5] * 4,
        'group_width': [128] * 4,
        'big_width': [True] * 4,
        'stride_pattern': [1, 2, 2, 2],},
    'F5': {
        'width': [256, 512, 1536, 1536], 'depth': [6, 12, 36, 18],
        'train_imsize': 416, 'test_imsize': 544,
        'RA_level': '415', 'drop_rate': 0.5,
        'expansion': [0.5] * 4,
        'group_width': [128] * 4,
        'big_width': [True] * 4,
        'stride_pattern': [1, 2, 2, 2],},
    'F6': {
        'width': [256, 512, 1536, 1536], 'depth': [7, 14, 42, 21],
        'train_imsize': 448, 'test_imsize': 576,
        'RA_level': '415', 'drop_rate': 0.5,
        'expansion': [0.5] * 4,
        'group_width': [128] * 4,
        'big_width': [True] * 4,
        'stride_pattern': [1, 2, 2, 2],},
    'F7': {
        'width': [256, 512, 1536, 1536], 'depth': [8, 16, 48, 24],
        'train_imsize': 480, 'test_imsize': 608,
        'RA_level': '415', 'drop_rate': 0.5,
        'expansion': [0.5] * 4,
        'group_width': [128] * 4,
        'big_width': [True] * 4,
        'stride_pattern': [1, 2, 2, 2],},
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

def SqueezeExcite(in_channel, out_channel, se_ratio = 0.5, hidden_ch = None):
  if se_ratio is None and hidden_ch is None:
    raise Exception('either se_ratio or hidden_ch must be provided!');
  if hidden_ch is None:
    hidden_ch = max(1, int(in_channel * se_ratio));
  inputs = tf.keras.Input((None, None, in_channel)); # inputs.shape = (batch, height, width, in_channel)
  results = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis = (1,2), keepdims = True))(inputs); # results.shape = (batch, 1, 1, in_channel)
  results = tf.keras.layers.Dense(hidden_ch, activation = tf.keras.activations.relu)(results); # results.shape = (batch, 1, 1, hidden_ch)
  results = tf.keras.layers.Dense(out_channel, activation = tf.keras.activations.sigmoid)(results); # results.shape = (batch, 1, 1, out_channel)
  return tf.keras.Model(inputs = inputs, outputs = results);

def StochDepth(in_channel, drop_rate, scale_by_keep = False):
  # sample wise dropout
  training = tf.python.keras.backend.learning_phase();
  inputs = tf.keras.Input((None, None, in_channel));
  results = inputs;
  if not training:
    return tf.keras.Model(inputs = inputs, outputs = results);
  r = tf.keras.layers.Lambda(lambda x: tf.random.uniform(shape = (tf.shape(x)[0], 1, 1, 1), dtype = tf.float32))(inputs);
  mask = tf.keras.layers.Lambda(lambda x, dr: tf.math.floor(1. - dr + x), arguments = {'dr': drop_rate})(r);
  if scale_by_keep:
    results = tf.keras.layers.Lambda(lambda x, dr: x / (1. - dr))(results);
  results = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([results, mask]);
  return tf.keras.Model(inputs = inputs, outputs = results);

class AutoScaled(tf.keras.layers.Layer):
  def __init__(self, scalar = True, initializer = 'zeros', **kwargs):
    assert initializer in ['ones', 'zeros'];
    self.scalar = scalar;
    self.initializer = initializer;
    super(AutoGained, self).__init__(**kwargs);
  def build(self, input_shape):
    if self.initializer == 'ones':
      initializer = tf.keras.initializers.Ones();
    elif self.initializer == 'zeros':
      initializer = tf.keras.initializers.Zeros();
    if self.scalar == False:
      self.gain = self.add_weight(shape = input_shape, dtype = tf.float32, initializer = initializer);
  def call(self, inputs):
    results = inputs * self.gain;
    return results;
  def get_config(self):
    config = super(AutoScaled, self).get_config();
    config['scalar'] = self.scalar;
    config['initializer'] = self.initializer;
    return config;
  @classmethod
  def from_config(cls, config):
    return cls(**config);

def NFBlock(in_channel, out_channel, kernel_size = 3, alpha = 0.2, beta = 1.0, stride = 1, group_size = 128, big_width = True, expansion = 0.5, use_two_convs = True, se_ratio = 0.5, stochdepth_rate = None):
  # normalization free residual block
  # alpha: variance weight of residual branch, the variance of the output of the normalization free residual block is (1 + alpha**2) * var(inputs)
  # beta: sqrt(var(inputs))
  inputs = tf.keras.Input((None, None, in_channel));
  results = tf.keras.layers.GELU()(inputs);
  # 1) normalize input to make results ~ N(0,1)
  results = tf.keras.layers.Lambda(lambda x, b: x / b, arguments = {'b': beta})(results);
  # 2) short cut output
  if stride > 1:
    # NOTE: the first block of a stage
    shortcut = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = 'same')(results);
    shortcut = WSConv2D(out_channel, kernel_size = (1,1), padding = 'same', name = 'conv_shortcut')(shortcut);
  elif in_channel != out_channel:
    # NOTE: the first block of a stage
    shortcut = WSConv2D(out_channel, kernel_size = (1,1), padding = 'same', name = 'conv_shortcut')(results);
  else:
    # NOTE: within one stage the var(shortcut) is accumulated, rather than normalized
    shortcut = inputs;
  # 3) residual branch uses weight scaled convolution to prevent meanshift
  width = int((out_channel if big_width else in_channel) * expansion);
  results = WSConv2D(group_size * (width // group_size), kernel_size = (1,1), groups = width // group_size, padding = 'same', name = 'conv0', activation = tf.keras.activations.gelu)(results);
  results = WSConv2D(group_size * (width // group_size), kernel_size = (kernel_size, kernel_size), groups = width // group_size, padding = 'same', name = 'conv1')(results);
  if use_two_convs:
    results = tfa.layers.GELU(results);
    results = WSConv2D(group_size * (width // group_size), kernel_size = (kernel_size, kernel_size), groups = width // group_size, padding = 'same', name = 'conv1b')(results);
  results = tfa.layers.GELU(results);
  results = WSConv2D(out_channel, kernel_size = (1,1), padding = 'same', name = 'conv2')(results); # results.shape = (batch, height, width, out_channel)
  # use channel attention
  attention = SqueezeExcite(out_channel, out_channel, se_ratio)(results); # ch_attention.shape = (batch, height, width, out_channel)
  results = tf.keras.layers.Lambda(lambda x: x[0] * 2 * x[1])([attention, results]); # results.shape = (batch, height, width, out_channel)
  res_avg_var = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(tf.math.reduce_variance(x, axis = (0,1,2))))(results);
  # use sample wise dropout
  if stochdepth_rate is not None and 0. < stochdepth_rate < 1.:
    results = StochDepth(out_channel, stochdepth_rate)(results); # results.shape = (batch, height, width, out_channel)
  results = AutoScaled(scalar = True, initializer = 'zeros', name = 'skip_gain')(results); # results.shape = (batch, height, width, out_channels)
  # 4) combine the scaled residual and shortcut
  results = tf.keras.layers.Lambda(lambda x, a: x[0] * a + x[1], arguments = {'a': alpha})([results, shortcut]);
  return tf.keras.Model(inputs = inputs, outputs = (results, res_avg_var));

def NFNet(variant = 'F0', width = 1., use_two_convs = True, se_ratio = 0.5, stochdepth_rate = 0.1):
  assert variant in ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7'];
  # NOTE: inputs must satisfy that mean(inputs) = 0, var(inputs) = 1
  inputs = tf.keras.Input((None, None, 3)); # inputs.shape = (batch, height, width, 3)
  # 1) stem
  results = WSConv2D(16, kernel_size = (3,3), strides = (2,2), padding = 'same', name = 'stem_conv0', activation = tf.keras.activations.gelu)(inputs);
  results = WSConv2D(32, kernel_size = (3,3), strides = (1,1), padding = 'same', name = 'stem_conv1', activation = tf.keras.activations.gelu)(results);
  results = WSConv2D(64, kernel_size = (3,3), strides = (1,1), padding = 'same', name = 'stem_conv2', activation = tf.keras.activations.gelu)(results);
  results = WSConv2D(nfnet_params[variant]['width'][0] // 2, kernel_size = (3,3), strides = (2,2), padding = 'same', name = 'stem_conv3')(results);
  # 2) body
  index = 0;
  expected_std = 1.; # WSConv2D does not change distribution, therefore results ~ N(0,1)
  for block_width, stage_depth, expand_ratio, group_size, big_wdith, stride in zip(nfnet_params[variant]['width'], nfnet_params[variant]['depth'], nfnet_params[variant]['expansion'],
                                                                                   nfnet_params[variant]['group_width'], nfnet_params[variant]['big_width'], nfnet_params[variant]['stride_pattern']):
    for block_index in range(stage_depth):
      # NOTE: within a stage the var(shortcut) is accumulated, rather than normalized
      results, res_avg_var = NFBlock(results.shape[-1], int(block_width * width), beta = expected_std, stride = stride if block_index == 0 else 1,
                                     group_size = group_size, big_width = big_width, expansion = expand_ratio, use_two_convs = use_two_convs,
                                     se_ratio = se_ratio, stochdepth_rate = stochdepth_rate * index / sum(nfnet_params[variant]['depth']))(results);
      index += 1;
      expected_std = 1. if block_index == 0 else (expected_std**2 + alpha**2)**0.5;

  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":
  import numpy as np;
  a = np.random.normal(size = (4,224,224,3));
  model = NFNet();
  b = model(a);
  model.save('model.h5');
