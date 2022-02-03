import tensorflow as tf

class DenseRagged(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, activation='linear', **kwargs):
        super(DenseRagged, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
        self.units = units
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
    def build(self, input_shape):
        last_dim = input_shape[-1]
        self.kernel = self.add_weight('kernel', shape=[last_dim, self.units], trainable=True)
        if self.use_bias:
            self.bias = self.add_weight('bias', shape=[self.units,], trainable=True)
        else:
            self.bias = None
        super(DenseRagged, self).build(input_shape)
    def call(self, inputs):
        outputs = tf.ragged.map_flat_values(tf.matmul, inputs, self.kernel)
        if self.use_bias:
            outputs = tf.ragged.map_flat_values(tf.nn.bias_add, outputs, self.bias)
        outputs = tf.ragged.map_flat_values(self.activation, outputs)
        return outputs

class PermopRagged(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PermopRagged, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        super(PermopRagged, self).build(input_shape)
    def call(self, inputs):
        out = tf.math.reduce_sum(inputs, axis=1)
        return out
