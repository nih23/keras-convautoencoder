import theano
from keras import backend as K
#from keras.backend.theano_backend import _on_gpu
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.core import Dense, Layer
from theano import tensor as T
from theano.sandbox.cuda import dnn


class SumLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], 1, input_shape[2], input_shape[3])

    def get_output(self, train=False):
        X = self.get_input(train)
        return X.sum(axis=1, keepdims=True)


class DePool2D(UpSampling2D):
    '''Simplar to UpSample, yet traverse only maxpooled elements

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, channels, upsampled_rows, upsampled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, upsampled_rows, upsampled_cols, channels)` if dim_ordering='tf'.

    # Arguments
        size: tuple of 2 integers. The upsampling factors for rows and columns.
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
    '''
    input_ndim = 4

    def __init__(self, pool2d_layer, *args, **kwargs):
        self._pool2d_layer = pool2d_layer
        super(DePool2D, self).__init__(*args, **kwargs)

    def call(self, inputs):
        X = inputs
        if self.data_format == 'channels_first':
            output = K.repeat_elements(X, self.size[0], axis=2)
            output = K.repeat_elements(output, self.size[1], axis=3)
        elif self.data_format == 'channels_last':
            output = K.repeat_elements(X, self.size[0], axis=2)
            output = K.repeat_elements(output, self.size[1], axis=3)
            output = K.permute_dimensions(output, (0, 2, 3, 1))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        f = K.gradients(loss=K.sum(self._pool2d_layer.output), variables=self._pool2d_layer.input) * output
        f = K.squeeze(f, axis=0)
        print(str(K.get_variable_shape(f)))
        return f

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[2] * self.size[0], input_shape[3] * self.size[1], input_shape[1])
        self.output_dim = output_shape
        return output_shape


def deconv2d_fast(x, kernel, strides=(1, 1), border_mode='valid', data_format='channels_last',
                  image_shape=None, filter_shape=None):
    '''
    border_mode: string, "same" or "valid".
    '''
    if data_format not in {'channels_first', 'channels_last'}:
        raise Exception('Unknown data_format ' + str(data_format))
    if data_format == 'channels_last':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        x = K.permute_dimensions(x, (0, 1, 2, 3))
        kernel = K.permute_dimensions(kernel, (0, 1, 3, 2))

        if image_shape:
            image_shape = (image_shape[0], image_shape[3],
                           image_shape[1], image_shape[2])
        if filter_shape:
            filter_shape = (filter_shape[3], filter_shape[2],
                            filter_shape[0], filter_shape[1])

        if border_mode == 'same':
            th_border_mode = 'full'
            assert (strides == (1, 1))
        elif border_mode == 'valid':
            th_border_mode = 'valid'
        else:
            raise Exception('Border mode not supported: ' + str(border_mode))

        conv_out = K.conv2d(x, kernel, padding=border_mode, strides=strides, data_format=data_format)
    if data_format == 'channels_first':
        raise Exception('Theano backend not implemented.')
        #conv_out = K.permute_dimensions(conv_out, (0, 2, 3, 1))
    return conv_out


class Deconvolution2D(Convolution2D):
    '''Convolution operator for filtering windows of two-dimensional inputs.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, nb_row, nb_col)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, nb_row, nb_col, nb_filter)` if dim_ordering='tf'.


    # Arguments
        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: 'valid' or 'same'.
        subsample: tuple of length 2. Factor by which to subsample output.
            Also called strides elsewhere.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegular            print(single_image.shape)izer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
    '''
    input_ndim = 4

    def __init__(self, binded_conv_layer, filters, kernel_size, nb_out_channels=1, *args, **kwargs):
        self._binded_conv_layer = binded_conv_layer
        self.nb_out_channels = nb_out_channels
        #self.output_dim = self._binded_conv_layer.output_dim
        #print("input shape conv2d: I=" + str(self._binded_conv_layer.input_shape[2]))
#        kwargs['filters'] = self._binded_conv_layer.filters
#        kwargs['nb_row'] = self._binded_conv_layer.input_shape[1]
#        kwargs['nb_col'] = self._binded_conv_layer.input_shape[2]
        self.filters = filters
        self.kernel_size = kernel_size
#        self.input_shape = self._binded_conv_layer.input_shape
        super(Deconvolution2D, self).__init__(filters=filters, kernel_size=kernel_size, *args, **kwargs)


    def build(self, input_shape):
        print("data format: " + str(self._binded_conv_layer.data_format))
        if self.data_format == 'channels_first':
            #channel_axis = 1
            raise Exception('Theano backend not implemented.')
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        self.kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.bias = K.zeros((self.nb_out_channels,))
        self.params = [self.bias]
        self.regularizers = []
        self.kernel = K.permute_dimensions(self._binded_conv_layer.kernel, (1, 0, 2, 3))


        if self.kernel_regularizer:
            self.kernel_regularizer.set_param(self.kernel)
            self.regularizers.append(self.kernel_regularizer)

        if self.bias_regularizer:
            self.bias_regularizer.set_param(self.bias)
            self.regularizers.append(self.bias_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

#        if self.initial_weights is not None:
#            self.set_weights(self.initial_weights)
#            del self.initial_weights

    def compute_output_shape(self, input_shape):
        output_shape = list(super(Deconvolution2D, self).compute_output_shape(input_shape))

        if self.data_format == 'channels_first':
            #output_shape[1] = self.nb_out_channels
            raise Exception('Theano backend not implemented.')
        elif self.data_format == 'channels_last':
            output_shape[3] = self.nb_out_channels
        else:
            raise Exception('Invalid data format: ' + self.data_format)
        print("shapi: " + str(output_shape))
        return tuple(output_shape)

    def call(self, inputs):
        X = inputs
        conv_out = deconv2d_fast(X, self.kernel,
                                 strides=self.strides,
                                 border_mode=self.padding,
                                 data_format=self.data_format,
                                 image_shape=inputs.shape,
                                 filter_shape=self.kernel_shape)
        if self.data_format == 'channels_first':
            raise Exception('Theano backend not implemented.')
            #output = conv_out + K.reshape(self.bias, (1, self.nb_out_channels, 1, 1))
        elif self.data_format == 'channels_last':
            output = conv_out + K.reshape(self.bias, (1, 1, 1, self.nb_out_channels))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'filters': self.filters,
                  'input_shape': self.input_shape,
                  '_binded_conv_layer': self._binded_conv_layer.get_config(),
                  'kernel_initializer': self.kernel_initializer.get_config() if self.kernel_initializer else None,
                  'data_format': self.data_format,
                  'activation': self.activation.__name__,
                  'padding': self.padding,
                  'strides': self.strides,
                  'kernel_regularizer': self.kernel_regularizer.get_config() if self.kernel_regularizer else None,
                  'bias_regularizer': self.bias_regularizer.get_config() if self.bias_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'kernel_constraint': self.kernel_constraint.get_config() if self.kernel_constraint else None,
                  'bias_constraint': self.bias_constraint.get_config() if self.bias_constraint else None}
        base_config = super(Convolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DependentDense(Dense):
    def __init__(self, units, master_layer, init='glorot_uniform', activation='linear', weights=None,
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, input_dim=None, **kwargs):
        self.master_layer = master_layer
        self.units = units
        self.initial_weights = weights
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel = K.transpose(self.master_layer.kernel)
        super(DependentDense, self).__init__(units, **kwargs)

    def build(self, input_shape):
        #input_dim = input_shape[-1]
        self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        self.params = [self.bias]
        self.regularizers = []
        if self.kernel_regularizer:
            self.kernel_regularizer.set_param(self.kernel)
            self.regularizers.append(self.kernel_regularizer)

        if self.bias_regularizer:
            self.bias_regularizer.set_param(self.bias)
            self.regularizers.append(self.bias_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
            
        def get_config(self):
            config = {'name': self.__class__.__name__,
                      'units': self.units,
                      'input_shape': self.input_shape,
                      'master_layer':  self.master_layer.get_config(),
                      'kernel_initializer': self.kernel_initializer.get_config() if self.kernel_initializer else None,
                      'data_format': self.data_format,
                      'activation': self.activation.__name__,
                      'padding': self.padding,
                      'strides': self.strides,
                      'kernel_regularizer': self.kernel_regularizer.get_config() if self.kernel_regularizer else None,
                      'bias_regularizer': self.bias_regularizer.get_config() if self.bias_regularizer else None,
                      'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                      'kernel_constraint': self.kernel_constraint.get_config() if self.kernel_constraint else None,
                      'bias_constraint': self.bias_constraint.get_config() if self.bias_constraint else None}
            base_config = super(Dense, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
