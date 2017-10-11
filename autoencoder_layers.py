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
#            print('DePool2D: th')
            output = K.repeat_elements(X, self.size[0], axis=2)
            output = K.repeat_elements(output, self.size[1], axis=3)
        elif self.data_format == 'channels_last':
#            print('DePool2D: tf')
            output = K.repeat_elements(X, self.size[0], axis=2)
            output = K.repeat_elements(output, self.size[1], axis=3)
            output = K.permute_dimensions(output, (0, 2, 3, 1))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        #f = T.grad(T.sum(self._pool2d_layer.get_activations(inputs)), wrt=self._pool2d_layer.call(inputs)) * output
#        print("input -> " + str(X.shape))
#        print("output -> " + str(output.shape))
        f = K.gradients(loss=K.sum(self._pool2d_layer.output), variables=self._pool2d_layer.input) * output
        f = K.squeeze(f, axis=0)
        print(str(K.get_variable_shape(f)))
        return f

    def compute_output_shape(self, input_shape):
 #       print("input -> " + str(input_shape))
 #       print("ups -> " + str(input_shape[2] * self.size[0]))

        output_shape = (input_shape[0], input_shape[2] * self.size[0], input_shape[3] * self.size[1], input_shape[1])
        print("depool out shape: " + str(output_shape))
        self.output_dim = output_shape
        return output_shape


def deconv2d_fast(x, kernel, strides=(1, 1), border_mode='valid', data_format='channels_last',
                  image_shape=None, filter_shape=None):
    '''
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    '''
    if data_format not in {'channels_first', 'channels_last'}:
        raise Exception('Unknown data_format ' + str(data_format))
    #print("df: " + data_format)
    if data_format == 'channels_last':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        #x = x.dimshuffle((0, 3, 1, 2))
        #print("X prior Shuffle " + str(x.shape))
        x = K.permute_dimensions(x, (0, 1, 2, 3))
        #kernel = kernel.dimshuffle((2, 3, 0, 1))
        #print("K prior Shuffle " + str(kernel.shape))
        kernel = K.permute_dimensions(kernel, (0, 1, 3, 2))

        if image_shape:
            image_shape = (image_shape[0], image_shape[3],
                           image_shape[1], image_shape[2])
        if filter_shape:
            filter_shape = (filter_shape[3], filter_shape[2],
                            filter_shape[0], filter_shape[1])

    # if _on_gpu() and dnn.dnn_available():
    #     if border_mode == 'same':
    #         assert (strides == (1, 1))
    #         conv_out = dnn.dnn_conv(img=x,
    #                                 kerns=kernel,
    #                                 border_mode='full')
    #         shift_x = (kernel.shape[2] - 1) // 2
    #         shift_y = (kernel.shape[3] - 1) // 2
    #         conv_out = conv_out[:, :,
    #                    shift_x:x.shape[2] + shift_x,
    #                    shift_y:x.shape[3] + shift_y]
    #     else:
    #         conv_out = dnn.dnn_conv(img=x,
    #                                 conv_mode='cross',
    #                                 kerns=kernel,
    #                                 border_mode=border_mode,
    #                                 subsample=strides)
    # else:
        if border_mode == 'same':
            th_border_mode = 'full'
            assert (strides == (1, 1))
        elif border_mode == 'valid':
            th_border_mode = 'valid'
        else:
            raise Exception('Border mode not supported: ' + str(border_mode))

        #conv_out = T.nnet.conv2d(x, kernel, 
        #   border_mode=th_border_mode,
        #   subsample=strides,
        #   filter_flip=False,  # <<<<< IMPORTANT 111, dont flip kern
        #   input_shape=image_shape,
        #   filter_shape=filter_shape)
        print(x.shape)
        print(kernel.shape)
        conv_out = K.conv2d(x, kernel, padding=border_mode, strides=strides, data_format=data_format)
        if border_mode == 'same':
            shift_x = (kernel.shape[2] - 1) // 2
            shift_y = (kernel.shape[3] - 1) // 2
            conv_out = conv_out[:, :,
            shift_x:x.shape[2] + shift_x,
            shift_y:x.shape[3] + shift_y]
    if data_format == 'channels_first':
#        conv_out = conv_out.dimshuffle((0, 2, 3, 1))
        conv_out = K.permute_dimensions(conv_out, (0, 2, 3, 1))
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

    def __init__(self, binded_conv_layer, nb_out_channels=1, *args, **kwargs):
        self._binded_conv_layer = binded_conv_layer
        self.nb_out_channels = nb_out_channels
        #self.output_dim = self._binded_conv_layer.output_dim
        #print("input shape conv2d: I=" + str(self._binded_conv_layer.input_shape[2]))
        kwargs['filters'] = self._binded_conv_layer.filters
        kwargs['nb_row'] = self._binded_conv_layer.input_shape[1]
        kwargs['nb_col'] = self._binded_conv_layer.input_shape[2]
        super(Deconvolution2D, self).__init__(*args, **kwargs)


    def build(self, input_shape):
        print("data format: " + str(self._binded_conv_layer.data_format))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        print('input dim: ' + str(input_shape))
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
        print("deconv2d output shape")
        output_shape = list(super(Deconvolution2D, self).compute_output_shape(input_shape))

        if self.data_format == 'channels_first':
            output_shape[1] = self.nb_out_channels
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
            output = conv_out + K.reshape(self.bias, (1, self.nb_out_channels, 1, 1))
        elif self.data_format == 'channels_last':
            output = conv_out + K.reshape(self.bias, (1, 1, 1, self.nb_out_channels))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        #output = self.activation(output)
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(Convolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DependentDense(Dense):
    def __init__(self, output_dim, master_layer, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None, **kwargs):
        self.master_layer = master_layer
        self.units = output_dim
        self.initial_weights = weights
        self.W_regularizer = W_regularizer
        self.bias_regularizer = b_regularizer
        self.kernel = K.transpose(self.master_layer.kernel)
#        print("units: " + str(output_dim))
        super(DependentDense, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        #self.bias = K.zeros((self.master_layer.units,))
        #print(self.master_layer.kernel.shape)
        input_dim = input_shape[-1]
#        print("units@build: " + str(self.units))
#        print("kernel units@build: " + str(self.kernel.shape))
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
