from __future__ import print_function
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, Input, BatchNormalization, ZeroPadding2D, \
    Activation, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Concatenate, GlobalMaxPooling2D
from keras.models import Model
from keras.backend import image_data_format, int_shape
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.utils import plot_model


channel_axis = 3 if image_data_format() == 'channels_last' else 1
eps = 1.001e-5
num_classes = 10

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def conv_layer(x, num_filters, kernel, stride=1, padding='same', layer_name="conv"):
    conv = Conv2D(num_filters,
                  kernel_size=kernel,
                  use_bias=False,
                  strides=stride,
                  padding=padding,
                  name=layer_name)(x)
    return conv


def Global_Average_Pooling(x, stride=1, name=None):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return GlobalAveragePooling2D(name=name)(x)
    # But maybe you need to install h5py and curses or not


def Average_pooling(x, pool_size=[2, 2], stride=2, name=None):
    return AveragePooling2D(pool_size, strides=stride, name=name)(x)


def Max_pooling(x, pool_size=[3, 3], stride=2, padding='SAME', name=None):
    return MaxPooling2D(pool_size=pool_size, strides=stride, padding=padding, name=name)(x)


def activation_fn(x, name=None):
    return Activation('relu', name=name)(x)


def batch_normalization_fn(x, name=None):
    return BatchNormalization(axis=channel_axis, epsilon=eps, name=name)(x)

def dropout_fn(x, rate):
    return Dropout(rate=rate)(x)

def dense_fn(layer, filters=100):
    return Dense(filters)(layer)

def classifier_fn(layer, num_labels=2, actv='softmax'):
    return Dense(num_labels, activation=actv)(layer)

def concat_fn(layers, axis=channel_axis, name=None):
    return Concatenate(axis=axis, name=name)(layers)

def load_densenet_model(use_weights, pooling='avg'):
    weights = 'imagenet' if use_weights == True else None
    base_model = DenseNet121(include_top=False, weights=weights, input_tensor=Input(shape=(224, 224, 3)),
                             input_shape=(224, 224, 3), pooling=pooling)
    return base_model

def load_inceptionresnet_model(use_weights, pooling='avg', input_tensor=None):
    weights = 'imagenet' if use_weights == True else None
    base_model = InceptionResNetV2(include_top=False, weights=weights, input_tensor=input_tensor,
                             input_shape=(299, 299, 3), pooling=pooling)
    return base_model

class DenseNetInceptionResnetModel():
    def __init__(self, num_labels, use_imagenet_weights=True):
        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights
        self.model = self.get_model()

    def get_model(self):
        dense_model = load_densenet_model(self.use_imagenet_weights, pooling='avg')
        dense_out = dense_model.layers[-1].output
        dense_input = dense_model.layers[0].output
        inception_model = load_inceptionresnet_model(self.use_imagenet_weights, pooling='avg', input_tensor=dense_input)
        inception_out = inception_model.layers[-1].output
        out = concat_fn([dense_out, inception_out], 1)
        classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=dense_model.input, outputs=classifier)
        return model

# Base Model
class DenseNetBaseModel():
    def __init__(self, num_labels, use_imagenet_weights=True):
        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights
        self.model = self.get_model()

    def get_model(self):
        base_model = load_densenet_model(self.use_imagenet_weights)
        # Freeze high layers in densent model
        # for layer in base_model.layers:
        #     layer.trainable = False
        #     if layer.name == 'pool2_relu':
        #         break

        out = base_model.layers[-1].output
        classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=base_model.input, outputs=classifier)
        return model

# Densenet Modify

class DenseNet121_Modify():
    def __init__(self, num_labels, use_imagenet_weights=True):
        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights
        self.model = self.get_model()

    def get_model(self):
        base_model = load_densenet_model(self.use_imagenet_weights)
        out = base_model.get_layer("pool4_pool").output
        for i in range(1, 12):
            out = conv2d_bn(out, i * 64, 3, 3)
        # out = dropout_fn(base_model.layers[-1].output, 0.5)
        out = Global_Average_Pooling(out)
        classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')

        model = Model(inputs=base_model.input, outputs=[classifier])
        return model


# InceptionResnet Model
class InceptionResNetModel():
    def __init__(self, num_labels, use_imagenet_weights=True):
        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights
        self.model = self.get_model()

    def get_model(self):
        base_model = load_inceptionresnet_model(self.use_imagenet_weights)
        out = base_model.layers[-1].output
        classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=base_model.input, outputs=classifier)
        return model

class DensenetWISeRModel():
    def __init__(self, num_labels, use_imagenet_weights=True):
        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights
        self.model = self.get_model()

    def get_model1(self):
        # dense_model = load_densenet_model(self.use_imagenet_weights)
        # densenet_out = dense_model.layers[-1].output

        # Add Slice Branch
        # slice_input = dense_model.layers[0].output
        slice_input = Input(shape=(224, 224, 3))
        x = conv2d_bn(slice_input, 320, 224, 5, 'valid')
        x = Max_pooling(x=x, pool_size=[1, 5], stride=3, padding='valid', name=None)
        out = Global_Average_Pooling(x)

        # combine densenet with Slice Branch
        # out = concat_fn([densenet_out, slice_out], axis=1)
        out = dense_fn(out, 2048)
        out = dense_fn(out, 2048)
        classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=slice_input, outputs=classifier)
        return model

    def dense_block(self, x, blocks, name):
        """A dense block.

        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        for i in range(blocks):
            x = self.conv_block(x, 32, name=name + '_block_' + str(i + 1))
        return x

    def transition_block(self, x, reduction, name):
        """A transition block.

        # Arguments
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        bn_axis = 3 if image_data_format() == 'channels_last' else 1
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name=name + '_bn')(x)
        x = Activation('relu', name=name + '_relu')(x)
        x = Conv2D(int(int_shape(x)[bn_axis] * reduction), 1,
                          use_bias=False,
                          name=name + '_conv')(x)
        x = AveragePooling2D([1,2], strides=2, name=name + '_pool')(x)
        return x


    def conv_block(self, x, growth_rate, name):
        """A building block for a dense block.

        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.

        # Returns
            Output tensor for the block.
        """
        bn_axis = 3 if image_data_format() == 'channels_last' else 1

        x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                       name=name + '_1_bn')(x)
        x1 = Activation('relu', name=name + '_1_relu')(x1)
        x1 = Conv2D(growth_rate, [1, 3],
                           padding='same',
                           use_bias=False,
                           name=name + '_2_conv')(x1)
        x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
        return x

    def DenseNet(self, blocks,
                 input_shape=(224, 224, 3),
                 classes=172):

        # img_input = Input(shape=input_shape)

        bn_axis = 3 if image_data_format() == 'channels_last' else 1

        dense_model = load_densenet_model(self.use_imagenet_weights)
        densenet_out = dense_model.layers[-1].output
        slice_input = dense_model.layers[0].output

        # slice_input = img_input
        x = conv2d_bn(slice_input, 320, 224, 5, 'valid')
        x = Max_pooling(x=x, pool_size=[1, 5], stride=3, padding='valid', name=None)

        x = self.dense_block(x, blocks[0], name='conv2')
        x = self.transition_block(x, 0.5, name='pool2_')
        x = self.dense_block(x, blocks[1], name='conv3')
        x = self.transition_block(x, 0.5, name='pool3_')
        x = self.dense_block(x, blocks[2], name='conv4')
        x = self.transition_block(x, 0.5, name='pool4_')
        x = self.dense_block(x, blocks[3], name='conv5')

        x = BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='bn_')(x)
        x = Activation('relu', name='relu_')(x)


        x = GlobalAveragePooling2D(name='avg_pool_')(x)

        x = concat_fn([x, densenet_out], 1)

        x = Dense(classes, activation='softmax', name='fc172')(x)
        inputs = dense_model.input

        model = Model(inputs, x, name='densenet')

        return model


    def get_model(self):

        dense_model = load_densenet_model(self.use_imagenet_weights)
        densenet_out = dense_model.layers[-1].output
        slice_input = dense_model.layers[0].output
        x = conv2d_bn(slice_input, 320, 224, 5, 'valid')
        x = Max_pooling(x=x, pool_size=[1, 5], stride=3, padding='valid', name=None)

        x = GlobalAveragePooling2D(name='avg_pool_')(x)

        x = concat_fn([x, densenet_out], 1)

        x = Dense(self.num_labels, activation='softmax', name='fc172')(x)

        model = Model(dense_model.input, x, name='densenet')
        # model = self.DenseNet([6, 12, 24, 16], classes=self.num_labels)
        return model

