"""
Script that defines the detection models used by the run_model_training_routine.py in the context
of the qMRI project and tumor detection
"""

import numpy as np
from typing import Union
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Add,
    LayerNormalization,
    Embedding,
    MultiHeadAttention,
    Flatten,
    Reshape,
)
from tensorflow.keras.initializers import glorot_uniform
import tensorflow_addons as tfa

# %% EfficientNet


def EfficientNet(
    num_classes: int,
    input_shape: Union[list, tuple],
    data_augmentation: bool = True,
    scale_image: bool = True,
    image_normalization_stats: tuple = None,
    use_age: bool = False,
    use_age_thr_tabular_network: bool = False,
    debug: bool = False,
    pretrained: bool = True,
    freeze_weights: bool = False,
):
    """
    Make sure not to use the functional API if not you get a
    nested model from which is difficult to get the GradCam
    """

    denseDropoutRate = 0.8

    # building  model
    img_input = Input(shape=input_shape, name="image")
    x = img_input

    # normalize of scale the image
    if image_normalization_stats:
        # apply image normalization with given mean and variance
        x = tf.keras.layers.Normalization(
            mean=image_normalization_stats[0], variance=image_normalization_stats[1]
        )(x)
    elif scale_image:
        x = tf.keras.layers.Rescaling(scale=1 / 127.5, offset=-1)(x)

    # data augmentation
    if data_augmentation:
        x = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
        x = tf.keras.layers.RandomRotation(
            factor=0.25, fill_mode="nearest", interpolation="bilinear"
        )(x)
        x = tf.keras.layers.RandomZoom(
            height_factor=0.2,
            width_factor=0.2,
            fill_mode="nearest",
            interpolation="bilinear",
        )(x)
        x = tf.keras.layers.RandomTranslation(
            height_factor=0.2,
            width_factor=0.2,
            fill_mode="nearest",
            interpolation="bilinear",
        )(x)
        x = tf.keras.layers.RandomContrast(factor=0.2)(x)

    # use EfficientNet from keras as feature extractor
    efficientNet = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top=False,
        weights="imagenet" if pretrained else None,
        pooling="avg",
        input_tensor=img_input,
    )

    # froze model layers
    if freeze_weights:
        efficientNet.trainable = False

    # build model
    x = BatchNormalization()(efficientNet.output)
    x = Dropout(denseDropoutRate)(x)

    # classifier
    # x = Dense(
    #     units=128,
    #     activation=denseActivation,
    #     kernel_regularizer=denseRegularizer,
    #     kernel_constraint=denseConstrain,
    # )(x)
    # x = BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU(
    #     alpha=0.3,
    # )(x)
    # x = Dropout(denseDropoutRate)(x)

    if use_age:
        # @ToDo Implement tabular network
        # if use_age_thr_tabular_network:
        # else:
        age_input = Input(shape=(1,), name="age")
        enc_age = tf.keras.layers.Flatten(name="flatten_csv")(age_input)
        x = tf.keras.layers.concatenate([x, enc_age])

    output = Dense(
        units=num_classes,
        activation="softmax",
        name="label",
    )(x)

    if use_age:
        model = tf.keras.Model(
            inputs=[img_input, age_input], outputs=output, name="EfficientNet"
        )
    else:
        model = tf.keras.Model(inputs=img_input, outputs=output, name="EfficientNet")

    # print model if needed
    if debug is True:
        print(model.summary())
    print(model.summary())
    return model


#%% ResNet18


def ResNet50(
    num_classes: int,
    input_shape: Union[list, tuple],
    use_age: bool = False,
    use_age_thr_tabular_network: bool = False,
    debug: bool = False,
    pretrained: bool = True,
    freeze_weights: bool = False,
):
    """
    Make sure not to use the functional API if not you get a
    nested model from which is difficult to get the GradCam
    """

    denseRegularizer = "L2"
    denseConstrain = None
    denseActivation = None
    denseDropoutRate = 0.8

    img_input = Input(shape=input_shape, name="image")
    x = tf.keras.applications.mobilenet.preprocess_input(img_input)

    # use EfficientNet from keras as feature extractor
    resnet50 = tf.keras.applications.resnet.ResNet152(
        include_top=False,
        weights="imagenet" if pretrained else None,
        pooling="avg",
        input_tensor=x,
    )

    # froze model layers
    if freeze_weights:
        resnet50.trainable = False

    # build model
    x = BatchNormalization()(resnet50.output)
    x = Dropout(denseDropoutRate)(x)

    # classifier
    # x = Dense(
    #     units=128,
    #     activation=denseActivation,
    #     kernel_regularizer=denseRegularizer,
    #     kernel_constraint=denseConstrain,
    # )(x)
    # x = BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU(
    #     alpha=0.3,
    # )(x)
    # x = Dropout(denseDropoutRate)(x)

    if use_age:
        # @ToDo Implement tabular network
        # if use_age_thr_tabular_network:
        # else:
        age_input = Input(shape=(1,), name="age")
        enc_age = tf.keras.layers.Flatten(name="flatten_csv")(age_input)
        x = tf.keras.layers.concatenate([x, enc_age])

    output = Dense(
        units=num_classes,
        activation="softmax",
        name="label",
    )(x)

    if use_age:
        model = tf.keras.Model(
            inputs=[img_input, age_input], outputs=output, name="EfficientNet"
        )
    else:
        model = tf.keras.Model(inputs=img_input, outputs=output, name="EfficientNet")

    # print model if needed
    if debug is True:
        print(model.summary())
    print(model.summary())
    return model


# %% SIMPLE DETECTION MODEL
def SimpleDetectionModel_TF(
    num_classes: int,
    input_shape: Union[list, tuple],
    data_augmentation: bool = True,
    scale_image: bool = True,
    image_normalization_stats: tuple = None,
    kernel_size: Union[list, tuple] = (3, 3),
    pool_size: Union[list, tuple] = (2, 2),
    use_age: bool = False,
    age_normalization_stats: tuple = None,
    use_age_thr_tabular_network: bool = False,
    debug: bool = False,
    use_pretrained: bool = False,
    pretrained_model_path: str = None,
    freeze_weights: bool = True,
):

    convRegularizer = tf.keras.regularizers.l2(l=0.00001)
    denseRegularizer = "L2"

    convConstrain = None
    denseConstrain = None

    convActivation = None
    denseActivation = "relu"

    convDropoutRate = 0.4
    denseDropoutRate = 0.2

    # building  model
    if use_pretrained:
        print("Using pre-trained model weights. Using the image encoder weights only.")
        pre_t_model = tf.keras.models.load_model(pretrained_model_path, compile=False)
        img_input = pre_t_model.inputs
        x = pre_t_model.layers[-6].output

        if freeze_weights:
            pre_t_model.trainable = False

    else:
        img_input = Input(shape=input_shape, name="image")
        x = img_input

        # normalize of scale the image
        if image_normalization_stats:
            # apply image normalization with given mean and variance
            x = tf.keras.layers.Normalization(
                mean=image_normalization_stats[0], variance=image_normalization_stats[1]
            )(x)
        elif scale_image:
            x = tf.keras.layers.Rescaling(scale=1 / 127.5, offset=-1)(x)

        # data augmentation
        if data_augmentation:
            x = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
            x = tf.keras.layers.RandomRotation(
                factor=0.25, fill_mode="nearest", interpolation="bilinear"
            )(x)
            x = tf.keras.layers.RandomZoom(
                height_factor=0.2,
                width_factor=0.2,
                fill_mode="nearest",
                interpolation="bilinear",
            )(x)
            x = tf.keras.layers.RandomTranslation(
                height_factor=0.2,
                width_factor=0.2,
                fill_mode="nearest",
                interpolation="bilinear",
            )(x)
            x = tf.keras.layers.RandomContrast(factor=0.2)(x)

        for nbr_filter in [64, 128, 256, 512]:
            x = Conv2D(
                filters=nbr_filter,
                kernel_size=kernel_size,
                activation=convActivation,
                padding="same",
                kernel_regularizer=convRegularizer,
            )(x)
            x = tfa.layers.InstanceNormalization()(x)
            x = tf.keras.layers.LeakyReLU(
                alpha=0.3,
            )(x)
            x = Conv2D(
                filters=nbr_filter,
                kernel_size=kernel_size,
                activation=convActivation,
                padding="same",
                kernel_regularizer=convRegularizer,
                kernel_constraint=convConstrain,
            )(x)
            x = tfa.layers.InstanceNormalization()(x)
            x = tf.keras.layers.LeakyReLU(
                alpha=0.3,
            )(x)
            x = MaxPooling2D(pool_size=pool_size)(x)
            x = tf.keras.layers.SpatialDropout2D(rate=convDropoutRate)(x)

    # classifier
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dense(
        units=512,
        activation=denseActivation,
        kernel_regularizer=denseRegularizer,
        kernel_constraint=denseConstrain,
    )(x)

    # x = tfa.layers.InstanceNormalization()(x)
    # x = tf.keras.layers.LeakyReLU(
    #     alpha=0.3,
    # )(x)
    x = Dropout(denseDropoutRate)(x)

    if use_age:
        # @ToDo Implement tabular network
        # if use_age_thr_tabular_network:
        # else:
        age_input = Input(shape=(1,), name="age")
        enc_age = tf.keras.layers.Flatten(name="flatten_csv")(age_input)
        if age_normalization_stats:
            enc_age = tf.keras.layers.Normalization(
                mean=age_normalization_stats[0], variance=age_normalization_stats[1]
            )(enc_age)
        x = tf.keras.layers.concatenate([x, enc_age])

    output = Dense(
        units=num_classes,
        activation="softmax",
        name="label",
    )(x)

    if use_age:
        model = tf.keras.Model(inputs=[img_input, age_input], outputs=output)
    else:
        model = tf.keras.Model(inputs=img_input, outputs=output)

    # print model if needed
    if debug is True:
        print(model.summary())
    print(model.summary())
    return model


# %% RESNET MODEL


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, use_dropout: bool = False):
        super(ResnetIdentityBlock, self).__init__(name="")
        filters1, filters2, filters3 = filters
        self.use_dropout = use_dropout

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tfa.layers.InstanceNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding="same")
        self.bn2b = tfa.layers.InstanceNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tfa.layers.InstanceNormalization()

        self.dp2 = tf.keras.layers.SpatialDropout2D(rate=0.2)

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        if self.use_dropout:
            return self.dp2(x)
        else:
            return tf.nn.relu(x)


class ResnetConvBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, use_dropout: bool = False):
        super(ResnetConvBlock, self).__init__(name="")
        filters1, filters2, filters3 = filters
        self.use_dropout = use_dropout

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tfa.layers.InstanceNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding="same")
        self.bn2b = tfa.layers.InstanceNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tfa.layers.InstanceNormalization()

        self.conv2s = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2s = tfa.layers.InstanceNormalization()

        self.dp2 = tf.keras.layers.SpatialDropout2D(rate=0.2)

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        s = self.conv2s(input_tensor)
        s = self.bn2s(s)

        x += s

        if self.use_dropout:
            return self.dp2(x)
        else:
            return tf.nn.relu(x)


class ConvBlock(tf.keras.Model):
    def __init__(
        self, kernel_size, filters, pool: bool = False, use_dropout: bool = True
    ):
        super(ConvBlock, self).__init__(name="")
        self.pool = pool
        self.use_dropout = use_dropout

        self.conv2 = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=(1, 1),
            padding="valid",
            name="conv2_1",
            kernel_initializer=glorot_uniform(seed=0),
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.mp2 = tf.keras.layers.MaxPooling2D()
        self.dp2 = tf.keras.layers.SpatialDropout2D(rate=0.2)

    def call(self, input_tensor, training=False):
        x = self.conv2(input_tensor)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        if self.use_dropout:
            x = self.dp2(x)

        if self.pool:
            return self.mp2(x)
        else:
            return x


def ResNet9(
    num_classes: int,
    input_shape: Union[list, tuple],
    use_age: bool = False,
    use_age_thr_tabular_network: bool = False,
    debug: bool = False,
):
    # building  model
    img_input = Input(shape=input_shape, name="image")

    # first conv
    x = ConvBlock(filters=64, kernel_size=(3, 3), pool=False)(img_input)

    # second convolution + MaxPooling
    x = ConvBlock(filters=64, kernel_size=(3, 3), pool=True)(x)

    # conv 3 to 5 using residual block
    x = ResnetConvBlock(filters=(128, 128, 128), kernel_size=(3, 3), use_dropout=True)(
        x
    )

    # conv 6 to 9 using residual block
    x = ResnetConvBlock(filters=(256, 256, 256), kernel_size=(3, 3), use_dropout=True)(
        x
    )

    # dense classifier
    x = GlobalAveragePooling2D()(x)
    if use_age:
        # @ToDo Implement tabular network
        # if use_age_thr_tabular_network:
        # else:
        age_input = Input(shape=(1,), name="age")
        enc_age = tf.keras.layers.Flatten(name="flatten_csv")(age_input)

        x = tf.keras.layers.concatenate([x, enc_age])

    x = Dense(units=128)(x)
    x = Dropout(0.2)(x)
    output = Dense(units=num_classes, activation="softmax", name="label")(x)

    if use_age:
        model = tf.keras.Model(inputs=[img_input, age_input], outputs=output)
    else:
        model = tf.keras.Model(inputs=img_input, outputs=output)

    # print model if needed
    if debug is True:
        print(model.summary())

    return model


# %% ############################## Vision Transformer model
class ShiftedPatchTokenization(tf.keras.layers.Layer):
    def __init__(
        self,
        image_size,
        patch_size,
        num_patches,
        projection_dim,
        vanilla=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vanilla = vanilla  # Flag to swtich to vanilla patch extractor
        self.image_size = image_size
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.num_patches = num_patches
        self.flatten_patches = Reshape((self.num_patches, -1))
        self.projection_dim = projection_dim
        self.projection = Dense(units=self.projection_dim)
        self.layer_norm = LayerNormalization(epsilon=10e-6)

    def crop_shift_pad(self, images, mode):
        # Build the diagonally shifted images
        if mode == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif mode == "left-down":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = self.half_patch
            shift_width = 0
        elif mode == "right-up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        else:
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = self.half_patch

        # Crop the shifted images and pad them
        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=self.image_size,
            target_width=self.image_size,
        )
        return shift_pad

    def call(self, images):
        if not self.vanilla:
            # Concat the shifted images with the original image
            images = tf.concat(
                [
                    images,
                    self.crop_shift_pad(images, mode="left-up"),
                    self.crop_shift_pad(images, mode="left-down"),
                    self.crop_shift_pad(images, mode="right-up"),
                    self.crop_shift_pad(images, mode="right-down"),
                ],
                axis=-1,
            )
        # Patchify the images and flatten it
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        flat_patches = self.flatten_patches(patches)
        if not self.vanilla:
            # Layer normalize the flat patches and linearly project it
            tokens = self.layer_norm(flat_patches)
            tokens = self.projection(tokens)
        else:
            # Linearly project the flat patches
            tokens = self.projection(flat_patches)
        return (tokens, patches)

    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "num_patches": self.num_patches,
                "projection_dim": self.projection_dim,
                "vanilla": self.vanilla,
            }
        )
        return config


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

    def call(self, encoded_patches):
        encoded_positions = self.position_embedding(self.positions)
        encoded_patches = encoded_patches + encoded_positions
        return encoded_patches

    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {
                "num_patches": self.num_patches,
                "num_patches": self.num_patches,
            }
        )
        return config


class MultiHeadAttentionLSA(tf.keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The trainable temperature term. The initial value is
        # the square root of the key dimension.
        self.tau = tf.Variable(tf.math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x


def ViT(
    input_size: Union[list, tuple],
    num_classes: int,
    use_age: bool = False,
    use_age_thr_tabular_network: bool = False,
    use_gradCAM: bool = False,
    patch_size: int = 16,
    projection_dim: int = 64,
    num_heads: int = 4,
    mlp_head_units: Union[list, tuple] = (512, 256),
    transformer_layers: int = 8,
    transformer_units: int = None,
    debug=False,
):

    # ViT parameters
    patch_size = patch_size
    num_patches = (input_size[0] // patch_size) * (input_size[1] // patch_size)
    transformer_layers = transformer_layers
    if not transformer_units:
        transformer_units = [projection_dim * 2, projection_dim]
    else:
        transformer_units = transformer_units

    # DEFINE BUILDING BLOCKS
    # ################ MLP (the classifier)
    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = Dense(units, activation=tf.nn.gelu)(x)
            x = Dropout(dropout_rate)(x)
        return x

    # # ################# PATCH EXTRACTION
    class Patches(tf.keras.layers.Layer):
        def __init__(self, patch_size):
            super(Patches, self).__init__()
            self.patch_size = patch_size

        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches

        def get_config(self):

            config = super().get_config().copy()
            config.update(
                {
                    "patch_size": self.patch_size,
                }
            )
            return config

    # ################  PATCH ENCODING LAYER
    class PatchEncoder(tf.keras.layers.Layer):
        def __init__(self, num_patches, projection_dim):
            super(PatchEncoder, self).__init__()
            self.num_patches = num_patches
            self.projection_dim = projection_dim
            self.projection = Dense(units=projection_dim)
            self.position_embedding = Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )

        def call(self, patch):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            encoded = self.projection(patch) + self.position_embedding(positions)
            return encoded

        def get_config(self):

            config = super().get_config().copy()
            config.update(
                {
                    "num_patches": self.num_patches,
                    "projection_dim": self.projection_dim,
                }
            )
            return config

    # ACTUALLY BUILD THE MODEL
    img_input = Input(shape=input_size, name="image")
    patches = Patches(patch_size)(img_input)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)
    if use_age:
        # @ToDo Implement tabular network
        # if use_age_thr_tabular_network:
        # else:
        age_input = Input(shape=(1,), name="age")
        enc_age = Flatten(name="flatten_csv")(age_input)

        representation = tf.keras.layers.concatenate([representation, enc_age])

    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = Dense(num_classes, activation="softmax", name="label")(features)
    # Create the Keras model.
    # Create the Keras model.
    if use_age:
        model = tf.keras.Model(inputs=[img_input, age_input], outputs=logits)
    else:
        model = tf.keras.Model(inputs=img_input, outputs=logits)

    # print model if needed
    if debug is True:
        print(model.summary())

    return model


def ViT_2(
    input_size: Union[list, tuple],
    num_classes: int,
    use_age: bool = False,
    use_age_thr_tabular_network: bool = False,
    use_gradCAM: bool = False,
    patch_size: int = 16,
    projection_dim: int = 64,
    num_heads: int = 4,
    mlp_head_units: Union[list, tuple] = (512, 256),
    transformer_layers: int = 8,
    transformer_units: int = None,
    debug=False,
    vanilla: bool = False,
):

    # ViT parameters
    patch_size = patch_size
    num_patches = (input_size[0] // patch_size) * (input_size[1] // patch_size)
    transformer_layers = transformer_layers
    if not transformer_units:
        transformer_units = [projection_dim * 2, projection_dim]
    else:
        transformer_units = transformer_units

    # ACTUALLY BUILD THE MODEL
    img_input = Input(shape=input_size, name="image")

    # Create patches.
    (tokens, _) = ShiftedPatchTokenization(
        image_size=input_size[0],
        patch_size=patch_size,
        num_patches=num_patches,
        projection_dim=projection_dim,
        vanilla=vanilla,
    )(img_input)

    # Encode patches.
    encoded_patches = PatchEncoder(
        num_patches=num_patches, projection_dim=projection_dim
    )(tokens)

    # Build the diagonal attention mask
    diag_attn_mask = 1 - tf.eye(num_patches)
    diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        if not vanilla:
            attention_output = MultiHeadAttentionLSA(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1, attention_mask=diag_attn_mask)
        else:
            attention_output = MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)

    if use_age:
        # @ToDo Implement tabular network
        # if use_age_thr_tabular_network:
        # else:
        age_input = Input(shape=(1,), name="age")
        enc_age = tf.keras.layers.Flatten(name="flatten_csv")(age_input)

        representation = tf.keras.layers.concatenate([representation, enc_age])

    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = Dense(num_classes, name="label")(features)
    # Create the Keras model.
    if use_age:
        model = tf.keras.Model(inputs=[img_input, age_input], outputs=logits)
    else:
        model = tf.keras.Model(inputs=img_input, outputs=logits)

    # print model if needed
    if debug is True:
        print(model.summary())

    return model


# %% ############################ AGE ONLY MODELS


def age_only_model(
    num_classes: int,
    model_version: str = "age_to_classes",
    debug: bool = False,
):
    denseRegularizer = "L2"
    denseConstrain = None

    # building  model
    age_input = Input(shape=[1], name="image")
    x = age_input

    if model_version != "age_to_classes":
        if model_version == "simple_age_encoder":
            layer_specs = [num_classes]
        elif model_version == "large_age_encoder":
            layer_specs = [8, 16, 32]
        # build model based on layer_specs
        for nodes in layer_specs:
            x = Dense(
                units=nodes,
                activation=None,
                kernel_regularizer=denseRegularizer,
                kernel_constraint=denseConstrain,
            )(x)

            x = tfa.layers.InstanceNormalization()(x)
            x = tf.keras.layers.LeakyReLU(
                alpha=0.3,
            )(x)

    output = Dense(
        units=num_classes,
        activation="softmax",
        name="label",
    )(x)

    model = tf.keras.Model(inputs=age_input, outputs=output)

    # print model if needed
    if debug is True:
        print(model.summary())
    print(model.summary())
    return model
