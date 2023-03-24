# %%
"""
Script that trains an attention based bag-aggregation methods to classify pediatric brain tumors based on the encoding produced by the SDM4 model.
Conde inspired by the https://arxiv.org/abs/1802.04712 and https://keras.io/examples/vision/attention_mil_classification/

STEPS
- Load dataset information (using splitting from the training of the SDM4 model to avoid bias)
- load the trained SDM4 model (used for feature extraction)
- create one bag for each of the subjects from the training and validation spit using the instances for each subjects: 
 - load the image
 - get encoded vector
 - return bag with encoded indtances
- Build MIL-attention based model
 - shared encoder of the encodigns
 - attention layer
 - aggredation 
 - prediction
- visualize the most important instances in the bag
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import glob
import csv
import json
import numpy as np
import argparse
import importlib
import logging
import random
from pathlib import Path
from distutils.util import strtobool

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.utils.extmath import softmax

# local imports
import utilities
import data_utilities
import models
import losses
import tf_callbacks

su_debug_flag = True

# --------------------------------------
# read the input arguments and set the base folder
# --------------------------------------
if not su_debug_flag:
    parser = argparse.ArgumentParser(
        description="Run cross validation training for tumor detection."
    )
    parser.add_argument(
        "-wf",
        "--WORKING_FOLDER",
        required=True,
        type=str,
        help="Provide the working folder where the trained model will be saved.",
    )
    parser.add_argument(
        "-df",
        "--IMG_DATASET_FOLDER",
        required=True,
        type=str,
        help="Provide the Image Dataset Folder where the folders for each modality are located (see dataset specifications in the README file).",
    )
    parser.add_argument(
        "-mr_modelities",
        "--MR_MODALITIES",
        nargs="+",
        required=False,
        default=["T2"],
        help="Specify which MR modalities to use during training (T1 and/or T2)",
    )

    parser.add_argument(
        "-gpu",
        "--GPU_NBR",
        default=0,
        type=str,
        help="Provide the GPU number to use for training.",
    )

    parser.add_argument(
        "-n_folds",
        "--NBR_FOLDS",
        required=False,
        type=int,
        default=1,
        help="Number of cross validation folds.",
    )
    parser.add_argument(
        "-lr",
        "--LEARNING_RATE",
        required=False,
        type=float,
        default=0.0001,
        help="Learning rate",
    )
    parser.add_argument(
        "-batch_size",
        "--BATCH_SIZE",
        required=False,
        type=int,
        default=16,
        help="Specify batch size. Default 16",
    )
    parser.add_argument(
        "-e",
        "--MAX_EPOCHS",
        required=False,
        type=int,
        default=300,
        help="Number of max training epochs.",
    )
    parser.add_argument(
        "-path_to_encoder_model",
        "--PATH_TO_ENCODER_MODEL",
        required=False,
        type=str,
        default=None,
        help="Specify the path to the encoder model.",
    )
    parser.add_argument(
        "-mil_use_age",
        "--MIL_USE_AGE",
        required=False,
        dest="USE_AGE",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Specify if the MIL model should use the age information. If true, the age information is encoded using a fuly connected model and feature fusion is used to combine image and age infromation.",
    )
    parser.add_argument(
        "-age_normalization",
        "--AGE_NORMALIZATION",
        required=False,
        dest="AGE_NORMALIZATION",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Specify if the age should be normalized. If True, age is normalized using mean and std from the trianing dataset ([-1,1] norm).",
    )
    parser.add_argument(
        "-loss",
        "--LOSS",
        required=False,
        type=str,
        default="CCE",
        help="Specify loss to use during model training (categorical cross entropy CCE, MCC, binary categorical cross entropy BCE. Other can be defined and used. Just implement.",
    )
    parser.add_argument(
        "-debug_dataset_fraction",
        "--DEBUG_DATASET_FRACTION",
        required=False,
        type=float,
        default=1.0,
        help="Specify the percentage of the dataset to use during training and validation. This is for debug",
    )
    parser.add_argument(
        "-optimizer",
        "--OPTIMIZER",
        required=False,
        type=str,
        default="SGD",
        help="Specify which optimizer to use. Here one can set SGD or ADAM. Others can be implemented.",
    )
    # other parameters
    parser.add_argument(
        "-rns",
        "--RANDOM_SEED_NUMBER",
        required=False,
        type=int,
        default=29122009,
        help="Specify random number seed. Useful to have models trained and tested on the same data.",
    )

    args_dict = dict(vars(parser.parse_args()))

else:
    args_dict = {
        "WORKING_FOLDER": r"C:\Users\iulta54\Documents\PediatricBrainTumorClassification",
        "IMG_DATASET_FOLDER": r"C:\Datasets\CBTN\EXTRACTED_SLICES_TFR_MERGED_FROM_TB_20230320",
        "MR_MODALITIES": ["T2"],
        "GPU_NBR": "0",
        "NBR_FOLDS": 1,
        "LEARNING_RATE": 0.0001,
        "BATCH_SIZE": 32,
        "MAX_EPOCHS": 50,
        "PATH_TO_ENCODER_MODEL": r"C:\Users\iulta54\Documents\PEDIAT~1\TRAINE~1\TEST_optm_ADAM_SDM4_TFRdata_True_modality_T2_loss_MCC_and_CCE_Loss_lr_0.0001_batchSize_32_pretrained_False_frozenWeight_True_useAge_True_no_encoder_useGradCAM_False_seed_1111\fold_1\last_model\last_model",
        "PATH_TO_CONFIGURATION_FILES": r"C:\Users\iulta54\Documents\PEDIAT~1\TRAINE~1\TEST_optm_ADAM_SDM4_TFRdata_True_modality_T2_loss_MCC_and_CCE_Loss_lr_0.0001_batchSize_32_pretrained_False_frozenWeight_True_useAge_True_no_encoder_useGradCAM_False_seed_1111",
        "MIL_USE_AGE": True,
        "AGE_NORMALIZATION": True,
        "LOSS": "MCC_and_CCE_Loss",
        "RANDOM_SEED_NUMBER": 1111,
        "MR_MODALITIES": ["T2"],
        "DEBUG_DATASET_FRACTION": 1,
        "MODEL_NAME": "MIL_TEST",
        "OPTIMIZER": "ADAM",
    }

# revise model name
args_dict[
    "MODEL_NAME"
] = f'{args_dict["MODEL_NAME"]}_optm_{args_dict["OPTIMIZER"]}_loss_{args_dict["LOSS"]}_lr_{args_dict["LEARNING_RATE"]}_batchSize_{args_dict["BATCH_SIZE"]}_useAge_{args_dict["MIL_USE_AGE"]}_seed_{args_dict["RANDOM_SEED_NUMBER"]}'

# --------------------------------------
# set GPU (or device)
# --------------------------------------

# import tensorflow
import tensorflow as tf
import tensorflow_addons as tfa
import warnings

tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.utils import to_categorical
from tensorflow_addons.optimizers import Lookahead

# from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()

devices = tf.config.list_physical_devices("GPU")

if devices:
    print(f'Running training on GPU # {args_dict["GPU_NBR"]} \n')
    warnings.simplefilter(action="ignore", category=FutureWarning)
    tf.config.experimental.set_memory_growth(devices[0], True)
else:
    Warning(
        f"ATTENTION!!! MODEL RUNNING ON CPU. Check implementation in case GPU is wanted."
    )
# -------------------------------------
# Check that the given folder exist
# -------------------------------------
for folder, fd in zip(
    [
        args_dict["WORKING_FOLDER"],
        args_dict["IMG_DATASET_FOLDER"],
        args_dict["PATH_TO_ENCODER_MODEL"],
    ],
    ["working folder", "image dataset folder", "encoder model"],
):
    if not any([os.path.isdir(folder), os.path.isfile(folder)]):
        raise ValueError(f"{fd.capitalize} not found. Given {folder}.")

# -------------------------------------
# Create folder where to save the model
# -------------------------------------
args_dict["SAVE_PATH"] = os.path.join(
    args_dict["WORKING_FOLDER"], "trained_models_archive", args_dict["MODEL_NAME"]
)
Path(args_dict["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)

# print input variables
max_len = max([len(key) for key in args_dict])
[
    print(f"{key:{max_len}s}: {value} ({type(value)})")
    for key, value in args_dict.items()
]

# %% LOAD THE train_validation_test.json and config.json files from the encoder model

with open(
    os.path.join(args_dict["PATH_TO_CONFIGURATION_FILES"], "train_val_test_files.json")
) as file:
    train_val_test_files = json.load(file)
    test_files = [
        os.path.join(args_dict["IMG_DATASET_FOLDER"], args_dict["MR_MODALITIES"][0], f)
        for f in train_val_test_files["test"]
    ]

    # get also per subject files
    per_subjects_files = dict.fromkeys(
        [os.path.basename(f).split("_")[2] for f in test_files]
    )
    for subj in per_subjects_files:
        per_subjects_files[subj] = {"files": [], "gt": [], "last": [], "best": []}
        per_subjects_files[subj]["files"] = [
            f for f in test_files if subj == os.path.basename(f).split("_")[2]
        ]

# open also the configuration file
with open(
    os.path.join(args_dict["PATH_TO_CONFIGURATION_FILES"], "config.json")
) as file:
    config = json.load(file)

# %% LOAD MODEL
loaded_enc_model = tf.keras.models.load_model(args_dict["PATH_TO_ENCODER_MODEL"])
# refine model to only get the image encoding vector
# get index to the global average pooling layer (but using the output of the batch norm that follows the global average pooling)
idx = [
    i
    for i, l in enumerate(loaded_enc_model.layers)
    if l.name == "global_average_pooling2d"
][0]
img_input = loaded_enc_model.inputs
encoded_image = loaded_enc_model.layers[idx + 1].output
enc_model = tf.keras.Model(inputs=img_input, outputs=encoded_image)
enc_model.summary()

# %% DEFINE UTILITY THAT RETURNS A BAG OF ELEMENTS, ONE FOR EACH SUBJECT
def get_per_subject_bag_enc(
    subject_file_paths, enc_model, config_file, bag_size: int = 5
):
    """
    Given a list of files, the encoding model and the configuration file which was used to train the
    encoder model, returns a numpy array with size [bag_size, enc_dim] containing the encodings of the subject images.

    Steps:
    - create generator to consume the subject files
    - encode the images
    - aggregate in one bag
    """

    # build generator
    target_size = (224, 224)
    img_gen, _ = data_utilities.tfrs_data_generator(
        file_paths=subject_file_paths,
        input_size=target_size,
        batch_size=1,
        buffer_size=10,
        return_gradCAM=config_file["USE_GRADCAM"],
        return_age=config_file["USE_AGE"],
        dataset_type="test",
        nbr_classes=config_file["NBR_CLASSES"],
        output_as_RGB=True
        if any(
            [
                config_file["MODEL_TYPE"] == "EfficientNet",
                config_file["MODEL_TYPE"] == "ResNet50",
            ]
        )
        else False,
    )

    # for each image get the encoded version using the encoding model
    enc_images = enc_model.predict(img_gen, verbose=0)
    # do something with the images
    if enc_images.shape[0] < bag_size:
        # randomly oversample to get the right number of instantces for the bag
        for i in range(bag_size - enc_images.shape[0]):
            enc_images = np.concatenate(
                [
                    enc_images,
                    np.expand_dims(
                        enc_images[np.random.randint(enc_images.shape[0])], axis=0
                    ),
                ]
            )

    if enc_images.shape[0] > bag_size:
        enc_images = enc_images[0:bag_size, :]

    # get also the label for this bag
    bag_label = next(iter(img_gen))[1]

    return (enc_images, bag_label.numpy())


# build file names for each subjects int he training and validation sets
image_files = [
    os.path.join(args_dict["IMG_DATASET_FOLDER"], args_dict["MR_MODALITIES"][0], f)
    for f in train_val_test_files["train"][0]
]
image_files.extend(
    [
        os.path.join(args_dict["IMG_DATASET_FOLDER"], args_dict["MR_MODALITIES"][0], f)
        for f in train_val_test_files["validation"][0]
    ]
)
# get also per subject files
per_subjects_files = dict.fromkeys(
    [os.path.basename(f).split("_")[2] for f in image_files]
)
for subj in per_subjects_files:
    per_subjects_files[subj] = [
        f for f in image_files if subj == os.path.basename(f).split("_")[2]
    ]

# get the actual subject bags of instances for each subject
# %%
tr_bags, tr_bags_labels = [], []
for idx, subject_files in enumerate(per_subjects_files.values()):
    print(f"Working on subject {idx+1:} of {len(per_subjects_files)}")
    bag, labels = get_per_subject_bag_enc(subject_files, enc_model, config, bag_size=3)
    tr_bags.append(bag)
    tr_bags_labels.append(labels)
    if idx == 50:
        break

# %% CREATE MIL layer

from tensorflow import keras
from tensorflow.keras import layers


class MILAttentionLayer(tf.keras.layers.Layer):
    """Implementation of the attention-based Deep MIL layer.

    Args:
      weight_params_dim: Positive Integer. Dimension of the weight matrix.
      kernel_initializer: Initializer for the `kernel` matrix.
      kernel_regularizer: Regularizer function applied to the `kernel` matrix.
      use_gated: Boolean, whether or not to use the gated mechanism.

    Returns:
      List of 2D tensors with BAG_SIZE length.
      The tensors are the attention scores after softmax with shape `(batch_size, 1)`.
    """

    def __init__(
        self,
        weight_params_dim,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        use_gated=False,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    def build(self, input_shape):

        # Input shape.
        # List of 2D tensors with shape: (batch_size, input_dim).
        input_dim = input_shape[0][1]

        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True,
        )

        self.w_weight_params = self.add_weight(
            shape=(self.weight_params_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w_regularizer,
            trainable=True,
        )

        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.weight_params_dim),
                initializer=self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True,
            )
        else:
            self.u_weight_params = None

        self.input_built = True

    def call(self, inputs):

        # Assigning variables from the number of inputs.
        instances = [self.compute_attention_scores(instance) for instance in inputs]

        # Apply softmax over instances such that the output summation is equal to 1.
        alpha = tf.math.softmax(instances, axis=0)

        return [alpha[i] for i in range(alpha.shape[0])]

    def compute_attention_scores(self, instance):

        # Reserve in-case "gated mechanism" used.
        original_instance = instance

        # tanh(v*h_k^T)
        instance = tf.math.tanh(tf.tensordot(instance, self.v_weight_params, axes=1))

        # for learning non-linear relations efficiently.
        if self.use_gated:

            instance = instance * tf.math.sigmoid(
                tf.tensordot(original_instance, self.u_weight_params, axes=1)
            )

        # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
        return tf.tensordot(instance, self.w_weight_params, axes=1)


# %% MIL MODEL
def create_model(num_classes, instance_shape, bag_size: int = 25):

    # Extract features from inputs.
    inputs, embeddings = [], []
    shared_dense_layer_1 = layers.Dense(128, activation="relu")
    shared_dense_layer_2 = layers.Dense(64, activation="relu")
    for _ in range(bag_size):
        inp = layers.Input(instance_shape)
        dense_1 = shared_dense_layer_1(inp)
        dense_2 = shared_dense_layer_2(dense_1)
        inputs.append(inp)
        embeddings.append(dense_2)

    # Invoke the attention layer.
    alpha = MILAttentionLayer(
        weight_params_dim=256,
        kernel_regularizer=keras.regularizers.l2(0.01),
        use_gated=True,
        name="alpha",
    )(embeddings)

    # Multiply attention weights with the input layers.
    multiply_layers = [
        layers.multiply([alpha[i], embeddings[i]]) for i in range(len(alpha))
    ]

    # Concatenate layers.
    concat = layers.concatenate(multiply_layers, axis=1)

    # Classification output node.
    output = layers.Dense(num_classes, activation="softmax")(concat)

    return keras.Model(inputs, output)


model = create_model(
    num_classes=tr_bags[0][1].shape[-1],
    instance_shape=tr_bags[0][0].shape[-1],
    bag_size=tr_bags[0][0].shape[0],
)
model.summary()
# %% COMPILE MODEL

learning_rate_fn = tfa.optimizers.CyclicalLearningRate(
    initial_learning_rate=1e-2,
    maximal_learning_rate=1e-1,
    scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
    step_size=2 * len(tr_bags),
)

if args_dict["OPTIMIZER"] == "SGD":
    print(f'{" "*6}Using SGD optimizer.')
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate_fn
        if learning_rate_fn
        else args_dict["LEARNING_RATE"],
        decay=1e-6,
        momentum=0.9,
        nesterov=True,
        clipvalue=0.5,
    )

elif args_dict["OPTIMIZER"] == "ADAM":
    print(f'{" "*6}Using AdamW optimizer.')

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate_fn
        if learning_rate_fn
        else args_dict["LEARNING_RATE"],
        weight_decay=0.0001,
    )

# wrap using LookAhead which helps smoothing out validation curves
optimizer = Lookahead(optimizer, sync_period=5, slow_step_size=0.5)

if args_dict["LOSS"] == "MCC":
    print(f'{" "*6}Using MCC loss.')
    importlib.reload(losses)
    loss = losses.MCC_Loss()
    what_to_monitor = tfa.metrics.MatthewsCorrelationCoefficient(
        num_classes=config["NBR_CLASSES"]
    )
elif args_dict["LOSS"] == "MCC_and_CCE_Loss":
    print(f'{" "*6}Using sum of MCC and CCE loss.')
    importlib.reload(losses)
    loss = losses.MCC_and_CCE_Loss()
    what_to_monitor = tfa.metrics.MatthewsCorrelationCoefficient(
        num_classes=config["NBR_CLASSES"]
    )
elif args_dict["LOSS"] == "CCE":
    print(f'{" "*6}Using CCE loss.')
    loss = tf.keras.losses.CategoricalCrossentropy()
    what_to_monitor = "val_accuracy"
elif args_dict["LOSS"] == "BCE":
    print(f'{" "*6}Using BCS loss.')
    loss = tf.keras.losses.BinaryCrossentropy()
    what_to_monitor = "val_accuracy"
else:
    raise ValueError(
        f"The loss provided is not available. Implement in the losses.py or here."
    )

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[
        "accuracy",
        tfa.metrics.MatthewsCorrelationCoefficient(num_classes=config["NBR_CLASSES"]),
    ],
)

# %% TRAIN MODEL

import tqdm


def train(train_data, train_labels, val_data, val_labels, model, config):

    # Train model.
    # Prepare callbacks.
    # Path where to save best weights.

    # Take the file name from the wrapper.
    file_path = "/tmp/best_model_weights.h5"

    # Initialize model checkpoint callback.
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        file_path,
        monitor="val_loss",
        verbose=0,
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )

    # Initialize early stopping callback.
    # The model performance is monitored across the validation data and stops training
    # when the generalization error cease to decrease.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, mode="min"
    )

    # Compile model.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Fit model.
    model.fit(
        tf.stack(train_data),
        tf.stack(train_labels),
        validation_data=(val_data, val_labels),
        epochs=20,
        # class_weight=config["CLASS_WEIGHTS"],
        batch_size=1,
        callbacks=[early_stopping, model_checkpoint],
        verbose=0,
    )

    # Load best weights.
    model.load_weights(file_path)

    return model


# Training model(s).
trained_models = [
    train(tr_bags, tr_bags_labels, tr_bags, tr_bags_labels, model, config)
]


# %%
POSITIVE_CLASS = 1
BAG_COUNT = 1000
VAL_BAG_COUNT = 300
BAG_SIZE = 3
PLOT_SIZE = 3
ENSEMBLE_AVG_COUNT = 1


def create_bags(input_data, input_labels, positive_class, bag_count, instance_count):

    # Set up bags.
    bags = []
    bag_labels = []

    # Normalize input data.
    input_data = np.divide(input_data, 255.0)

    # Count positive samples.
    count = 0

    for _ in range(bag_count):

        # Pick a fixed size random subset of samples.
        index = np.random.choice(input_data.shape[0], instance_count, replace=False)
        instances_data = input_data[index]
        instances_labels = input_labels[index]

        # By default, all bags are labeled as 0.
        bag_label = 0

        # Check if there is at least a positive class in the bag.
        if positive_class in instances_labels:

            # Positive bag will be labeled as 1.
            bag_label = 1
            count += 1

        bags.append(instances_data)
        bag_labels.append(np.array([bag_label]))

    print(f"Positive bags: {count}")
    print(f"Negative bags: {bag_count - count}")

    return (list(np.swapaxes(bags, 0, 1)), np.array(bag_labels))


# Load the MNIST dataset.
(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()

# Create training data.
train_data, train_labels = create_bags(
    x_train, y_train, POSITIVE_CLASS, BAG_COUNT, BAG_SIZE
)

# Create validation data.
val_data, val_labels = create_bags(
    x_val, y_val, POSITIVE_CLASS, VAL_BAG_COUNT, BAG_SIZE
)
