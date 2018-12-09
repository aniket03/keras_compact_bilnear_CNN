import os
import sys

import functools
import numpy as np
import pandas as pd

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.metrics import top_k_categorical_accuracy
from keras.optimizers import SGD

from data_helpers import load_dataset
from vgg_cbcnn import vgg_16_cbcnn


if __name__=='__main__':

    use_gpu = sys.argv[1]
    if not int(use_gpu):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # File names
    train_file_name = 'train_images.txt'
    val_file_name = 'val_images.txt'

    # Base directory to CUB200 dataset
    base_dir = 'data/CUB_200_2011/images/'

    # Get necessary data from raw csvs
    train_mat = pd.read_csv(train_file_name, sep=' ').as_matrix()
    train_filenames = train_mat[:, 0]
    train_labels = train_mat[:, 1]

    val_mat = pd.read_csv(val_file_name, sep=' ').as_matrix()
    val_filenames = val_mat[:, 0]
    val_labels = val_mat[:, 1]

    train_file_paths = [os.path.join(base_dir, train_file) for train_file in train_filenames]
    val_file_paths = [os.path.join(base_dir, val_file) for val_file in val_filenames]

    # Important constants
    batch_size = 64
    resize_height = 256
    resize_width = 256
    cropped_height = 224
    cropped_width = 224
    no_classes = 200
    INITIAL_LR = 1.0
    train_count = 4794
    val_count = 1199
    vgg_weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    DNN_BEST_MODEL = 'ft_last_layer.hdf5'
    EPOCHS_PATIENCE_BEFORE_STOPPING = 5
    EPOCHS_PATIENCE_BEFORE_DECAY = 2

    train_generator = load_dataset(train_file_paths, train_labels, batch_size, no_classes, resize_height, resize_width,
                                   cropped_height, cropped_width)
    val_generator = load_dataset(val_file_paths, val_labels, batch_size, no_classes, resize_height, resize_width,
                                 cropped_height, cropped_width)

    # Set batches of training and validation required
    train_batches = int(np.ceil(train_count / batch_size))
    val_batches = int(np.ceil(val_count / batch_size))

    cbcnn_model = vgg_16_cbcnn(input_shape=(cropped_height, cropped_width, 3), no_classes=no_classes,
                               bilinear_output_dim=8192, sum_pool=True, weight_decay_constant=5e-6,
                               multi_label=False, weights_path=vgg_weights_path)

    # Make cbcnn model layers as non trainable [Except for last layer]
    for layer in cbcnn_model.layers[:-1]:
        layer.trainable = False
    print (cbcnn_model.summary())

    top3_acc = functools.partial(top_k_categorical_accuracy, k=3)
    top5_acc = functools.partial(top_k_categorical_accuracy, k=5)
    top3_acc.__name__ = 'top3_acc'
    top5_acc.__name__ = 'top5_acc'

    # Model training section
    sgd = SGD(lr=INITIAL_LR, momentum=0.9)
    cbcnn_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics = ['accuracy', top3_acc, top5_acc])
    check_pointer = ModelCheckpoint(monitor='val_loss', filepath=DNN_BEST_MODEL, verbose=1, save_best_only=True)
    early_stopper = EarlyStopping(monitor='val_loss', patience=EPOCHS_PATIENCE_BEFORE_STOPPING)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=EPOCHS_PATIENCE_BEFORE_DECAY,
                                             verbose=1, min_lr=1e-7)

    cbcnn_model.fit_generator(generator=train_generator, steps_per_epoch=train_batches,
                              epochs=100, verbose=1, validation_data=val_generator,
                              validation_steps=val_batches,
                              callbacks=[check_pointer, reduce_lr_on_plateau, early_stopper])
