import h5py
import numpy as np
import os
import pandas as pd
from keras.engine import topology
from sklearn.metrics import accuracy_score

from data_helpers import load_dataset
from vgg_cbcnn import vgg_16_cbcnn


if __name__=='__main__':

    # Important constants
    batch_size = 500
    no_classes = 200
    resize_height = 256
    resize_width = 256
    cropped_height = 224
    cropped_width = 224
    model_file_path = 'ft_all_layer.hdf5'
    test_file_name = 'test_images.txt'
    base_dir = 'data/CUB_200_2011/images'

    cbcnn_model = vgg_16_cbcnn(input_shape=(cropped_height, cropped_width, 3), no_classes=no_classes,
                                   bilinear_output_dim=8192, sum_pool=True, weight_decay_constant=5e-6,
                                   multi_label=False, weights_path=None)

    # Initialize model with weights from trained model
    with h5py.File(model_file_path, mode='r') as f:
        topology.load_weights_from_hdf5_group(f['model_weights'], cbcnn_model.layers)

    # Get necessary data from raw csvs
    test_mat = pd.read_csv(test_file_name, sep=' ').as_matrix()
    test_filenames = test_mat[:, 0]
    test_labels = test_mat[:, 1]
    test_count = len(test_filenames)

    # Get test data set generator
    test_file_paths = [os.path.join(base_dir, test_file) for test_file in test_filenames]
    test_generator = load_dataset(test_file_paths, test_labels, batch_size, no_classes, resize_height, resize_width,
                                  cropped_height, cropped_width)

    # Get test predictions
    cnt = 0
    test_batches = int(np.ceil(test_count / batch_size))

    actual_labels = []
    predicted_labels = []
    for test_batch_tuple in test_generator:
        if cnt == test_batches:
            break

        test_data = test_batch_tuple[0]
        test_labels = test_batch_tuple[1]

        predicted_batch_labels = cbcnn_model.predict(test_data)

        actual_labels += [np.argmax(test_label) for test_label in test_labels]
        predicted_labels += [np.argmax(predicted_batch_label) for predicted_batch_label in predicted_batch_labels]

        cnt += 1

    print ("Classification accuracy", accuracy_score(actual_labels, predicted_labels))