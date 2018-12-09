import numpy as np

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils


def crop_img(img, height, width, cropped_height, cropped_width):

    left = (width - cropped_width) / 2
    top = (height - cropped_height) / 2
    right = (width + cropped_width) / 2
    bottom = (height + cropped_height) / 2
    img = img.crop((left, top, right, bottom))

    return img


def image_n_label_generator(list_of_files, labels_dict, no_classes,
                            resize_height=512, resize_width=512,
                            cropped_height=448, cropped_width=448):
    file_counter = 0
    np.random.shuffle(list_of_files)

    while True:

        if file_counter == len(list_of_files):
            file_counter = 0
            np.random.shuffle(list_of_files)  # So that images are taken up in refreshed order on each epoch

        filename = list_of_files[file_counter]
        file_counter += 1
        label_value = labels_dict[filename]
        one_hot_label = np_utils.to_categorical(label_value, no_classes)
        try:
            img = load_img(filename)
            img = img.resize((resize_width, resize_height))
            img = crop_img(img, resize_height, resize_width, cropped_height, cropped_width)
            img = img_to_array(img)
        except Exception as exc:
            print('Exception from image n label generator: ', str(exc))
            raise exc

        yield (img, one_hot_label)


def group_by_batch(dataset, batch_size):
    while True:
        sources, targets = zip(*[next(dataset) for i in range(batch_size)])
        sources_array = np.stack(sources)
        targets_array = np.stack(targets)

        sources_array = preprocess_input(sources_array)  # Image-net specific pre-processing
        batch = (sources_array, targets_array)
        yield batch


def load_dataset(files_list, labels_list, batch_size, no_classes, resize_height=512, resize_width=512,
                 cropped_height=448, cropped_width=448):

    labels_dict = dict(zip(files_list, labels_list))

    generator = image_n_label_generator(files_list, labels_dict, no_classes, resize_height, resize_width,
                                        cropped_height, cropped_width)

    generator = group_by_batch(generator, batch_size)
    return generator
