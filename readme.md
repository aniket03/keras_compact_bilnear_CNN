# Keras Implementation of Compact Bilinear Pooling

This repository contains the keras implementation of Compact Bilinear CNN. Compact bilinear pooling was first
introduced in paper [Compact biinear pooling](https://arxiv.org/pdf/1511.06062.pdf) and gives significant
performance on fine-grained image classification tasks such as bird species classification and aeroplane type
categorization.

## Usage
1. Download CUB_200_2011 dataset from [here](www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz).
   Extract the contents in folder `data/CUB_200_2011`
2. Download the VGG16 weights file from [here](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5)
   in the working directory
3. For training only the last fully connected layer of the network, use
    `python train_cbcnn_last.py 0  # If training without GPU`
    `python train_cbcnn_last.py 1  # If training with GPU`
4. For training the complete net after the last layer has been tuned, use:
    `python train_cbcnn_all.py 0  # If training without GPU`
    `python train_cbcnn_all.py 0  # If training with GPU`
5. For testing the trained model, run
    `python test_model.py`

## More on Applications of Compact bilinear CNN
At [Squad](https://www.squadplatform.com/), we further utilized Compact bilinear CNN for solving complex use cases
like apparel items classification, and retrieval of similar item from in-shop and street-to-shop domain. The proposed
pipeline and results are presented in the research paper [Fine-grained Apparel Classification and Retrieval
without rich annotations] (https://arxiv.org/abs/1811.02385)

## References
If you utilize Compact bilinear CNN for your research then pls refer
`Gao, Y., Beijbom, O., Zhang, N., & Darrell, T. (2016). Compact bilinear pooling.
In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 317-326).`

Additionally if you utilize Compact bilinear CNN for solving Apparel items categorization or related use case, pls refer
`Bhatnagar, A., & Aggarwal, S. (2018). Fine-grained Apparel Classification and Retrieval without rich annotations.
arXiv preprint arXiv:1811.02385.`

## Acknowledgments
I have referred Compact bilinear pooling implementation in Caffe by the authors [here](https://github.com/gy20073/compact_bilinear_pooling)
and tensorflow implementation [here](https://github.com/ronghanghu/tensorflow_compact_bilinear_pooling)