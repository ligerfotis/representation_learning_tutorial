This is the coding part of the lecture Deep Representation Learning in PyTorch.

python 3.10

#### Autoencoder
In this demo we will implement a simple autoencoder. The autoencoder will be trained on the MNIST dataset. The autoencoder will be implemented in the file autoencoder.py. The file autoencoder.py contains a class Autoencoder on the MNIST dataset.
We compare the performance of the fully connected autoencoder with a convolutional autoencoder.
Jupyter notebooks: 
* autoencoder_mnist.ipynb
* autoencoder_cifar10.ipynb

#### Contractive Learning (SimCLR)
In this demo we implemented the SimClR [1] algorithm and trained it on the cifar10 dataset.

Download the pretrained encoder [here](https://cloud.cps.unileoben.ac.at/index.php/s/feHYqRHwDy7mMDm) and put it in the folder `runs`.

The code was adapted from this [repo](https://github.com/sthalles/SimCLR/tree/master)

Jupyter notebooks:
* simclr.ipynb


[1] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning (ICML'20), Vol. 119. JMLR.org, Article 149, 1597–1607.