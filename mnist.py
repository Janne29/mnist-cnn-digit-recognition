import struct
import numpy as np

"""
Laden der Bilder

"""

def load_images(path):

    with open(path, "rb") as f:

        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)

        return images

"""
Laden der dazugehörigen Labels

"""

def load_labels(path):

    with open(path, "rb") as f:

        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels

"""
Laden des gesamten Datensatzes

"""

def load_mnist():

    path = "MNIST/"

    train_images = load_images(path + "train-images.idx3-ubyte")
    train_labels = load_labels(path + "train-labels.idx1-ubyte")

    test_images = load_images(path + "t10k-images.idx3-ubyte")
    test_labels = load_labels(path + "t10k-labels.idx1-ubyte")


    train_images = train_images[..., np.newaxis] 
    test_images = test_images[..., np.newaxis]

    train_images = train_images.astype(np.float32) / 255
    test_images = test_images.astype(np.float32) / 255

    return ( train_images, train_labels ) , ( test_images, test_labels )