import json
import os
import matplotlib.pyplot as plt
import numpy as np
from keras import models

from sklearn.metrics import confusion_matrix
from mnist import load_mnist

os.makedirs("plots", exist_ok=True)


"""
Laden der Trainingshistorie

"""

def load_history():
    with open("history.json", "r", encoding="utf-8") as file:
        return json.load(file)



"""
Vorhersage der Testlabels

"""

def predict():

    model = models.load_model("mnist_cnn.keras")

    train, test = load_mnist()

    test_images, test_labels = test

    test_predictions = np.argmax(model.predict(test_images), axis=1)

    return test_labels, test_predictions, test_images



"""
Graphische Darstellung des Fehlers pro Epoche

"""

def plot_loss(save = True):

    history = load_history()

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(6,4))
    plt.plot(history["loss"], label = "Trainingsfehler", linewidth = 2)
    
    plt.xlabel("Epoche", fontsize=12)
    plt.ylabel("Fehler", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if save: plt.savefig("plots/loss.png", dpi=300)

    plt.show()
    plt.close()



"""
Graphische Darstellung der Genauigkeit pro Epoche

"""

def plot_accuracy(save = True):

    history = load_history()

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(6,4))
    plt.plot(history["accuracy"], label = "Trainingsgenauigkeit", linewidth = 2)
    
    plt.xlabel("Epoche", fontsize=12)
    plt.ylabel("Genauigkeit", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if save: plt.savefig("plots/accuracy.png", dpi=300)

    plt.show()
    plt.close()



"""
Graphische Darstellung einer Konfusionsmatrix

"""

def plot_confusion_matrix(save = True):

    test_labels, test_predictions, test_images = predict()

    matrix = confusion_matrix(test_labels, test_predictions)

    plt.figure(figsize=(8,6))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Konfusionsmatrix")

    plt.xlabel("Predicted Label")
    plt.ylabel("Correct Label")

    plt.colorbar()

    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))

    plt.tight_layout()

    if save: plt.savefig("plots/confusion_matrix.png", dpi=300)

    plt.show()
    plt.close()



"""
Graphische Darstellung von Fehlklassifikationen

"""

def plot_misclassified(save = True):

    test_labels, test_predictions, test_images = predict()

    misclassified = np.where(test_labels != test_predictions)[0]

    plt.figure(figsize=(10,6))

    for i, index in enumerate(misclassified[:15]):

        plt.subplot(3, 5, i+1)

        plt.imshow(test_images[index].reshape(28,28), cmap = "grey")

        plt.title(f"T:{test_labels[index]} P:{test_predictions[index]}")
        plt.axis("off")

    plt.suptitle("Fehlklassifikationen")
    plt.tight_layout()

    if save: plt.savefig("plots/misclassified.png", dpi=300)

    plt.show()
    plt.close()





