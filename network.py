from keras import layers, models, optimizers
import json

from mnist import load_mnist
from plot import plot_accuracy, plot_loss, plot_confusion_matrix, plot_misclassified



class Network():

    """
    Modell und Trainingsdaten initialisieren
    
    """

    def __init__(self):

        # Datensatz laden

        train, test = load_mnist()

        self.train_images, self.train_labels = train
        self.test_images, self.test_labels = test

        # Validation Data: Teil des Trainingssatzes

        val_size = 10000

        self.val_images = self.train_images[-val_size:]
        self.val_labels = self.train_labels[-val_size:]

        self.train_images = self.train_images[:-val_size]
        self.train_labels = self.train_labels[:-val_size]

        # Modell-Architektur (CNN mit Convolution-Layers und Pooling)
        
        self.model = models.Sequential([

            layers.Input(shape=(28, 28, 1)),

            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),

        ])

        # Modell konfigurieren

        self.configure()


    """
    Modell konfigurieren (Fehlerfunktion, Metriken etc. definieren)
    
    """
    
    def configure(self):

        self.model.compile(
            loss = "sparse_categorical_crossentropy",
            optimizer = optimizers.Adam(learning_rate=1e-3),
            metrics = ["accuracy"]
        )


    """
    Training des Modells
    
    """
    
    def train(self, epochs, batch_size):

        # Training des Modells

        history = self.model.fit(
            self.train_images, self.train_labels,
            batch_size = batch_size,
            epochs = epochs,
            validation_data = (self.val_images, self.val_labels),
            verbose = 2
        )

        # Speichern des Trainingsverlaufs

        with open("history.json", "w", encoding="utf-8") as file:
            json.dump(history.history, file)

        return history
    

    """
    Testen des Modells

    """

    def test(self):

        test_loss, test_accuracy = self.model.evaluate(
            self.test_images,
            self.test_labels,
            verbose = 2
        )

        return test_loss, test_accuracy


    """
    Speichern des Modells

    """

    def save(self):
        self.model.save("mnist_cnn.keras")


    """
    Laden des bereits bestehenden Networks
    
    """

    def load(self):
        self.model = models.load_model("mnist_cnn.keras")


"""
Ausführung des Modells

"""

if __name__ == "__main__":

    network = Network()
    network.train(10, 128)
    network.save()
    print(network.test())

    plot_accuracy()
    plot_loss()
    plot_confusion_matrix()
    plot_misclassified()

