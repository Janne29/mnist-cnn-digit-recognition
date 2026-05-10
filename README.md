# CNN zur Erkennung handgeschriebener Ziffern

Dieses Projekt entstand im Rahmen meiner Facharbeit zum Thema neuronale Netze und Bilderkennung.

Ziel des Projekts ist es, ein Convolutional Neural Network (CNN) zu trainieren, das handgeschriebene Ziffern aus dem [MNIST-Datensatz](http://yann.lecun.com/exdb/mnist/) erkennt. Dabei werden die Bilddaten geladen, normalisiert, mit einem neuronalen Netz verarbeitet und anschließend ausgewertet.

## Projektinhalt

- Laden des MNIST-Datensatzes
- Normalisierung der Bilddaten
- Aufbau eines CNN mit Keras
- Training mit Validierungsdaten
- Speichern des trainierten Modells
- Auswertung mit Accuracy, Loss, Konfusionsmatrix und Fehlklassifikationen

## Modellarchitektur

Das Modell besteht aus zwei Convolutional-Layern mit Max-Pooling, gefolgt von einem Dense-Layer und einer Softmax-Ausgabe für die zehn Ziffernklassen.

Aufbau:

```txt
Input: 28x28x1
Conv2D(32) + ReLU
MaxPooling2D
Conv2D(64) + ReLU
MaxPooling2D
Flatten
Dense(128) + ReLU
Dense(10) + Softmax
```

## Datensatz

Für das Projekt wird der [MNIST-Datensatz](http://yann.lecun.com/exdb/mnist/) verwendet. Dieser ist nicht im Repository enthalten und muss lokal im Ordner `MNIST/` abgelegt werden.

Erwartete Ordnerstruktur:

```txt
MNIST/
  train-images.idx3-ubyte
  train-labels.idx1-ubyte
  t10k-images.idx3-ubyte
  t10k-labels.idx1-ubyte
```

## Installation

```bash
pip install -r requirements.txt
```