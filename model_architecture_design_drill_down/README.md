# Model Architecture Design Drill Down

There are three different implementations which are detailed below:

## Implementation 1

### Process

Model architecture was designed using convolution, batch normalization, max
pooling, global average pooling, and ReLU activation layers and trained without
using any data augmentation techniques. The focus was to keep the model
parameters to less than 8,000, and try to hit 99.4% test accuracy.

### Result observed

* Parameters: 7,240
* Best Train Accuracy: 99.5050
* Best Test Accuracy: 99.03

### Analysis

* Model is overfitting, need to introduce regularization by adding Dropout
  layer.

## Implementation 2

### Process

The core model architecture was maintained, but added dropout layer to reduce
overfitting.

### Result observed

* Parameters: 7,240
* Best Train Accuracy: 99.2116
* Best Test Accuracy: 99.19

### Analysis

* Model is not overfitting, but at the same time not reaching the target in the
  given number of epochs. Need to introduce data augmentation, and try varying
  learning rates and optimizers.

## Implementation 3

### Process

Keeping the model architecture same as the one used for Implementation 2,
applicable data augmentation strategies are added. The model is trained with
varying learning rates and also two optimizers - SGD and AdamW - are evaluated.

### Result observed

* Parameters: 7,240
* Best Train Accuracy: 99.52
* Best Test Accuracy: 98.93

### Analysis

The best result was observed with AdamW optimizer and a learning rate of 0.35,
but the most consistent result was observed with learning rate of 0.2 where the
target accuracy of >99.4% was achieved 4 times which was the highest.
