# Keras implementation of Shake-Shake regularization

Implementation of the Shake-Shake regularization layer (https://arxiv.org/pdf/1705.07485.pdf). The layers.py file contains the ShakeShake layer, which can be used in your custom Keras models. It outputs a random weighted average of two input tensors. During backpropagation different random weights are used, disturbing the learning process and improving generalization. The models.py file contains adaptions of ResNet34 (https://arxiv.org/abs/1512.03385) with the ShakeShake layer for both the ImageNet and CIFAR-10 dataset.

![Shake-Shake regularization](https://raw.githubusercontent.com/jonnedtc/Shake-Shake-Keras/master/images/shakeshake.PNG)
<sub><sup>Figure from https://arxiv.org/pdf/1705.07485.pdf</sup></sub>

### Saving and loading

A custom model with the ShakeShake layer is saved in the same way as a regular model. However when loading the model make sure to include the layer as a custom object.

```python
import keras
from layers import ShakeShake

# save model
keras.models.save_model(model, 'filename.h5')

# load model
model = keras.models.load_model('filename.h5', custom_objects={'ShakeShake': ShakeShake})
```

### CIFAR-10

Quick test comparing this model with a similar ResNet-34 on the CIFAR-10 dataset. Both networks were trained with the Adam optimizer and cross-entropy loss for 200 epochs in batches of 128, using a learning rate of 1e-3 (decreased tenfold at epoch 120 and 160) and minor augmentation (translation and horizontal flipping).


![results](https://raw.githubusercontent.com/jonnedtc/Shake-Shake-Keras/master/images/result.png)

The Shake-Shake regularization achieved 93.44% accuracy on the CIFAR-10 test set versus 89.13% accuracy from the regular ResNet-34. However the improvement comes at the cost of almost double the model size and training time.