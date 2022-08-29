```python
import tensorflow as tf #Red neuronal convolucional
import pandas as pd #Visualizar data
import numpy as np #La libreria MNIST con los datos esta en formato numpy
import matplotlib.pyplot as plt #Graficas
from tensorflow.keras.models import Sequential #Tipo de modelo secuelcial, para agruegar capas
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
#Dense para crear capas full conected (RNFC)
#Dropout para evitar el overfiting, desactivando un porcetaje de neuronas determinado
#Flatten para convetir las salidas en un vector 1D
#Conv2D para relizar operaciones de convolucion
#MaxPooling para realizar operaciones de Maxpooling
```


```python
mnist_data = tf.keras.datasets.mnist

#data de entrenamiento y data de testeo
(train_images, train_labels),(test_images, test_labels) = mnist_data.load_data()

#este set de datos ya viene con data de entrenamiento y de test separados
print("Dimensiones del set de entrenamiento:", train_images.shape)
print("Imagenes de entrenamiento:", train_images.shape[0])
print("Imagenes de testeo:", test_images.shape[0])
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11490434/11490434 [==============================] - 4s 0us/step
    Dimensiones del set de entrenamiento: (60000, 28, 28)
    Imagenes de entrenamiento: 60000
    Imagenes de testeo: 10000
    


```python
#Variables

#cantidad de digitos a clasificar
num_classes = 10
#tamanio de cada subconjunto (para no agarra las 60000 y llenar la memoria)
batch_size = 128
#cuanta veces va a recorrer todo el cojunto de entrenamiento
epochs = 5
#forma de las imagenes
input_shape = (28, 28, 1)
```


```python
#Nomalizamos los valores entre 0 y 1
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255
print("ValMinTR", np.amin(train_images))
print("ValMinTE", np.amin(test_images))
print("ValMaxTR", np.amax(train_images))
print("ValMaxTE", np.amax(test_images))
print("")

#Establecemos el numero de canales en 1 ya que la imgen esta en escala de grises
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
print("Dim", train_images.shape)
print("Dim", test_images.shape)
print("")

#Convertimos los vectores de clase en matrices binarias
print(test_labels)
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)
print(test_labels)
```

    ValMinTR 0.0
    ValMinTE 0.0
    ValMaxTR 1.0
    ValMaxTE 1.0
    
    Dim (60000, 28, 28, 1)
    Dim (10000, 28, 28, 1)
    
    [7 2 1 ... 4 5 6]
    [[0. 0. 0. ... 1. 0. 0.]
     [0. 0. 1. ... 0. 0. 0.]
     [0. 1. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]
    


```python
#Creacion del modelo
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), #32 filtros de 3x3
                 activation="relu", 
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation="relu")) #64 filtros de 3x3
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) #Flatten para covertir el vector en uno unidemensional
model.add(Dropout(0.5)) #Apagamo el 50% de las neuronas para reforzar el aprendisaje
model.add(Dense(num_classes, activation="softmax")) 
#esto hara que cada salida tenga un valor entre 0 y 1 de tipo probabilistico

model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                     
     max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 11, 11, 64)        18496     
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)         0         
     2D)                                                             
                                                                     
     flatten (Flatten)           (None, 1600)              0         
                                                                     
     dropout (Dropout)           (None, 1600)              0         
                                                                     
     dense (Dense)               (None, 10)                16010     
                                                                     
    =================================================================
    Total params: 34,826
    Trainable params: 34,826
    Non-trainable params: 0
    _________________________________________________________________
    


```python
#Compilamos el modelo
model.compile(loss="categorical_crossentropy", 
              optimizer="adam", 
              metrics=["accuracy"])

#Entrenamiento
entrenado = model.fit(train_images, train_labels, 
          batch_size=batch_size, 
          epochs=epochs, 
          verbose=1)
```

    Epoch 1/5
    469/469 [==============================] - 22s 45ms/step - loss: 0.3489 - accuracy: 0.8935
    Epoch 2/5
    469/469 [==============================] - 22s 47ms/step - loss: 0.1093 - accuracy: 0.9660
    Epoch 3/5
    469/469 [==============================] - 22s 48ms/step - loss: 0.0800 - accuracy: 0.9752
    Epoch 4/5
    469/469 [==============================] - 21s 46ms/step - loss: 0.0681 - accuracy: 0.9789
    Epoch 5/5
    469/469 [==============================] - 22s 46ms/step - loss: 0.0606 - accuracy: 0.9810
    


```python
#Validation data
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f"TEST LOSS: {test_loss}")
print(f"TEST ACCURACY: {test_accuracy}")
print("")  

#Historial
frame = pd.DataFrame(entrenado.history)
print("PRECISION POR EPOCAS")
print(frame)
```

    313/313 [==============================] - 2s 5ms/step - loss: 0.0349 - accuracy: 0.9889
    TEST LOSS: 0.03494371846318245
    TEST ACCURACY: 0.9889000058174133
    
    PRECISION POR EPOCAS
           loss  accuracy
    0  0.348861  0.893550
    1  0.109256  0.965950
    2  0.079964  0.975183
    3  0.068063  0.978850
    4  0.060597  0.981033
    


```python

```


```python

```
