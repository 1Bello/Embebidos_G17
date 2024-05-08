
import os
import zipfile
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
if tf.__version__.split('.')[0] != '2':
    raise Exception((f"The script is developed and tested for tensorflow 2. "
                     f"Current version: {tf.__version__}"))

if sys.version_info.major < 3:
    raise Exception((f"The script is developed and tested for Python 3. "
                     f"Current version: {sys.version_info.major}"))

local_zip = "dataset.zip"
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall("")  
zip_ref.close()

train_pick_dir = 'training\cars_pick'
train_sport_dir = 'training\cars_sport'
validation_pick_dir = 'validation\cars_pick'
validation_sport_dir = 'validation\cars_sport'

train_pick_names = os.listdir(train_pick_dir)
print(train_pick_names[:10])
train_sport_names = os.listdir(train_sport_dir)
print(train_sport_names[:10])

# Print the lengths of the datasets
print(len(train_pick_names))
print(len(train_sport_names))


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)), # Notar el cambio de input con respecto a los modelos que habiamos estudiado antes
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), # Misma estructura de la clase pasada primero extraemos caracterisrticas y despues pasamos por layers dense
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Solo una neurona de salida porque estamos clasificando entre humanos y caballos
])
print(model.summary()) # Notar ya el tamano de la red


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])


# Vamos a usar las siguientes tecnicas para aumentar el dataset:
train_datagen = ImageDataGenerator( # no es que genere imagenes sino que le aplica una transformacion cada vez que la imagen es llamada
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
      )

train_generator = train_datagen.flow_from_directory(
        'training',  # Vamos directo a la carpeta con cabllos o humanos
        target_size=(100, 100),
        batch_size=128, # El batch nos dice cada cuanto tiempo se van a actualizar los pesos, si no se define es 1 y se actualizan despues de cada imagen. Mas rapido y tomamos promedios. Para considerarse una epoca tenemos que pasar por todos los btchs
        class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
        'validation',
        target_size=(100, 100),
        class_mode='binary')


NUM_EPOCHS = 100
history = model.fit(
      train_generator,
      steps_per_epoch=8, # cuántos batches del conjunto de entrenamiento se utilizan para entrenar el modelo durante una epoch. ceil(num_samples / batch_size)
      epochs=NUM_EPOCHS,
      verbose=1,
      validation_data=validation_generator)


# Buena practica y buen grafico para usar en su presentacion
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.xlim([0,NUM_EPOCHS])
plt.ylim([0.4,1.0])
plt.show()