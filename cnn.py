### TF - MMNIST Fully Conntected NN

def train_mnist():

    # Class for callback
    class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.99):
          print("\nReached 99% accuracy so cancelling training!")
          self.model.stop_training = True
    callbacks = myCallback()

    # data loading
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    x_train = x_train/255
    x_test = x_test/255

    # model definition
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # model compiling
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    # model fitting
    return history.epoch, history.history['acc'][-1]


## TF - CNN


def train_mnist_conv():

    # Class for callback
    class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.998):
          print("\nReached 99% accuracy so cancelling training!")
          self.model.stop_training = True
    callbacks = myCallback()    

    # data loading
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    training_images=training_images.reshape([60000, 28, 28, 1])
    training_images=training_images / 255.0
    test_images = test_images.reshape([10000, 28, 28, 1])
    test_images=test_images/255.0
    
    # model definition
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),

        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.Dense(10, activation='softmax')

        ])
    # model compiling
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # model fitting
    history = model.fit(training_images,training_labels, epochs=20,callbacks=[callbacks])

    return history.epoch, history.history['acc'][-1]

## TF - CNN Using Data Generator


from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_happy_sad_model():

    DESIRED_ACCURACY = 0.999
    class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>DESIRED_ACCURACY):
          print("\nReached 99% accuracy so cancelling training!")
          self.model.stop_training = True
    callbacks = myCallback()


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),

        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
        

    model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])
        

    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1/255)
    train_generator = train_datagen.flow_from_directory(
            '/tmp/h-or-s',  # This is the source directory for training images
            target_size=(150, 150),  # All images will be resized to 150x150
            batch_size=128,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')


    # model fitting
    history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1,
    callbacks=[callbacks])

    return history.history['acc'][-1]
