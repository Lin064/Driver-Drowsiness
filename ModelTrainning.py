import tensorflow as tf
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPool2D, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.ops.init_ops_v2 import glorot_uniform


def trainModel():
    batchsize = 8
    train_dir = r'./data/Train'
    test_dir = r'./data/Test'
    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=0.2, shear_range=0.2,
                                       zoom_range=0.2, width_shift_range=0.2,
                                       height_shift_range=0.2, validation_split=0.2)

    train_data = train_datagen.flow_from_directory(train_dir,
        target_size=(80, 80), batch_size=batchsize, class_mode='categorical', subset='training')

    validation_data = train_datagen.flow_from_directory(train_dir,
        target_size=(80, 80), batch_size=batchsize, class_mode='categorical', subset='validation')

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_data = test_datagen.flow_from_directory(test_dir,
                                    target_size=(80,80),batch_size=batchsize,class_mode='categorical')
    #call back
    filepath = r'./model/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                    monitor='val_loss',
                                                    verbose=0,
                                                    save_best_only=False,
                                                    save_weights_only=False,
                                                    mode='auto',
                                                    save_freq='epoch',
                                                    period=1)
    learning_rate = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=3)
    callbacks = [checkpoint, learning_rate]
    model = getMode()
    model.fit_generator(train_data,
                        steps_per_epoch=train_data.samples // 8,
                        validation_data=validation_data,
                        validation_steps=validation_data.samples // 8,
                        callbacks=callbacks,
                        epochs=20)

def getMode():
    model = Sequential()
    #model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding='same', strides=(1, 1), name='conv1', activation='relu',input_shape=(80,80,3),
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', strides=(1, 1), name='conv2', activation='relu',
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv3', activation='relu',
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv4', activation='relu',
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv5', activation='relu',
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv6', activation='relu',
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv7', activation='relu',
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=glorot_uniform(seed=0), name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer=glorot_uniform(seed=0), name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', kernel_initializer=glorot_uniform(seed=0), name='fc3'))


    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

if __name__ == '__main__':
    trainModel()
