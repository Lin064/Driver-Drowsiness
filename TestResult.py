import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator


def TestResult():
    train_dir = r'./data/Train'
    test_dir = r'./data/Test'
    batchsize = 8
    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=0.2, shear_range=0.2,
                                       zoom_range=0.2, width_shift_range=0.2,
                                       height_shift_range=0.2, validation_split=0.2)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_data = test_datagen.flow_from_directory(test_dir,
                                                 target_size=(80, 80), batch_size=batchsize, class_mode='categorical')
    model = tf.keras.models.load_model('./model/weights.07-0.39.hdf5')
    model.evaluate(test_data)
if __name__ == '__main__':
    TestResult()
