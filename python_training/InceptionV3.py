import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input, InceptionV3
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
# import tensorflowjs as tfjs

from sklearn.metrics import accuracy_score, roc_auc_score

BATCH_SIZE = 8
TARGET_SIZE = (299, 299)
PATH_TRAIN = 'D:/CREAR_APLICACIONES/04_CUARTA_SEMANA/deep_learning_Chest_X_Ray/Images_train_val_test/train'
PATH_VAL = 'D:/CREAR_APLICACIONES/04_CUARTA_SEMANA/deep_learning_Chest_X_Ray/Images_train_val_test/val'
PATH_TEST = 'D:/CREAR_APLICACIONES/04_CUARTA_SEMANA/deep_learning_Chest_X_Ray/Images_train_val_test/test'
EPOCH_INIT = 5
EPOCH_END = 200
FILE_MODEL = 'D:/CREAR_APLICACIONES/04_CUARTA_SEMANA/deep_learning_Chest_X_Ray/server_flask/models/Inception_V3'


def training():
    datagen = ImageDataGenerator(width_shift_range=0.4,
                                 height_shift_range=0.4,
                                 shear_range=0.4,
                                 zoom_range=0.4,
                                 horizontal_flip=True,
                                 vertical_flip=False,
                                 preprocessing_function=preprocess_input)

    train_generator = datagen.flow_from_directory(
        PATH_TRAIN,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True)

    val_generator = datagen.flow_from_directory(
        PATH_VAL,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True)

    test_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
        PATH_TEST,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False)

    basic_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')

    for layer in basic_model.layers:
        layer.trainable = False

    input_tensor = basic_model.input

    x = basic_model.output
    x = BatchNormalization()(x)
    x = Dropout(.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_tensor, outputs=x)
    model.compile(optimizer=Adam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(train_generator,
                        epochs=EPOCH_INIT,
                        steps_per_epoch=train_generator.n / (BATCH_SIZE))

    for layer in model.layers:
        layer.W_regularizer = l2(1e-3)
        layer.trainable = True

    model = Model(inputs=input_tensor, outputs=x)
    model.compile(optimizer=Adam(lr=1e-4, decay=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath=FILE_MODEL, verbose=1, save_best_only=True)
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='auto', min_delta=1e-4)

    model.fit_generator(train_generator,
                        epochs=EPOCH_END,
                        validation_data=val_generator,
                        steps_per_epoch=train_generator.n / BATCH_SIZE,
                        validation_steps=val_generator.n / BATCH_SIZE,
                        callbacks=[checkpointer, earlyStopping],
                        verbose=1,
                        initial_epoch=EPOCH_INIT,
                        shuffle=True)

    best_model = load_model(FILE_MODEL)

    pred_test = best_model.predict_generator(test_generator, steps=test_generator.n / BATCH_SIZE, verbose=1)
    class_test = test_generator.classes
    class_predict = (pred_test > 0.5).astype(int).reshape((len(pred_test), 1))

    print('Accuracy: ' + str(accuracy_score(class_test, class_predict)))
    print('AUC: ' + str(roc_auc_score(class_test, pred_test)))

    print('Model complete!!')

def convert_keras_to_js():
    model = load_model(FILE_MODEL)
    tfjs.converters.save_keras_model(model,
                                     'D:/CREAR_APLICACIONES/04_CUARTA_SEMANA/deep_learning_Chest_X_Ray/tensorflow.js/model_js')
    print('Exit!!!')


if __name__ == '__main__':
    training()
    convert_keras_to_js()
