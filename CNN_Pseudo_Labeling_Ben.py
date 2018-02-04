import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

def get_scaled_imgs(df):
    imgs = []

    for i, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2

        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)

# Initilize control training set

df_train1 = pd.read_json('train.json')

Xtrain1 = get_scaled_imgs(df_train1)
Ytrain1 = np.array(df_train1['is_iceberg'])

df_train1.inc_angle = df_train1.inc_angle.replace('na', 0)
idx_tr = np.where(df_train1.inc_angle > 0)

Ytrain1 = Ytrain1[idx_tr[0]]
Xtrain1 = Xtrain1[idx_tr[0], ...]

# Define common CNN model

def getModel():
    # Build keras model

    model = Sequential()

    # CNN 1
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    # CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    # You must flatten the data for the dense layers
    model.add(Flatten())

    # Dense 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    # Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    # Output
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

model = getModel()
model.summary()

batch_size = 32
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

# Predict on testing set

history1 = model.fit(Xtrain1, Ytrain1, batch_size=batch_size, epochs=3, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)

# Original submission

df_test = pd.read_json('test.json')
df_test.inc_angle = df_test.inc_angle.replace('na',0)
Xtest = (get_scaled_imgs(df_test))
pred_test = model.predict(Xtest)

# submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
# submission.to_csv('submission_original.csv', index=False)

# Initilize pseudo-labeled training set

# df_train21 = df_test
# df_train21['is_iceberg'] = np.around(pred_test)

# Halve the pseudo-labeled training set

df_train21 = df_test
df_train21['is_iceberg'] = np.around(pred_test)
df_train21 = np.array_split(df_train21, 2)[0]

# Don't round testing labels

# df_train21['is_iceberg'] = pred_test

df_train2 = pd.concat([df_train1, df_train21])

Xtrain2 = get_scaled_imgs(df_train2)
Ytrain2 = np.array(df_train2['is_iceberg'])

df_train2.inc_angle = df_train2.inc_angle.replace('na', 0)
idx_tr = np.where(df_train2.inc_angle > 0)

Ytrain2 = Ytrain2[idx_tr[0]]
Xtrain2 = Xtrain2[idx_tr[0], ...]

# Re-train the model on the larger training-set

model = getModel()
model.summary()

batch_size = 32
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

history2 = model.fit(Xtrain2, Ytrain2, batch_size=batch_size, epochs=3, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)

# Pseudo-labeling submission

# pred_test2 = model.predict(Xtest)
#
# submission2 = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test2.reshape((pred_test2.shape[0]))})
# submission2.to_csv('submission_pseudo_labeling.csv', index=False)

# Two plot test of psuedo-labeling approach

fig = plt.figure()
plt.plot(history1.history['acc'])
plt.plot(history2.history['acc'])
plt.plot(history1.history['loss'])
plt.plot(history2.history['loss'])
plt.title('Effect of Pseudo-Labeling on Model Performance')
plt.ylabel('Accuracy/Loss')
plt.xlabel('epoch')
plt.legend(['Original Accuracy', 'Pseudo-labled Accuracy', 'Original Loss', 'Pseudo-labled Loss'], loc='lower left')
fig.savefig('performance_compared_4_epoch_test_half.png')