import pandas as pd # Used to open CSV files 
import numpy as np # Used for matrix operations
import cv2 # Used for image augmentation
from matplotlib import pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
np.random.seed(666)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import KFold 
 
def get_scaled_imgs(df):
    imgs = []
    
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)


df_train = pd.read_json('train.json') # this is a dataframe
df_test4pseudo = pd.read_json('test.json')

df_lab = pd.read_csv('submission54.csv')
df_test4pseudo['is_iceberg'] = df_lab.is_iceberg

df_t4pf = df_test4pseudo[(df_test4pseudo.is_iceberg >= .98) | (df_test4pseudo.is_iceberg <= .02)]

Ypseudo = df_t4pf.is_iceberg.round()

Xtrain1 = get_scaled_imgs(df_train)

df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)

Xtrain1 = Xtrain1[idx_tr[0],...]
Ytrain1 = np.array(df_train['is_iceberg'])
Ytrain1 = Ytrain1[idx_tr[0]]

proportion = .1
Xval = Xtrain1[:int((proportion)*len(Xtrain1))]
Yval = Ytrain1[:int((proportion)*len(Ytrain1))]

Xtrain2 = Xtrain1[int((proportion)*len(Xtrain1)):]
Ytrain2 = Ytrain1[int((proportion)*len(Ytrain1)):]

Xpt1 = get_scaled_imgs(df_t4pf)
Ypt1 = np.array(Ypseudo)

df_t4pf.inc_angle = df_t4pf.inc_angle.replace('na',0)
idx_tr = np.where(df_t4pf.inc_angle>0)

Xpt1 = Xpt1[idx_tr[0],...]
Ypt1 = Ypt1[idx_tr[0]]
 
Xtrain = np.concatenate((Xtrain2, Xpt1))
Ytrain = np.concatenate((Ytrain2, Ypt1))

def get_more_images_imG(imgs):
    # define the augmentations
    seq1 = iaa.Sequential([
            iaa.Sometimes(0.5, 
                          iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.ContrastNormalization((0.75, 1.5)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)
            ], random_order=True) #apply the augmenters in random order
             
    seq2 = iaa.Sequential([
            iaa.CropAndPad(
                    percent=(-0.05, 0.05),
                    pad_mode=ia.ALL,
                    pad_cval=(0,255))])
    
    seq3 = iaa.Sequential([
            iaa.Fliplr(1.0)]) # horizontally flip the images
    
    seq4 = iaa.Sequential([
            iaa.Flipud(1.0)])
    
    seq5 = iaa.Sequential([
            iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45), # rotate by -45 to +45 degrees
                    shear=(-16, 16), # shear by -16 to +16 degrees
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                   cval=(0, 255))]) # if mode is constant, use a cval between 0 and 255)
    
    
    return np.concatenate((imgs, seq3.augment_images(imgs), seq4.augment_images(imgs)))


Xtr_more = get_more_images_imG(Xtrain)
Ytr_more = np.concatenate((Ytrain, Ytrain, Ytrain))
#Ytr_more = np.concatenate((Ytrain, Ytrain, Ytrain))
#Ytr_more = np.concatenate((Ytrain, Ytrain))

def getModel():
    #Build keras model
    
    model=Sequential()
    
    # CNN 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    #CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    # You must flatten the data for the dense layers
    model.add(Flatten())

    #Dense 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    #Dense 2
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

#------------------------------------------------------------------------------------------------------------------------------------------------------
# Let's view progress 
history = model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=50, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], shuffle=True, validation_data=(Xval, Yval))

print(history.history.keys())
#
fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower left')
#
fig.savefig('performance.png')
#---------------------------------------------------------------------------------------

model.load_weights(filepath = '.mdl_wts.hdf5')

score = model.evaluate(Xtrain, Ytrain, verbose=1)
print('Train score:', score[0])
print('Train accuracy:', score[1])
score = model.evaluate(Xval, Yval, verbose=1)
print('Eval score:', score[0])

df_test = pd.read_json('test.json')
df_test.inc_angle = df_test.inc_angle.replace('na',0)
Xtest = (get_scaled_imgs(df_test))
pred_test = model.predict(Xtest)

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
print(submission.head(10))

submission.to_csv('submission.csv', index=False)
