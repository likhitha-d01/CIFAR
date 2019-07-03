import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
#import required modules
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from CIFAR_functions import *
folder_path= 'C:\\Users\\dell\\Desktop\\IISC\\cifar-10-batches-py'

#Exploring the dataset
batch_id = 3
sample_id = 7000
display_stats(folder_path, batch_id, sample_id)

# Preprocess all the data and save it
preprocess_and_save_data(folder_path, normalize, one_hot_encode)

# load the saved dataset
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

#Deciding the training batch size
df_train_x = valid_features[:2000,:]
df_train_y = valid_labels[:2000,:]

#Loading the label names
label = load_label_names();

#Loading the test data and normalizing it
test_x, test_y = load_cfar10_test(folder_path)
test_x = normalize(test_x)

#Plotting the first five examples in the training data
ax = plt.subplots(1,5)
for i in range(0,5):   #validate the first 5 records
    ax[1][i].imshow(df_train_x[i].reshape(32,32,3), cmap='gray')
    for j in range(0,len(df_train_y[i])):
        if df_train_y[i,j]==1:
            b=j
    ax[1][i].set_title(label[b])

#Defining the CNN model
def cnn_model(result_class_size):
    model = Sequential()
    #use Conv2D to create our first convolutional layer, with 32 filters, 5x5 filter size, 
    #input_shape = input image with (height, width, channels), activate ReLU to turn negative to zero
    model.add(Conv2D(32, (5, 5), input_shape=(32,32,3), activation='relu'))
    #add a pooling layer for down sampling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # add another conv layer with 16 filters, 3x3 filter size, 
    model.add(Conv2D(16, (3, 3), activation='relu'))
    #set 20% of the layer's activation to zero, to void overfit
    model.add(Dropout(0.2))
    #convert a 2D matrix in a vector
    model.add(Flatten())
    #add fully-connected layers, and ReLU activation
    model.add(Dense(130, activation='relu'))
    model.add(Dense(50, activation='relu'))
    #add a fully-connected layer with softmax function to squash values to 0...1 
    model.add(Dense(result_class_size, activation='softmax'))   
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model

#define the model output size and get the summary
model = cnn_model(df_train_y.shape[1])
model.summary()  

random_seed = 3
#validate size = 8%
split_train_x, split_val_x, split_train_y, split_val_y, = train_test_split(df_train_x, df_train_y,
                                               test_size = 0.08,
                                               random_state=random_seed)

#define model callback
reduce_lr = ReduceLROnPlateau(monitor='val_acc',factor=0.5,patience=3,min_lr=0.00001)
callbacks_list=[reduce_lr]

#define image generator
datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range 
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1  # randomly shift images vertically
        )

datagen.fit(split_train_x)

#train the model with callback and image generator
model.fit_generator(datagen.flow(split_train_x,split_train_y, batch_size=50),epochs = 10, 
validation_data = (split_val_x,split_val_y),verbose = 0, steps_per_epoch=20, 
callbacks=callbacks_list)

#predict the result and save it as a csv file
prediction = model.predict_classes(test_x, verbose=0)
data_to_submit = pd.DataFrame({"ImageId": list(range(1,len(prediction)+1)), "Label": prediction})
data_to_submit.to_csv("results.csv", header=True, index = False)

#validate the result by our own eyes
from random import randrange
#pick 10 images from testing data set
start_idx = randrange(test_x.shape[0]-10) 
  
fig, ax = plt.subplots(2,5, figsize=(15,8))
for j in range(0,2): 
  for i in range(0,5):
     ax[j][i].imshow(test_x[start_idx].reshape(32,32,3), cmap='gray')
     ax[j][i].set_title("Index:{} \nPrediction:{} \nLabel:{}".format(start_idx, prediction[start_idx],label[test_y[start_idx]]))
     start_idx +=1
