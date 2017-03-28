import csv
import cv2
import numpy as np
import sklearn
import gc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import common

def translate_image(img, y_value, distance=0, step=20, y_value_gain=0.20):
    # Translation of image data to create more training data
    # img : 3D image data
    # y_value : float label data for the image
    # distance : max distance of transformed images
    # step : steps between transformed images
    # y_value_gain : the gain of the label data over the transform distance
    # return : list of new images and corresponding label data

    rows, cols, _ = img.shape
    img_list = []
    y_list = []

    if distance != 0:
        gain = y_value_gain/distance
    else:
        gain = y_value_gain

    # add original un touched image
    img_list.append(img)
    y_list.append(y_value)

    # add step to include the distance in the image transform
    for offset in range(step, distance+step, step):
        M_right = np.float32([[1, 0, offset], [0, 1, 0]]) # move image right
        M_left = np.float32([[1, 0, -offset], [0, 1, 0]]) # move image left

        # shift the image to the right and append the process image to the list
        img_list.append(cv2.warpAffine(img, M_right, (cols, rows)))
        # shift the image to the left and append the process image to the list
        img_list.append(cv2.warpAffine(img, M_left, (cols, rows)))

        # change the output value varying until the final distance upto the
        # y_value_gain. => y_value + (y_value_gain * offset) / distance
        new_y = gain * offset

        y_list.append(y_value + new_y)  # add positive steer angle for a right shift
        y_list.append(y_value - new_y)  # add negative steer angle for a left shift

    return img_list, y_list

drive_data = []
steer_data = []
# Create normalised images for the training data
print('Creating the data')
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        angle = float(line[3])

        # only keep data with a steer angle
        if angle >= 0.01 or angle <= -0.01:
            for camera in range(0, 3):
                name = './data/IMG/' + line[camera].split('/')[-1]
                image = cv2.imread(name)

                # Left of image is a negative angle, Right positive
                if camera == 1:
                    # Left camera, make angle smaller to help return to center
                    angle += 0.3
                    distance = 0
                elif camera == 2:
                    # Right camera, make angle smaller to help return to center
                    angle -= 0.3
                    distance = 0
                else:
                    distance = 0

                img_list, angle_list = translate_image(image, angle,  step=20,
                                                       distance=distance)
                drive_data.extend(img_list)
                steer_data.extend(angle_list)

                img_list, angle_list = translate_image(np.fliplr(image), -angle,
                                                       distance=distance)
                drive_data.extend(img_list)
                steer_data.extend(angle_list)

print('data count: ', np.shape(drive_data))
train_data, val_data, train_labels, val_labels = train_test_split(drive_data,
                                                                  steer_data,
                                                                  test_size=0.15)
print('Finished collecting Data')

def generator(x_data, y_data, batch_size=16):
    # generates the data for run time memory efficiency
    # x_data : features input data
    # y_data : label data for the x_data
    # batch_size : size of each batch of data to be trained
    # return : batch sized, shuffled data for training
    num_samples = len(x_data)
    while 1:  # Loop forever so the generator never terminates
        shuf_img, shuf_angle = shuffle(x_data, y_data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = shuf_img[offset:offset+batch_size]

            batch_images = []
            for data in batch_samples:
                img = common.image_preprocess(data)
                batch_images.append(img)

            # trim image to only see section with road
            X_train = np.array(batch_images)
            y_train = np.array(shuf_angle[offset:offset+batch_size])

            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_data, train_labels, batch_size=32)
validation_generator = generator(val_data, val_labels, batch_size=32)

print('Starting the model')
from keras.layers import Flatten, Dense, Dropout, Input
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# create the model
x_input = Input(shape=(100, 320, 3))
x = BatchNormalization(epsilon=0.001,
                       mode=0,
                       axis=2,
                       momentum=0.99)(x_input)
x = Convolution2D(24, 5, 5,
                  subsample=(2,2),
                  activation='elu',
                  border_mode='valid',
                  dim_ordering='tf',
                  W_constraint=maxnorm(3))(x)
x = BatchNormalization(epsilon=0.001,
                       mode=0,
                       axis=2,
                       momentum=0.99)(x)
x = Dropout(0.4)(x)
x = Convolution2D(36, 5, 5,
                  subsample=(2,2),
                  activation='elu',
                  border_mode='valid',
                  dim_ordering='tf',
                  W_constraint=maxnorm(3))(x)
x = BatchNormalization(epsilon=0.001,
                       mode=0,
                       axis=2,
                       momentum=0.99)(x)
x = BatchNormalization(epsilon=0.001,
                       mode=0,
                       axis=2,
                       momentum=0.99)(x)
x = Dropout(0.4)(x)
x = Convolution2D(48, 5, 5,
                  subsample=(2,2),
                  activation='elu',
                  border_mode='valid',
                  dim_ordering='tf',
                  W_constraint=maxnorm(3))(x)
x = BatchNormalization(epsilon=0.001,
                       mode=0,
                       axis=2,
                       momentum=0.99)(x)
x = Dropout(0.4)(x)
x = Convolution2D(64, 3, 3,
                  activation='elu',
                  border_mode='valid',
                  dim_ordering='tf',
                  W_constraint=maxnorm(3))(x)
x = BatchNormalization(epsilon=0.001,
                       mode=0,
                       axis=2,
                       momentum=0.99)(x)
x = Dropout(0.4)(x)
x = Convolution2D(64, 3, 3,
                  activation='elu',
                  border_mode='valid',
                  dim_ordering='tf',
                  W_constraint=maxnorm(3))(x)
x = BatchNormalization(epsilon=0.001,
                       mode=0,
                       axis=2,
                       momentum=0.99)(x)
x = Dropout(0.4)(x)
x = Convolution2D(64, 3, 3,
                  activation='elu',
                  border_mode='valid',
                  dim_ordering='tf',
                  W_constraint=maxnorm(3))(x)
x = BatchNormalization(epsilon=0.001,
                       mode=0,
                       axis=2,
                       momentum=0.99)(x)

x = (Flatten())(x)
print(np.shape(x))
x = (Dense(100))(x)
x = (ELU(alpha=1.0))(x)
x = (Dropout(0.4))(x)
x = (Dense(50))(x)
x = (ELU(alpha=1.0))(x)
out = (Dense(1))(x)

model = Model(input=x_input, output=out)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_data),
                    validation_data=validation_generator,
                    nb_val_samples=len(val_data),
                    nb_epoch=4)
    
print('Saving model')
model.save('model_hard.h5')
    
print('Finished')
cv2.waitKey(0)
cv2.destroyAllWindows()
gc.collect()