import csv
import numpy as np
from scipy import ndimage

lines = []

with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)  
        
images = []
measurements = []
for line in lines:
    # CENTER CAM
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename    
    image = ndimage.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

    correction = 0.2

    # LEFT CAM
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    img_left = ndimage.imread(current_path)
    images.append(img_left)
    measurements.append(measurement + correction)

    # RIGHT CAM
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    img_left = ndimage.imread(current_path)
    images.append(img_left)
    measurements.append(measurement - correction)

    
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(np.fliplr(image))
    augmented_measurements.append(measurement*-1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))) 
model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')  
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)

# for layer in model.layers:
#     print(layer.output_shape) 
    
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
model.save('model.h5')
          

    
import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
