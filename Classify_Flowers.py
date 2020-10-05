from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling2D
from keras.layers import Convolution2D
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image
import numpy as np
import os
# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


#load and prepare images from folder
def Prepare_Image(main_folder):
    global X
    global y
    X =[]
    y =[]
    for inner_folder in os.listdir(main_folder):
       path =os.path.join(main_folder,inner_folder)
       for filename in os.listdir(path):
         image_path = os.path.join(path,filename)
         img = image.load_img(image_path)
         if img is not None:
            img = img.resize((64, 64))
            img = image.img_to_array(img)
            img = img/255
            X.append(img)
            y.append(inner_folder)


Prepare_Image("flowers")

#check first image which is pixels
print(X[0])
#check classes
print(np.unique(y))

#Encode categorical classes
label_encoder=LabelEncoder()
Y=label_encoder.fit_transform(y)

#convert every class to array of 5 values like daisy encoded 1 = [1,0,0,0,0]
Y=to_categorical(Y,5)

#Test Set and Train set
# Test set 10 %
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X , Y , test_size = 0.1 , random_state = 42)


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(Dropout(.25))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(1024, activation = 'relu'))
classifier.add(Dense(512, activation = 'relu'))
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(5, activation = 'softmax'))

#%%

# Compiling the CNN
classifier.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

#convert to array
x_train = np.array(x_train)
x_test = np.array(x_test)

#Fit Classifier
history = classifier.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), batch_size=64)

score = classifier.evaluate(x_test, y_test)

#predict single image
labels = ['daisy', 'rose', 'dandelion', 'sunflower', 'tulip']

def prepare(filepath):
    new_image =image.load_img(filepath)
    new_image = new_image.resize((64, 64))
    new_array = image.img_to_array(new_image)
    new_array =new_array.reshape(-1, 64, 64, 3)
    return new_array



prediction = classifier.predict_classes([prepare("growing-sunflowers.jpg")])
print(labels[(prediction[0])])

#result :sunflowers
