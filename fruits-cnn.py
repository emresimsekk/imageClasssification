#libraries
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import matplotlib.pyplot as plt
from glob import glob # class number

train_path="fruits-360/Training"
test_path="fruits-360/Test"

img=load_img("fruits-360/Training/Walnut/r2_322_100.jpg")
'''
plt.imshow(img)
plt.axis("off")
plt.show()
'''

x=img_to_array(img)
print(x.shape)

className=glob("fruits-360/Training/*")
numberOfClass=len(className)

#%% CNN MODEL
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())


model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())


model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.2))


model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(numberOfClass)) #output
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

batch_size=64


#%%  Data Generation Train-Test

train_datagen=ImageDataGenerator(rescale=1./255,
                   shear_range=0.3,
                   horizontal_flip=True,
                   zoom_range=0.3)

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
        train_path,
        target_size=x.shape[:2],
        batch_size=batch_size,
        color_mode="rgb",
        class_mode="categorical")

test_generator=train_datagen.flow_from_directory(
        test_path,
        target_size=x.shape[:2],
        batch_size=batch_size,
        color_mode="rgb",
        class_mode="categorical")

hist=model.fit_generator(
        generator=train_generator,
        steps_per_epoch=1600//batch_size,
        epochs=20,
        validation_data=test_generator,
        validation_steps=800//batch_size)

#%% Model Save

model.save_weights('fruits-1.h5')

#%% Model Evaluation

print(hist.history.keys())

plt.plot(hist.history["loss"],label="Train Loss")
plt.plot(hist.history["val_loss"],label="Validation Loss")#h["val_loss"]
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"], label="Train Acc")
plt.plot(hist.history["val_accuracy"],label="Validation Acc")
plt.legend()
plt.show()


#%% Confusion Matrix


'''
#%% Save history
import json
with open("fruits-1.json","w") as f:
    json.dump(hist.history,f)
    
#%% Load History
import codecs
with codes.open("fruits-1.json","r",encoding="utf-8") as f:
    h=json.loads(f.read())
'''