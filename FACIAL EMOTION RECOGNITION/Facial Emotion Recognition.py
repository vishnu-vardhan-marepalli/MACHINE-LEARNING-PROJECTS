#!/usr/bin/env python
# coding: utf-8

# In[1]:


from zipfile import ZipFile
file_name = "archive (5).zip"
with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print("Done")


# In[2]:


import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D
from keras.optimizers.legacy import Adam
from keras.layers import MaxPooling2D


# In[3]:


train_dir = 'train'
test_dir = 'test'
train_data = ImageDataGenerator(rescale = 1./255)
test_data = ImageDataGenerator(rescale = 1./255)
train_gen = train_data.flow_from_directory(
    train_dir, target_size = (48,48), batch_size = 64, color_mode = "grayscale", class_mode = "categorical"
)
test_gen = test_data.flow_from_directory(
    test_dir, target_size = (48,48), batch_size = 64, color_mode = "grayscale", class_mode = "categorical"
)


# In[4]:


class_labels = ['angry','disgust','fear','happy','nuetral','sad','surprise']
img, label = train_gen.next()
import random
from matplotlib import pyplot as plt
j = random.randint(0,(img.shape[0])-1)
image = img[j]
labl = class_labels[label[j].argmax()]
plt.imshow(image[:,:,0],cmap = 'gray')
plt.title(labl)
plt.show()


# # MODEL BUILDING

# In[5]:


model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape = (48,48,1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7,activation='softmax'))


# In[6]:


model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
model_info = model.fit_generator(
    train_gen,
    steps_per_epoch = 28709//64,
    epochs = 50,
    validation_data = test_gen,
    validation_steps = 7178 // 64

)


# In[7]:


model.save('model.h5')


# In[ ]:





# In[ ]:





# # TESTING THE MODEL

# In[8]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing.image import img_to_array


# In[9]:


model=tf.keras.models.load_model(r"C:/Users/VISHNU VARDHAN/model.h5",compile=False)


# In[10]:


img=image.load_img(r"C:\Users\VISHNU VARDHAN\Downloads\angry.jpg",target_size=(48,48),grayscale=True)


# In[11]:


img


# In[12]:


x=image.img_to_array(img)
x


# In[ ]:





# In[13]:


x=np.expand_dims(x,axis=0)


# In[14]:


x.ndim


# In[15]:


x.shape


# In[16]:


pred=model.predict(x)


# In[17]:


pred


# In[18]:


{'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'nuetral':4, 'sad':5, 'surprise':6}


# In[19]:


pred_class=np.argmax(pred,axis=1)
pred_class[0]


# In[20]:


index = ['angry','disgust','fear','happy','nuetral','sad','surprise']
result=str(index[pred_class[0]])


# In[21]:


result


# In[ ]:





# In[ ]:




