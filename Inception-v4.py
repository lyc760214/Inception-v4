#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

def Stem(input):
    conv1 = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3,3),
    strides=2)(input)
    
    conv2 = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3,3))(conv1)
    
    conv3 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=(3,3),
    padding='same')(conv2)
    
    pool1 = tf.keras.layers.MaxPool2D(
    pool_size=3,
    strides=2)(conv3)
    
    conv4 = tf.keras.layers.Conv2D(
    filters=96,
    kernel_size=(3,3),
    strides=2)(conv3)

    concate1 = tf.keras.layers.concatenate([pool1, conv4])
    
    conv5 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=(1,1),
    padding='same')(concate1)
    
    conv6 = tf.keras.layers.Conv2D(
    filters=96,
    kernel_size=(3,3))(conv5)
    
    conv7 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=(1,1),
    padding='same')(concate1)
    
    conv8 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=(7,1),
    padding='same')(conv7)
    
    conv9 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=(1,7),
    padding='same')(conv8)
    
    conv10 = tf.keras.layers.Conv2D(
    filters=96,
    kernel_size=(3,3))(conv9)
    
    concate2 = tf.keras.layers.concatenate([conv6, conv10])
    
    conv11 = tf.keras.layers.Conv2D(
    filters=192,
    kernel_size=(3,3),
    strides=2)(concate2)
    
    pool2 = tf.keras.layers.MaxPool2D(
    pool_size=2,
    strides=2)(concate2)
    
    concate3 = tf.keras.layers.concatenate([conv11, pool2])
    
    return concate3


# In[2]:


def Inception_A(input):
    pool1 = tf.keras.layers.AvgPool2D(
    pool_size=(2,2),
    strides=1,
    padding='same')(input)
    
    conv1 = tf.keras.layers.Conv2D(
    filters=96,
    kernel_size=(1,1),
    padding='same')(pool1)
    
    conv2 = tf.keras.layers.Conv2D(
    filters=96,
    kernel_size=(1,1),
    padding='same')(input)
    
    conv3 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=(1,1),
    padding='same')(input)
    
    conv4 = tf.keras.layers.Conv2D(
    filters=96,
    kernel_size=(3,3),
    padding='same')(conv3)
    
    conv5 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=(1,1),
    padding='same')(input)
    
    conv6 = tf.keras.layers.Conv2D(
    filters=96,
    kernel_size=(3,3),
    padding='same')(conv5)

    conv7 = tf.keras.layers.Conv2D(
    filters=96,
    kernel_size=(3,3),
    padding='same')(conv6)
    
    concate = tf.keras.layers.concatenate([conv1, conv2, conv4, conv7])
    
    output = tf.keras.layers.Activation(activation='relu')(concate)
    
    return output


# In[3]:


def Inception_B(input):
    pool1 = tf.keras.layers.AvgPool2D(
    pool_size=(2,2),
    strides=(1,1),
    padding='same')(input)
    
    conv1 = tf.keras.layers.Conv2D(
    filters=128,
    kernel_size=(1,1),
    padding='same')(pool1)
    
    conv2 = tf.keras.layers.Conv2D(
    filters=384,
    kernel_size=(1,1),
    padding='same')(input)
    
    conv3 = tf.keras.layers.Conv2D(
    filters=192,
    kernel_size=(1,1),
    padding='same')(input)
    
    conv4 = tf.keras.layers.Conv2D(
    filters=224,
    kernel_size=(7,1),
    padding='same')(conv3)
    
    conv5 = tf.keras.layers.Conv2D(
    filters=256,
    kernel_size=(1,7),
    padding='same')(conv4)
    
    conv6 = tf.keras.layers.Conv2D(
    filters=192,
    kernel_size=(1,1),
    padding='same')(input)
    
    conv7 = tf.keras.layers.Conv2D(
    filters=192,
    kernel_size=(1,7),
    padding='same')(conv6)
    
    conv8 = tf.keras.layers.Conv2D(
    filters=224,
    kernel_size=(7,1),
    padding='same')(conv7)
    
    conv9 = tf.keras.layers.Conv2D(
    filters=224,
    kernel_size=(1,7),
    padding='same')(conv8)
    
    conv10 = tf.keras.layers.Conv2D(
    filters=256,
    kernel_size=(7,1),
    padding='same')(conv9)
    
    concate = tf.keras.layers.concatenate([conv1, conv2, conv5, conv10])
    
    output = tf.keras.layers.Activation(activation='relu')(concate)
    
    return output


# In[4]:


def Inception_C(input):
    pool1 = tf.keras.layers.AvgPool2D(
    pool_size=(2,2),
    strides=(1,1),
    padding='same')(input)
    
    conv1 = tf.keras.layers.Conv2D(
    filters=256,
    kernel_size=(1,1),
    padding='same')(pool1)
    
    conv2 = tf.keras.layers.Conv2D(
    filters=256,
    kernel_size=(1,1),
    padding='same')(input)
    
    conv3 = tf.keras.layers.Conv2D(
    filters=384,
    kernel_size=(1,1),
    padding='same')(input)
    
    conv4 = tf.keras.layers.Conv2D(
    filters=256,
    kernel_size=(1,3),
    padding='same')(conv3)
    
    conv5 = tf.keras.layers.Conv2D(
    filters=256,
    kernel_size=(3,1),
    padding='same')(conv3)
    
    conv6 = tf.keras.layers.Conv2D(
    filters=384,
    kernel_size=(1,1),
    padding='same')(input)
    
    conv7 = tf.keras.layers.Conv2D(
    filters=448,
    kernel_size=(1,3),
    padding='same')(conv6)
    
    conv8 = tf.keras.layers.Conv2D(
    filters=512,
    kernel_size=(3,1),
    padding='same')(conv7)
    
    conv9 = tf.keras.layers.Conv2D(
    filters=256,
    kernel_size=(3,1),
    padding='same')(conv8)
    
    conv10 = tf.keras.layers.Conv2D(
    filters=256,
    kernel_size=(1,3),
    padding='same')(conv8)
    
    concate = tf.keras.layers.concatenate([conv1, conv2, conv4, conv5, conv9, conv10])
    
    output = tf.keras.layers.Activation(activation='relu')(concate)
    
    return output


# In[5]:


def Reduction_A(input):
    pool1 = tf.keras.layers.MaxPool2D(
    pool_size=(3,3),
    strides=(2,2))(input)
    
    conv1 = tf.keras.layers.Conv2D(
    filters=384,
    kernel_size=(3,3),
    strides=(2,2))(input)
    
    conv2 = tf.keras.layers.Conv2D(
    filters=192,
    kernel_size=(1,1),
    padding='same')(input)
    
    conv3 = tf.keras.layers.Conv2D(
    filters=224,
    kernel_size=(3,3),
    padding='same')(conv2)
    
    conv4 = tf.keras.layers.Conv2D(
    filters=256,
    kernel_size=(3,3),
    strides=(2,2))(conv3)
    
    concate = tf.keras.layers.concatenate([pool1, conv1, conv4])
    
    return concate


# In[6]:


def Reduction_B(input):
    pool1 = tf.keras.layers.MaxPool2D(
    pool_size=(3,3),
    strides = (2,2))(input)
    
    conv1 = tf.keras.layers.Conv2D(
    filters = 192, 
    kernel_size=(1,1),
    padding='same')(input)
    
    conv2 = tf.keras.layers.Conv2D(
    filters=192,
    kernel_size=(3,3),
    strides=(2,2))(conv1)
    
    conv3 = tf.keras.layers.Conv2D(
    filters=256,
    kernel_size=(1,1),
    padding='same')(input)
    
    conv4 = tf.keras.layers.Conv2D(
    filters=256,
    kernel_size=(1,7),
    padding='same')(conv3)
    
    conv5 = tf.keras.layers.Conv2D(
    filters=320,
    kernel_size=(7,1),
    padding='same')(conv4)
    
    conv6 = tf.keras.layers.Conv2D(
    filters=320,
    kernel_size=(3,3),
    strides=(2,2))(conv5)
    
    concate = tf.keras.layers.concatenate([pool1, conv2, conv6])
    
    return concate


# In[7]:


main_input = tf.keras.layers.Input(
    shape=(32,32,3), 
    dtype='float32', 
    name='main_input')


stacking = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3,3), activation='relu')(main_input)
stacking = Inception_A(stacking)
stacking = Reduction_A(stacking)
stacking = Inception_B(stacking)
stacking = Inception_B(stacking)
stacking = Inception_B(stacking)
stacking = Reduction_B(stacking)
stacking = Inception_C(stacking)

stacking = tf.keras.layers.AvgPool2D()(stacking)
stacking = tf.keras.layers.Flatten()(stacking)
stacking = tf.keras.layers.Dropout(rate=0.5)(stacking)
stacking = tf.keras.layers.Dense(units=10, activation='softmax')(stacking)

model = tf.keras.models.Model(inputs=[main_input], outputs=[stacking])
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.summary()


# In[8]:


tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)


# In[9]:


from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train
X_test = X_test


# In[ ]:


model.fit(X_train, y_train, epochs=128)

