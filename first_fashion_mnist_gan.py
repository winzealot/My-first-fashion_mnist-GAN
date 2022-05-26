#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

tf.random.set_seed(1)

print("Num Gpus available:", len(tf.config.list_physical_devices('GPU')))


# In[2]:


from tensorflow.keras.datasets import fashion_mnist

(xtr, ytr), (xtst, ytst) = fashion_mnist.load_data()


# In[3]:


import numpy as np

dataset = np.concatenate([xtr, xtst], axis=0)
dataset = np.expand_dims(dataset, -1).astype('float32')/255


# In[4]:


BATCH_SIZE = 64
BUFFER = 1024

dataset = np.reshape(dataset, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(dataset)
dataset = dataset.shuffle(buffer_size = BUFFER).batch(BATCH_SIZE)


# In[5]:


from tensorflow import keras
from tensorflow.keras import layers

NOISE_DIM = 150
generator = keras.models.Sequential([
    keras.layers.InputLayer(input_shape = (NOISE_DIM,)),
    layers.Dense(7*7*256),
    layers.Reshape(target_shape=(7,7,256)),
    layers.Conv2DTranspose(256,3,activation='LeakyReLU', strides=2, padding='same'),
    layers.Conv2DTranspose(128,3,activation='LeakyReLU', strides=2, padding='same'),
    layers.Conv2DTranspose(1,3,activation='sigmoid', padding='same')
])
generator.summary()


# In[6]:


discriminator = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(28,28,1)),
    layers.Conv2D(256,3, activation='relu', strides=2, padding='same'),
    layers.Conv2D(128,3, activation='relu', strides=2, padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

discriminator.summary()


# In[7]:


#setting learning rates with adam optimizers
#want to maintain nash equilibrium, if either the gen or the disc gets too strong the adversarial game fails
optimizerG = keras.optimizers.Adam(learning_rate=0.00001, beta_1 = 0.5)
optimizerD = keras.optimizers.Adam(learning_rate=0.00003, beta_1 = 0.5)

#binary classifier (real or fake)
lossFn = keras.losses.BinaryCrossentropy(from_logits=True)

#accuracy metric
gAccMetric = tf.keras.metrics.BinaryAccuracy()
dAccMetric = tf.keras.metrics.BinaryAccuracy()


# In[8]:


@tf.function
def trainDStep(data):
    #establish batch size
    batchSize = tf.shape(data)[0]
    #establish noise, the generator's input
    noise = tf.random.normal(shape=(batchSize, NOISE_DIM))
    
    #concatenating the real and fake labels to one full tensor
    y_true = tf.concat([
        tf.ones(batchSize, 1),
        tf.zeros(batchSize, 1)
    ], axis=0)
    
    #creates generated data then inputs it to discrimator
    with tf.GradientTape() as tape:
        fakedata = generator(noise)
        x = tf.concat([data, fakedata], axis=0)
        y_pred = discriminator(x)
        
        discriminatorLoss = lossFn(y_true, y_pred)
    
    #apply backwards path and update the weights
    grads = tape.gradient(discriminatorLoss, discriminator.trainable_weights)
    optimizerD.apply_gradients(zip(grads, discriminator.trainable_weights))
    
    #report accuracy
    dAccMetric.update_state(y_true, y_pred)
    
    #return loss
    return {
        'discriminator_loss': discriminatorLoss,
        'discriminator_accuracy':dAccMetric.result()
    }
    


# In[9]:


def trainGStep(data):
    batchSize = tf.shape(data)[0]
    noise = tf.random.normal(shape=(batchSize, NOISE_DIM))
    
    y_true = tf.ones(batchSize, 1)
    
    with tf.GradientTape() as tape:
        y_pred = discriminator(generator(noise))
        generatorLoss = lossFn(y_true, y_pred)
        
    grads = tape.gradient(generatorLoss, generator.trainable_weights)
    optimizerG.apply_gradients(zip(grads, generator.trainable_weights))
    
    gAccMetric.update_state(y_true, y_pred)
    
    return {
        'generator_loss':generatorLoss,
        'generator_accuracy':gAccMetric.result()
    }


# In[10]:


import matplotlib.pyplot as plt

def plotImages(model):
    
    images = model(np.random.normal(size=(81, NOISE_DIM)))
    
    plt.figure(figsize=(9,9))
    
    for i, image in enumerate(images):
        plt.subplot(9,9,i+1)
        plt.imshow(np.squeeze(image, -1), cmap='Greys')
        plt.axis('off')
    
    plt.show()


# In[11]:


for epoch in range(30):
    
    dLossSum = 0
    gLossSum = 0
    dAccSum = 0
    gAccSum = 0
    count = 0
    
    for batch in dataset:
        
        #remember you can train the discriminator more by making copies of the following code:
        dLoss = trainDStep(batch)
        dLossSum += dLoss['discriminator_loss']
        dAccSum += dLoss['discriminator_accuracy']
        
        #generator training
        gLoss = trainGStep(batch)
        gLossSum += gLoss['generator_loss']
        gAccSum += gLoss['generator_accuracy']
        
        count+=1
    #log the performance
    
    print("Epoch:{}, loss G:{:0.4f}, loss D:{:0.4f}, Acc G: {:0.2f}%, Acc D:{:0.2f}%".format(
        epoch, 
        gLossSum/count, 
        dLossSum/count, 
        100*gAccSum/count, 
        100*dAccSum/count
    ))
    
    if epoch%2 == 0:
        plotImages(generator)
        



