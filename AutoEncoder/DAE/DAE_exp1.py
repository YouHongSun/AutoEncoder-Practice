import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Conv2D, Flatten, Reshape, Conv2DTranspose
from keras import backend as k
from keras.datasets import mnist
from keras.models import Model
from keras.utils import plot_model
#import PIL import Image
np.random.seed(1337)
(x_train,_),(x_test,_)=mnist.load_data()
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
x_train=np.expand_dims(x_train,axis=3)
x_test=np.expand_dims(x_test,axis=3)
noise=np.random.normal(loc=0.5,scale=0.5,size=x_train.shape)
gray_train=x_train+noise
noise=np.random.normal(loc=0.5,scale=0.5,size=x_test.shape)
gray_test=x_test+noise
gray_train=np.clip(gray_train,0,1)
gray_test=np.clip(gray_test,0,1)
#
img_size=x_train.shape[1]
batch_size=32
input_shape=(img_size,img_size,1)
kernel_size=3
latent_dim=16
layer_filters=[32,64]
#Encoder
inputs=Input(shape=input_shape,name='encoder_input')
x=inputs
for filters in layer_filters:
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=2,activation='relu'            ,padding='same')(x)
shape=k.int_shape(x)
x=Flatten()(x)
latent=Dense(latent_dim,name='latent_vector')(x)
encoder=Model(inputs,latent,name='encoder')
encoder.summary()
plot_model(encoder,to_file='encoder.png')
#Decoder
latent_input=Input(shape=(latent_dim,),name='decoder_input')
x=Dense(shape[1]*shape[2]*shape[3])(latent_input)
x=Reshape((shape[1],shape[2],shape[3]))(x)
for filters in layer_filters:
    x=Conv2DTranspose(filters=filters,kernel_size=kernel_size,strides=2,activation='relu',padding='same')(x)
output=Conv2DTranspose(filters=1,kernel_size=kernel_size,activation='sigmoid',padding='same',name='decoder_output')(x)
decoder=Model(latent_input,output,name='decoder')
decoder.summary()
plot_model(decoder,to_file='decoder.png')
#Autoencoder
autoencoder=Model(inputs,decoder(encoder(inputs)),name='autoencoder')
autoencoder.compile(loss='mse',optimizer='adam')
autoencoder.summary()
plot_model(autoencoder,to_file='autoencoder.png')
#Fit
autoencoder.fit(gray_train,x_train,validation_data=(gray_test,x_test)
        ,epochs=20,batch_size=batch_size)
x_decoded=autoencoder.predict(gray_test)
imgs=np.zeros([4,2,img_size,img_size,1])
for i in range(4):
    imgs[i,0,:]=x_test[i,:]
    imgs[i,1,:]=x_decoded[i,:]
imgs=imgs.reshape((4*img_size,2*img_size))
plt.figure()
plt.imshow(imgs,cmap='gray')
plt.savefig(fname='Origin_Decode.png')
plt.show()
