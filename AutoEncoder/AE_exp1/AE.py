import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Flatten, Dense, Input, Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.utils import plot_model
from keras import backend as k

(x_train,_),(x_test,_)=mnist.load_data()
img_size=x_train.shape[1]
x_train=np.reshape(x_train,[-1,img_size,img_size,1])
x_test=np.reshape(x_test,[-1,img_size,img_size,1])
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
input_shape=(img_size,img_size,1)
batch_size=32
kernel_size=3
latent_dim=16
layer_filters=[32,64]
inputs=Input(shape=input_shape,name='encoder_input')
x=inputs
for filters in layer_filters:
    x=Conv2D(filters=filters,kernel_size=kernel_size,activation='relu',
            strides=2, padding='same')(x)
shape=k.int_shape(x)
x=Flatten()(x)
latent=Dense(latent_dim,name='latent_vector')(x)
encoder=Model(inputs,latent,name='encoder')
encoder.summary()
plot_model(encoder,to_file='encoder.png')
#
latent_input=Input(shape=(latent_dim,),name='decoder_input')
x=Dense(shape[1]*shape[2]*shape[3])(latent_input)
x=Reshape((shape[1],shape[2],shape[3]))(x)
for filters in layer_filters:
    x=Conv2DTranspose(filters=filters,kernel_size=kernel_size,activation='relu',
            strides=2,padding='same')(x)
output=Conv2DTranspose(filters=1,kernel_size=kernel_size,activation='sigmoid',
        padding='same',name='decoder_output')(x)
decoder=Model(latent_input,output,name='decoder')
decoder.summary()
plot_model(decoder,to_file='decoder.png')
#
autoencoder=Model(inputs,decoder(encoder(inputs)),name='autoencoder')
autoencoder.summary()
plot_model(autoencoder,to_file='autoencoder.png')
autoencoder.compile(loss='mse',optimizer='adam')
hist=autoencoder.fit(x_train,x_train,validation_data=(x_test,x_test),epochs=1,
        batch_size=batch_size)
x_decoded=autoencoder.predict(x_test)
imgs=np.concatenate([x_test[:8],x_decoded[:8]])
print(imgs.shape)
imgs=imgs.reshape((4,4,img_size,img_size))
imgs=np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.imshow(imgs,interpolation='none',cmap='gray')
plt.savefig('inputs_decoded.png')
plt.show()
