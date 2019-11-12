import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense,Input,Conv2D,Flatten,Reshape,Conv2DTranspose
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.datasets import cifar10
from keras.utils import plot_model
from keras import backend as k
import os
def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])
(x_train,_),(x_test,_)=cifar10.load_data()
img_row=x_train.shape[1]
img_col=x_train.shape[2]
channel=x_train.shape[3]
imgs_dir='saved_images'
save_dir=os.path.join(os.getcwd(),imgs_dir)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
#Display the 1st 100 input images (color and gray)
imgs=x_test[:100]
imgs=imgs.reshape((10,10,img_row,img_col,channel))
imgs=np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test color images (Ground Truth)')
plt.imshow(imgs,interpolation='none')
plt.savefig('%s/test_color.png'%imgs_dir)
plt.show()
#RGB 2 Gray
x_train_g=rgb2gray(x_train)
x_test_g=rgb2gray(x_test)
imgs=x_test_g[:100]
imgs=imgs.reshape((10,10,img_row,img_col))
imgs=np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Testn gray images (Input)')
plt.imshow(imgs,interpolation='none',cmap='gray')
plt.savefig('%s/test_gray.png'%imgs_dir)
plt.show()
#Preprocessing
x_train_g=x_train_g.astype('float32')/255
x_test_g=x_test_g.astype('float32')/255
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
x_train_g=np.expand_dims(x_train_g,axis=len(x_train_g.shape))
x_test_g=np.expand_dims(x_test_g,axis=len(x_test_g.shape))
#Hyper-parameters
input_shape=(img_row,img_col,1)
batch_size=32
epochs=40
kernel_size=3
latent_dim=256
layer_filters=[64,128,256]
#Encoder
inputs=Input(shape=input_shape,name='encoder_input')
x=inputs
for filters in layer_filters:
    x=Conv2D(filters=filters,
            kernel_size=kernel_size,
            strides=2,
            activation='relu',
            padding='same')(x)
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
    x=Conv2DTranspose(filters=filters,
            kernel_size=kernel_size,
            strides=2,
            activation='relu',
            padding='same')(x)
output=Conv2DTranspose(filters=channel,kernel_size=kernel_size,activation='sigmoid',padding='same',name='decoder_output')(x)
decoder=Model(latent_input,output,name='decoder')
decoder.summary()
plot_model(decoder,to_file='decoder.png')
#Autoencoder
autoencoder=Model(inputs,decoder(encoder(inputs)),name='autoencoder')
autoencoder.summary()
save_dir=os.path.join(os.getcwd(),'save_models')
model_name='colorized_ae_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath=os.path.join(save_dir,model_name)
lr_reducer=ReduceLROnPlateau(factor=np.sqrt(0.1),
        cooldown=0,patience=5,verbose=1,min_lr=0.5e-6)
checkpoint=ModelCheckpoint(filepath=filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=True)
autoencoder.compile(loss='mse',optimizer='adam')
callbacks=[lr_reducer,checkpoint]
autoencoder.fit(x_train_g,x_train,validation_data=(x_test_g,x_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks)
x_decoded=autoencoder.predict(x_test_g)
imgs=x_decoded[:100]
imgs=imgs.reshape((10,10,img_row,img_col,channel))
imgs=np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Colorized test images (Predicted)')
plt.imshow(imgs,interpolation='none')
plt.savefig('%s/colorized.png'%imgs_dir)
plt.show()
