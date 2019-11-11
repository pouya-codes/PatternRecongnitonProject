from PIL import Image
import cv2
import numpy as np
import stack_wrapper
import matplotlib.pyplot as plt
import UNet
import tensorflow as tf
from tensorflow import keras
tf.keras.optimizers.SGD
# TODO Add comments, test the model with different layers and filters

# load train Data
Train_X = np.zeros((30,512,512),dtype=np.int16)
Train_Y = np.zeros((30,512,512),dtype=np.int16)
sw_data  = stack_wrapper.Stack_wrapper('Data/train-volume.tif')
sw_lable = stack_wrapper.Stack_wrapper('Data/train-labels.tif')

for i in range(30) :
    Train_X[i] = sw_data.next()
    Train_Y[i] = sw_lable.next()

# normalizing data
Train_X = Train_X/255.0
Train_X = np.expand_dims(Train_X,axis=-1)
Train_Y = Train_Y/255.0
Train_Y = np.expand_dims(Train_Y,axis=-1)

def plot_history(histories) :
    plt.figure(figsize=(16,10))

    for name, history in histories :

        val = plt.plot(history.epoch,history.history['val_loss'],
                       '--',label=name.title()+' Val')
        plt.plot(history.epoch,history.history['loss'],
                 color=val[0].get_color(),label=name.title()+' Train')


    plt.xlabel('Epochs')
    plt.ylabel('Binary Crossentropy')
    plt.legend()
    plt.xlim([0,max(history.epoch)])
    plt.show()



# model = UNet.UNet()
# history = model.fit(Train_X,Train_Y,validation_split=0.2,epochs=50,batch_size=1)
# plot_history([('modified U-Net',history)])
# model.save("UNetWeights.h5")
#
model = tf.keras.models.load_model("UNetWeights.h5")


# load test Data
Test_X = np.zeros((30,512,512),dtype=np.int16)
sw_data  = stack_wrapper.Stack_wrapper('Data/test-volume.tif')
for i in range(30) :
    Test_X[i] = sw_data.next()

Test_X = Test_X/255.0
Test_X = np.expand_dims(Test_X,axis=-1)

result = model.predict(Test_X)

# save results as jpg files
for i in range(30) :
    cv2.imwrite("results/UNet/{}.jpg".format(100+i),
                cv2.hconcat(((Test_X[i][:,:,0]*255).astype(np.int),
                             (result[i][:,:,0]*255).astype(np.int))))


result = result > 0.5

np.save('result.npy',result)

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 1)
ax.imshow(np.reshape(Test_X[0]*255, (512, 512)), cmap="gray")

ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(result[0]*255, (512, 512)), cmap="gray")

plt.show()

