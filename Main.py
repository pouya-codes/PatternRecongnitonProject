from PIL import Image
import glob
import cv2
import numpy as np
import stack_wrapper
import matplotlib.pyplot as plt
import UNet
import tensorflow as tf
import sklearn
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow import keras
import DataAugmentation
tf.keras.optimizers.SGD
from sklearn import model_selection



np.random.seed(0)

all_files = np.array(glob.glob("AugmentedData/image*"),dtype=np.str)

X = []
y = []
for f in all_files:
    label = str(f).split('image_')[-1]
    X.append(cv2.imread("AugmentedData/image_"+label,cv2.IMREAD_GRAYSCALE))
    th, threshed = cv2.threshold(cv2.imread("AugmentedData/mask_" + label, cv2.IMREAD_UNCHANGED)
                                 , 128, 255, cv2.THRESH_BINARY)
    y.append(threshed)

X = np.expand_dims(np.array(X) / 255.0 ,axis=-1)
y = np.expand_dims(np.array(y) / 255.0 ,axis=-1)


indexs = np.arange(len(all_files))
np.random.shuffle(indexs)
train_idx = indexs[:270]
test_idx = indexs[270:300]

X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]

np.save("test_label.npy",y_test)


# TODO Add comments, test the model with different layers and filters
def plot_history(histories) :
    plt.figure(figsize=(16,10))
    print (len(histories))
    for name, history_lr in histories :
        val_accuracy = [hist.history['val_acc'] for hist in history_lr]
        train_accuracy = [hist.history['acc'] for hist in history_lr]

        maxs = np.max(val_accuracy,axis=0)
        mins = np.min(val_accuracy, axis=0)
        avg = np.average(val_accuracy, axis=0)

        val = plt.plot(history_lr[0].epoch,avg,'--',label=str(name.title()).replace('_',' = ')+' Val')
        plt.fill_between(history_lr[0].epoch, mins, maxs, edgecolor=val[0].get_color(), facecolor=val[0].get_color(), alpha=0.1)


        maxs = np.max(train_accuracy,axis=0)
        mins = np.min(train_accuracy, axis=0)
        avg = np.average(train_accuracy, axis=0)

        plt.plot(history_lr[0].epoch,avg,color=val[0].get_color(),label=str(name.title()).replace('_',' = ')+' Train')
        plt.fill_between(history_lr[0].epoch, mins, maxs, edgecolor=val[0].get_color(), facecolor=val[0].get_color(),alpha=0.1)


    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlim([0,max(history.epoch)])
    plt.show()


filters = [[64, 128, 256, 512, 1024]]
# used learning rates
lrs = [0.1,0.01,0.001,0.0001,0.00001]
lrs = [0.001,0.0001,0.00001]
# keras validation
kf = sklearn.model_selection.KFold(n_splits=5,shuffle=True)
histories = []
for lr in lrs:
    print(lr)

    history_lr = []
    split = 1
    for train_index, val_index in kf.split(X_train):
        file_name = f"lr_{lr}_s_{split}"
        split+=1
        mcp_save = ModelCheckpoint(f'weights/{file_name}.hdf5',
                                   save_best_only=True,
                                   monitor='val_loss',
                                   mode='min')
        csv_logger = CSVLogger(f'logs/{file_name}.log', separator=',', append=False)
        model = UNet.UNet(filters = filters[0],lr = lr)
        history = model.fit(X_train[train_index],y_train[train_index],validation_data=(X_train[val_index],y_train[val_index]),
                            callbacks=[mcp_save,csv_logger],epochs=100,batch_size=1,verbose=2)
        history_lr.append(history)
        result = model.predict(X_test,batch_size=8)
        np.save(file_name + '.npy', result)

    histories.append([f"lr_{lr}",history_lr])




plot_history(histories)







# save results as jpg files
#         result = result > 0.5
#         for i in range(len(Test_X)) :
#             cv2.imwrite("results/UNet/"+file_name+"_{}.jpg".format(i),
#                         cv2.hconcat(((Test_X[i][:,:,0]*255).astype(np.int),
#                                      (result[i][:,:,0]*255).astype(np.int))))


# model.save("UNetWeights_Aug.h5")

# model = tf.keras.models.load_model("UNetWeights.h5")



# load test Data
# Test_X = np.zeros((30,512,512),dtype=np.int16)
# sw_data  = stack_wrapper.Stack_wrapper('Data/test-volume.tif')
# for i in range(30) :
#     Test_X[i] = sw_data.next()
#
# Test_X = Test_X/255.0
# Test_X = np.expand_dims(Test_X,axis=-1)

#         result = result > 0.5
#
#
#
# fig = plt.figure()
# fig.subplots_adjust(hspace=0.4, wspace=0.4)
#
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(np.reshape(Test_X[0]*255, (512, 512)), cmap="gray")
#
# ax = fig.add_subplot(1, 2, 2)
# ax.imshow(np.reshape(result[0]*255, (512, 512)), cmap="gray")
#
# plt.show()
#
