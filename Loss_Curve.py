# this file used for ploting loss curves of different learning rates
import numpy as np
import matplotlib.pyplot as plt
import glob
def plot_history(histories) :
    plt.figure(figsize=(16,10))
    for key in histories.keys() :
        train_accuracy = [hist[:,0] for hist in histories[key]]
        val_accuracy = [hist[:,1] for hist in histories[key]]


        std = np.std(val_accuracy,axis=1)[0]
        avg = np.average(val_accuracy, axis=0)
        epochs = np.arange(0, len(avg))
        maxs = avg
        mins = avg
        for idx,std in enumerate(np.std(train_accuracy, axis=1)) :
            maxs += std
            mins -= std
        val = plt.plot(epochs,avg,'--',label=key+' Val')
        # plt.fill_between(epochs, avg-std, avg+std, edgecolor=val[0].get_color(), facecolor=val[0].get_color(), alpha=0.1)

        avg = np.average(train_accuracy, axis=0)
        maxs = avg
        mins = avg
        for idx,std in enumerate(np.std(train_accuracy, axis=1)) :
            maxs += std
            mins -= std
        plt.plot(epochs,avg,color=val[0].get_color(),label=key+' Train')
        # plt.fill_between(epochs, maxs, mins, edgecolor=val[0].get_color(), facecolor=val[0].get_color(),alpha=0.1)

    plt.plot([50, 50], [0, 1.2], linestyle='--', lw=2, color='gray',
              alpha=.8)
    plt.xlabel('Epochs',fontsize=18)
    plt.ylabel('Loss',fontsize=18)
    plt.legend(loc ='upper right',fontsize=14)
    plt.title('Loss value over number of epochs'.title(), fontsize=20)
    plt.xlim([0,100])
    plt.ylim([0,1.2])
    plt.savefig('figure5.png')
    plt.show()

histories = {}
for f in glob.glob("logs/*.log") :
    lr = f.split('\\')[-1].split('_s')[0].replace('_'," = ")
    rows = np.loadtxt(f, delimiter=',', skiprows=1)[:, [2, 4]]
    if lr in histories:
        histories[lr] = histories[lr] + [rows]
    else :
        histories[lr] = [rows]


plot_history(histories)
