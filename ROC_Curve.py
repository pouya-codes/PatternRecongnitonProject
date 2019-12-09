import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import glob
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# load test label
Test_Y = np.load ("Data/Test_Labels.npy")
print(Test_Y.shape)

# load results
results = {}
for fi in sorted(glob.glob("*.npy")) :
    print(fi)
    results[fi.split('.')[0]]= np.load(fi)


# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves




for key , values in results.items() :
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(3):
        # Compute ROC curve and area the curve

        fpr, tpr, thresholds = roc_curve(Test_Y[i].flatten(), values[i].flatten())
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plt.plot(fpr, tpr, lw=1, alpha=0.3,
        #          label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))




    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr,
             label=key + " Model",
             lw=2, alpha=.8)

    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')


plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

plt.xlim([0.00, 1.00])
plt.ylim([0.00, 1.00])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of propsed models'.title())
plt.legend(loc="lower right")
plt.show()