from libsvm.svmutil import *
import numpy as np
import cv2
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import matplotlib

train_x = []
num = 0
for num in range(1, 5001):
    im_path = "picture/train/train_" + str(num) + ".png"
    im = cv2.imread(im_path)
    if num < 10:
        print(im_path)
    num += 1
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    temp = im.reshape((len(im) * len(im[0])))
    train_x.append(temp / 255.0)
    
test_x = []
for num in range(1, 2501):
    im_path = "picture/test/test_" + str(num) + ".png"
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    temp = im.reshape((len(im) * len(im[0])))
    test_x.append(temp / 255.0)


train_y = []
for i in range(5):
    for j in range(1000):
        train_y.append(i)

test_y = []
for i in range(5):
    for j in range(500):
        test_y.append(i)

'''cost = []
c = 1
for i in range(5):
    cost.append(c)
    c *= 10

#adjust to different cost 
train = svm_problem(train_y, train_x)
for i in range(len(cost)):
    print("************************************************************")
    param_str = '-s 0 -t 2 -c ' + str(cost[i]) + ' -g 0.0001'
    val_str = param_str + " -v 4"
    print(param_str)
    m = svm_train(train, param_str)
    print(svm_train(train, val_str))
    p_label, p_acc, p_val = svm_predict(test_y, test_x, m)'''

#adjust to different type of kernels
'''for i in range(3):
    kernel_str = '-s 0 -t ' + str(i) + ' -c 100 -g 0.0001'
    param = svm_parameter(kernel_str)
    m = svm_train(train, param)
    svm_train(train, kernel_str + ' -v 4')
    p_label, p_acc, p_val = svm_predict(test_y, test_x, m)
    print("************************************************************")
    support_vectors = m.get_SV()
    print("num of support vector:", len(support_vectors))
    #print(support_vectors[0])'''

#nu-SVM using scikit
def accuracy(pred_test, test_y):
    """
    pred_test: list, the result generates from SVM function
    test_y: list, 
    """
    acc = 0
    for i in range(len(pred_test)):
        if pred_test[i] == test_y[i]:
            acc += 1
    return acc, acc / len(test_y)

#adjust to different type of kernel
'''kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernel_list)):
    nu_SVM = svm.NuSVC(kernel = kernel_list[i])
    scores = cross_val_score(nu_SVM, train_x, train_y, cv = 5)
    print(sum(scores) / 5)
    nu_SVM.fit(train_x, train_y)
    pred_test = nu_SVM.predict(test_x)
    print(kernel_list[i] + " kernel nu_SVM accuracy:", accuracy(pred_test, test_y))

#adjust the value of nu
nu_list = [0.1, 0.3, 0.5, 0.7, 0.9]
for i in range(len(nu_list)):
    nu_SVM = svm.NuSVC(nu = nu_list[i])
    scores = cross_val_score(nu_SVM, train_x, train_y, cv = 5)
    print(sum(scores) / 5)
    nu_SVM.fit(train_x, train_y)
    pred_test = nu_SVM.predict(test_x)
    print("nu = " + str(nu_list[i]) + " nu_SVM accuracy:", accuracy(pred_test, test_y))
'''


grid_C = [1, 1e1, 1e2, 1e3, 1e4]
grid_G = [1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4]
#grid search of best parameters of C and gamma
def grid(train_y, train_x):
    train = svm_problem(train_y, train_x)
    best_acc = 0
    best_param = None
    stat_table = []
    for c_t in grid_C:
        print("********************************C = ", c_t)
        t = []
        for g_t in grid_G:
            print("=============================g = ", g_t)
            kernel_str = '-s 0 -t 2 -c ' + str(c_t) + ' -g ' + str(g_t)
            param = svm_parameter(kernel_str)
            temp_acc = svm_train(train, kernel_str + ' -v 4')
            t.append(temp_acc)
            if temp_acc >= best_acc:
                best_param = {'C': c_t, 'Gamma': g_t, 'kernel': 'rbf'}
        stat_table.append(t)
    print(best_param)
    return param, stat_table

def heatmap(data, row_labels, col_labels, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Gamma Parameter')
    ax.set_ylabel('C Parameter')

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", **textkw):
    
    data = im.get_array()

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color='white')
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


'''fig, ax = plt.subplots()
param, stat_table = grid(train_y, train_x)
im, cbar = heatmap(np.array(stat_table), grid_C, grid_G, ax=ax, cmap='seismic_r')
texts = annotate_heatmap(im, valfmt='{x:.3f}')
plt.title('Hyperparameter Gridsearch')
fig.tight_layout()
plt.show()'''

fig, ax = plt.subplots()
param, stat_table = grid(test_y, test_x)
im, cbar = heatmap(np.array(stat_table), grid_C, grid_G, ax=ax, cmap='seismic_r')
texts = annotate_heatmap(im, valfmt='{x:.3f}')
plt.title('Hyperparameter Gridsearch')
fig.tight_layout()
plt.show()


