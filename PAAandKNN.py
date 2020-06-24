import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from numpy.random import RandomState
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn import metrics
import time
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
import sklearn as ola
from sklearn import metrics
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import plot_confusion_matrix

sklearn_version = ola.__version__
print(sklearn_version)
data=np.load("data/olivetti_faces.npy")
target=np.load("data/olivetti_faces_target.npy")

#We reshape images for machine learnig  model
X=data.reshape((data.shape[0],data.shape[1]*data.shape[2]))
print("target shape:",target.shape)
y=target.reshape((target.shape[0],1))
print("X shape:",X.shape)
print("y shape:",y.shape)

'''
The data set contains 10 face images for each subject. Of the face images, 70 percent will be used for training, 
30 percent for testing. Uses stratify feature to have equal number of training and test images for each subject. 
Thus, there will be 7 training images and 3 test images for each subject. You can play with training and test rates.
'''
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, stratify=target, random_state=0)
print("X_train shape:",X_train.shape)
print("y_train shape:{}".format(y_train.shape))
print("y_test shape:{}".format(y_test.shape))

for n_components in [100]:
    pca=PCA(n_components=n_components, whiten=True)
    pca.fit(X_train)
    print(pca)
print(pca)

X_train_pca=pca.transform(X_train)
X_test_pca=pca.transform(X_test)

neighbors1 = [10, 30, 70, 100, 130, 150]
neighbors2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

weigth = ['uniform', 'distance']
loo_cv=LeaveOneOut()
neighbors = neighbors1 + neighbors2
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
params={'n_neighbors':neighbors,
                'weights':weigth,
        'algorithm' : algorithm
                }

neighbors1 = []
neighbors2 = [1]
weigth = ['uniform']
loo_cv=LeaveOneOut()
neighbors = neighbors1 + neighbors2
algorithm = ['auto']
params={'n_neighbors':neighbors,
                'weights':weigth,
        'algorithm' : algorithm
                }

clf=KNeighborsClassifier()
#kfold=KFold(n_splits=3, shuffle=True, random_state=0)
#loo_cv=LeaveOneOut()
gridSearchCV=GridSearchCV(clf, params, cv=loo_cv)
gridSearchCV.fit(X_train_pca, y_train)
print("Grid search fitted..")
print(gridSearchCV.best_params_)
print(gridSearchCV.best_score_)
metrics.plot_confusion_matrix(gridSearchCV, X_test_pca, y_test, xticks_rotation='vertical', normalize='true', include_values=False)
plt.show()
print("train score:", gridSearchCV.score(X_train_pca, y_train))
print("test score:", gridSearchCV.score(X_test_pca, y_test))
print("grid search cross validation score:{}".format(gridSearchCV.score(X_test_pca, y_test)))