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

import warnings

warnings.filterwarnings('ignore')
#n_components_list = [1, 10, 30, 50, 90, 100, 150, 200]
n_components_list = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
n = 1
train_acc = []
test_acc = []
for n_components in n_components_list:
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    if False:
        clf = KNeighborsClassifier(n_neighbors=n, weights='uniform')

        clf.fit(X_train_pca, y_train)

        y_pred = clf.predict(X_test_pca)
        print(10 * "=", "{} Result for Neig={}".format(n_components, n), 10 * "=")
        print("Train Score: ", clf.score(X_train_pca, y_train))
        train_acc.append(clf.score(X_train_pca, y_train))
        # print("Accuracy score:{}".format(metrics.accuracy_score(y_test, y_pred)))
        print("Test Another Score: ", clf.score(X_test_pca, y_test))
        test_acc.append(clf.score(X_test_pca, y_test))
        print()
    if True:
        weigth = ['uniform']
        loo_cv = LeaveOneOut()
        neighbors = [n]
        params = {'n_neighbors': neighbors,
                  'weights': weigth,
                  }
        clf = KNeighborsClassifier()
        # kfold=KFold(n_splits=3, shuffle=True, random_state=0)
        # loo_cv=LeaveOneOut()
        gridSearchCV = GridSearchCV(clf, params, cv=loo_cv)
        gridSearchCV.fit(X_train_pca, y_train)

        y_pred = gridSearchCV.predict(X_test_pca)
        print("Grid search fitted..")
        print(gridSearchCV.best_params_)
        print(gridSearchCV.best_score_)
        print("train score:", gridSearchCV.score(X_train_pca, y_train))
        print("tst score:", gridSearchCV.score(X_test_pca, y_test))
        train_acc.append(gridSearchCV.score(X_train_pca, y_train))
        test_acc.append(gridSearchCV.score(X_test_pca, y_test))
        print("grid search cross validation score:{}".format(gridSearchCV.score(X_test_pca, y_test)))
        print()

##WIthout any PCA
# withCV
if True:
    weigth = ['uniform']
    loo_cv = LeaveOneOut()
    neighbors = [n]
    params = {'n_neighbors': neighbors,
              'weights': weigth,
              }
    clf = KNeighborsClassifier()

    gridSearchCV = GridSearchCV(clf, params, cv=loo_cv)
    gridSearchCV.fit(X_train, y_train)
    print("Grid search fitted..")
    print(gridSearchCV.best_params_)
    print(gridSearchCV.best_score_)
    print("A train score:", gridSearchCV.score(X_train, y_train))
    print("tst score:", gridSearchCV.score(X_test, y_test))
    train_acc.insert(0, gridSearchCV.score(X_train, y_train))
    test_acc.insert(0, gridSearchCV.score(X_test, y_test))
    print("grid search cross validation score:{}".format(gridSearchCV.score(X_test, y_test)))
    n_components_list.insert(0, 0)
    print()
    # without CV
    if False:
        clf = KNeighborsClassifier(n_neighbors=n, weights=w)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print(10 * "=", "{} Result for Neig={}".format(n_components, n), 10 * "=")
        print("Train Score: ", clf.score(X_train, y_train))
        train_acc.insert(0, clf.score(X_train, y_train))
        # print("Accuracy score:{}".format(metrics.accuracy_score(y_test, y_pred)))
        print("Test Another Score: ", clf.score(X_test, y_test))
        test_acc.insert(0, clf.score(X_test, y_test))
        n_components_list.insert(0, 0)
        print()
plt.plot(n_components_list, train_acc, 'r', n_components_list, test_acc, 'b')
plt.legend(["Train Accuracy", "Test Accuracy"])
plt.xlabel('Number of Components')
plt.ylabel('Accuracy')
plt.show()
print("Testing Acc Max---")
print("test-acc: ", test_acc[np.array(test_acc).argmax()], "train-acc: ", train_acc[np.array(test_acc).argmax()],
      ", para n_components = ", n_components_list[np.array(test_acc).argmax()], "e K=", n)
metrics.plot_confusion_matrix(gridSearchCV, X_test, y_test, xticks_rotation='vertical', include_values=False)
plt.show()
print(metrics.classification_report(y_test, y_pred))