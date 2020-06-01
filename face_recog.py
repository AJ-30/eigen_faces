#!/usr/bin/env python
# coding: utf-8

# In[103]:


import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load data
lfw_dataset = fetch_lfw_people(min_faces_per_person=100)

#data before PCA
print(lfw_dataset.images.shape)
_, h, w = lfw_dataset.images.shape
X = lfw_dataset.data
print("Data example : ",X)
y = lfw_dataset.target
print("Target example : ",y)
target_names = lfw_dataset.target_names
print("Target names : ", target_names)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 20)

#Choosing mumber of components
pca2 = PCA(n_components = 1140).fit(X)
data = pca2.transform(X)
per_var_ch = pca2.explained_variance_ratio_
cum_var = np.cumsum(per_var_ch)

print()

plt.figure(1, figsize = (6,4))
plt.clf
plt.plot(cum_var)
plt.grid()
plt.axis('tight')
plt.xlabel('number of eigenvectors')
plt.ylabel('% variance explained')
plt.show()

# Compute a PCA 
n_eigenvectors = 100
pca = PCA(n_components=n_eigenvectors, whiten=True).fit(X_train)


# apply PCA transformation to training data
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#post PCA data
st = '*'
print("\n",st*100,"\n")
print("Sample data vector : ", X_train_pca)
print("Shape of vectors post PCA : ", X_train_pca.shape)

#percentage variance change explained
per_var_ch = pca.explained_variance_ratio_ 
print("\n\nPercentage of variance explained by each of 100 eigenvector : \n",per_var_ch)
per_var_ch = np.cumsum(per_var_ch)
print("\n\nTotal percentage of variance explained using 100 eigenvectors : ",per_var_ch[-1])


# In[78]:


# train a classifier
print("Fitting the classifier to the training set")
clf = KNeighborsClassifier(n_neighbors = 6).fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))


# Visualization
def plot_gallery(images, titles, h, w, rows=3, cols=4):
    plt.figure(figsize=(1.8 * cols, 2.4 * rows))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())

def titles(y_pred, y_test, target_names):
    for i in range(y_pred.shape[0]):
        pred_name = target_names[y_pred[i]].split(' ')[-1]
        true_name = target_names[y_test[i]].split(' ')[-1]
        yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)

prediction_titles = list(titles(y_pred, y_test, target_names))
plot_gallery(X_test, prediction_titles, h, w)


eigenfaces = pca.components_.reshape((n_components, h, w))
eigenface_titles = ["eigenface {0}".format(i) for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()


# In[84]:


# train a neural net classifier
print("Fitting the classifier to the training set")
clf2 = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(X_train_pca, y_train)


y_pred = clf2.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))


# Visualization
def plot_gallery(images, titles, h, w, rows=3, cols=4):
    plt.figure(figsize=(1.8 * cols, 2.4 * rows))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())

def titles(y_pred, y_test, target_names):
    for i in range(y_pred.shape[0]):
        pred_name = target_names[y_pred[i]].split(' ')[-1]
        true_name = target_names[y_test[i]].split(' ')[-1]
        yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)

prediction_titles = list(titles(y_pred, y_test, target_names))
plot_gallery(X_test, prediction_titles, h, w)


eigenfaces = pca.components_.reshape((n_components, h, w))
eigenface_titles = ["eigenface {0}".format(i) for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()


# In[65]:


help(pca)


# In[105]:


eigenfaces.shape

