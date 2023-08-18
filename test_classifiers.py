import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.naive_bayes import GaussianNB



x_axis_labels = ['Original','Fake']

X_train = genfromtxt('data_df_X_train_deepfake_imagenet_finetuned.csv', delimiter=',')
y_train = pd.read_csv('data_df_y_train_deepfake_imagenet_finetuned.csv', delimiter=',',header=None)
X_test = genfromtxt('data_df_X_test_deepfake_imagenet_finetuned.csv', delimiter=',')
y_test = pd.read_csv('data_df_y_test_deepfake_imagenet_finetuned.csv', delimiter=',',header=None)


pd_y_train_labels = y_train[1]
pd_y_test_labels = y_test[1]

pd_y_train_images = y_train[0]
pd_y_test_images = y_test[0]

y_train_labels = pd_y_train_labels.to_numpy()
y_test_labels = pd_y_test_labels.to_numpy()

y_train_images = pd_y_train_images.to_numpy()
y_test_images = pd_y_test_images.to_numpy()

print(y_test_images[0])

################################################
###########################################################
from sklearn import svm

clf = svm.SVC()
clf.fit(X_train, pd_y_train_labels)
predictions_clf = clf.predict(X_test)


cm = confusion_matrix(pd_y_test_labels, predictions_clf)
print(cm)
f = sns.heatmap(cm,annot=True, fmt='g',xticklabels=x_axis_labels, yticklabels=x_axis_labels, annot_kws={"size": 16})
plt.show()

# Print classification metrics of LDA
print('SVM Accuracy ' + str(accuracy_score(pd_y_test_labels, predictions_clf)))
print('SVM F1 score  ' + str(f1_score(pd_y_test_labels, predictions_clf)))
print('SVM Precision  ' + str(precision_score(pd_y_test_labels, predictions_clf)))
print('SVM Recall  ' + str(recall_score(pd_y_test_labels, predictions_clf)))


print(clf.support_vectors_.shape)


from scipy.spatial import distance

distances = []

for arr in clf.support_vectors_:
    
    dst = distance.euclidean(arr.ravel(), X_test[0].ravel())
    distances.append(dst)

distances = np.array(distances)
closest_vectors = np.argpartition(distances,3)[:3]



for val in closest_vectors:
    map = np.all(X_train == (clf.support_vectors_[val]), axis=1)
    
    sample = np.where(map)[0][0]
    print(sample)

    print(np.array_equal(clf.support_vectors_[val],X_train[sample]))
    print(y_train_images[sample])




#############################################
error_rate = []
for i in range(1,40):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(X_train,y_train_labels)
 pred_i = knn.predict(X_test)
 error_rate.append(np.mean(pred_i != y_test_labels))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed',marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
req_k_value = error_rate.index(min(error_rate))+1
print("Minimum error:-",min(error_rate),"at K =",req_k_value)
plt.show()


knn_final = KNeighborsClassifier(n_neighbors=req_k_value)
knn_final.fit(X_train,y_train_labels)
preds = knn.predict(X_test)

cm = confusion_matrix(pd_y_test_labels, preds)
print(cm)
f = sns.heatmap(cm,annot=True, fmt='g',xticklabels=x_axis_labels, yticklabels=x_axis_labels, annot_kws={"size": 16})
plt.show()

# Print classification metrics of LDA
print('KNN Accuracy ' + str(accuracy_score(pd_y_test_labels, preds)))
print('KNN F1 score  ' + str(f1_score(pd_y_test_labels, preds)))
print('KNN Precision  ' + str(precision_score(pd_y_test_labels, preds)))
print('KNN Recall  ' + str(recall_score(pd_y_test_labels, preds)))








################################################
gnb = GaussianNB()

gnb.fit(X_train,y_train_labels)
preds_gnb = gnb.predict(X_test)

cm = confusion_matrix(pd_y_test_labels, preds_gnb)
print(cm)
f = sns.heatmap(cm,annot=True, fmt='g',xticklabels=x_axis_labels, yticklabels=x_axis_labels, annot_kws={"size": 16})
plt.show()

# Print classification metrics of LDA
print('NB Accuracy ' + str(accuracy_score(pd_y_test_labels, preds_gnb)))
print('NB F1 score  ' + str(f1_score(pd_y_test_labels, preds_gnb)))
print('NB Precision  ' + str(precision_score(pd_y_test_labels, preds_gnb)))
print('NB Recall  ' + str(recall_score(pd_y_test_labels, preds_gnb)))





