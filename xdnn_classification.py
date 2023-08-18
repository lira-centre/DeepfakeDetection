"""
Please cite:

Angelov, P., & Soares, E. (2020). Towards explainable deep neural networks (xDNN). Neural Networks, 130, 185-194
"""

import math
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from scipy.special import softmax
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
import seaborn as sn

import matplotlib.pyplot as plt

class xDNNClassifier:
    def train(self, Input):

        Images = Input['Images']
        Features = Input['Features']
        Labels = Input['Labels']
        CN = max(Labels)
        Prototypes = self.PrototypesIdentification(Images, Features, Labels, CN)
        Output = {}
        Output['xDNNParms'] = {}
        Output['xDNNParms']['Parameters'] = Prototypes
        MemberLabels = {}
        for i in range(0, CN + 1):
            MemberLabels[i] = Input['Labels'][Input['Labels'] == i]
        Output['xDNNParms']['CurrentNumberofClass'] = CN + 1
        Output['xDNNParms']['OriginalNumberofClass'] = CN + 1
        Output['xDNNParms']['MemberLabels'] = MemberLabels
        return Output

    def predict(self, Input):

        Params = Input['xDNNParms']
        datates = Input['Features']
        Test_Results = self.DecisionMaking(Params, datates)
        EstimatedLabels = Test_Results['EstimatedLabels']
        Scores = Test_Results['Scores']
        Output = {}
        Output['EstLabs'] = EstimatedLabels
        Output['Scores'] = Scores
        Output['ConfMa'] = confusion_matrix(Input['Labels'], Output['EstLabs'])
        Output['ClassAcc'] = np.sum(Output['ConfMa'] * np.identity(len(Output['ConfMa']))) / len(Input['Labels'])
        return Output

    def PrototypesIdentification(self, Image, GlobalFeature, LABEL, CL):
        data = {}
        image = {}
        label = {}
        Prototypes = {}
        for i in range(0, CL + 1):
            seq = np.argwhere(LABEL == i)
            data[i] = GlobalFeature[seq]
            image[i] = {}
            for j in range(0, len(seq)):
                image[i][j] = Image[seq[j][0]]
            label[i] = np.ones((len(seq), 1)) * i
        for i in range(0, CL + 1):
            Prototypes[i] = self.xDNNclassifier(data[i], image[i])

        return Prototypes

    def xDNNclassifier(self, Data, Image):
        L, N, W = np.shape(Data)
        radius = 1 - math.cos(math.pi / 6)
        Data_2 = Data ** 2
        Data_2 = Data_2.reshape(-1, 1024)
        Xnorm = np.sqrt(np.sum(Data_2, axis=1))

        data = Data.reshape(-1, 1024) / (Xnorm.reshape(-1, 1)) * (np.ones((1, W)))
        Centre = data[0,]
        Centre = Centre.reshape(-1, 1024)
        Center_power = np.power(Centre, 2)
        X = np.array([np.sum(Center_power)])
        Support = np.array([1])
        Noc = 1
        GMean = Centre.copy()
        Radius = np.array([radius])
        ND = 1
        VisualPrototype = {}
        VisualPrototype[1] = Image[0]
        Global_X = 1
        for i in range(2, L + 1):
            GMean = (i - 1) / i * GMean + data[i - 1,] / i
            GDelta = Global_X - np.sum(GMean ** 2, axis=1)
            CentreDensity = 1 / (1 + np.sum(((Centre - np.kron(np.ones((Noc, 1)), GMean)) ** 2), axis=1) / GDelta)
            CDmax = max(CentreDensity)
            CDmin = min(CentreDensity)
            DataDensity = 1 / (1 + np.sum((data[i - 1,] - GMean) ** 2) / GDelta)
            if i == 2:
                distance = cdist(data[i - 1,].reshape(1, -1), Centre.reshape(1, -1), 'euclidean')[0]
            else:
                distance = cdist(data[i - 1,].reshape(1, -1), Centre, 'euclidean')[0]
            value, position = distance.min(0), distance.argmin(0)
            if (DataDensity > CDmax or DataDensity < CDmin):
                # if (DataDensity > CDmax or DataDensity < CDmin) or value >2*Radius[position]:
                Centre = np.vstack((Centre, data[i - 1,]))
                Noc = Noc + 1
                VisualPrototype[Noc] = Image[i - 1]
                X = np.vstack((X, ND))
                Support = np.vstack((Support, 1))
                Radius = np.vstack((Radius, radius))
            else:
                Centre[position,] = Centre[position,] * (Support[position] / (Support[position] + 1)) + data[i - 1] / (
                            Support[position] + 1)
                Support[position] = Support[position] + 1
                Radius[position] = 0.5 * Radius[position] + 0.5 * (X[position,] - sum(Centre[position,] ** 2)) / 2
        dic = {}
        dic['Noc'] = Noc
        dic['Centre'] = Centre
        dic['Support'] = Support
        dic['Radius'] = Radius
        dic['GMean'] = GMean
        dic['Prototype'] = VisualPrototype
        dic['L'] = L
        dic['X'] = X
        return dic

    def DecisionMaking(self, Params, datates):
        PARAM = Params['Parameters']
        CurrentNC = Params['CurrentNumberofClass']
        LAB = Params['MemberLabels']
        VV = 1
        LTes = np.shape(datates)[0]
        EstimatedLabels = np.zeros((LTes))
        Scores = np.zeros((LTes, CurrentNC))
        for i in range(1, LTes + 1):
            data = datates[i - 1,]
            Data_2 = data ** 2
            Data_2 = Data_2.reshape(-1, 1024)
            Xnorm = np.sqrt(np.sum(Data_2, axis=1))
            data = data / Xnorm
            R = np.zeros((VV, CurrentNC))
            Value = np.zeros((CurrentNC, 1))
            for k in range(0, CurrentNC):
                #distance=np.sort(cdist(data.reshape(1, -1),PARAM[k]['Centre'],'minkowski'))[0]
                distance = np.sort(cdist(data.reshape(1, -1), PARAM[k]['Centre'], 'euclidean'))[0]
                Value[k] = distance[0]
            # Value = softmax(-1*Value**2).T
            Value = np.exp(-1 * Value ** 2).T
            Scores[i - 1,] = Value
            Value = Value[0]
            Value_new = np.sort(Value)[::-1]
            indx = np.argsort(Value)[::-1]
            EstimatedLabels[i - 1] = indx[0]
        LABEL1 = np.zeros((CurrentNC, 1))

        for i in range(0, CurrentNC):
            LABEL1[i] = np.unique(LAB[i])

        EstimatedLabels = EstimatedLabels.astype(int)
        EstimatedLabels = LABEL1[EstimatedLabels]
        dic = {}
        dic['EstimatedLabels'] = EstimatedLabels
        dic['Scores'] = Scores

        return dic

    def save_model(self, model, name='xDNN_model'):
        with open(name, 'wb') as file:
            pickle.dump(model, file)

    def load_model(self, name='xDNN_model'):
        with open(name, 'rb') as file:
            pickle_model = pickle.load(file)
        return pickle_model

    def results(self, predicted, y_test_labels):

        accuracy = accuracy_score(y_test_labels, predicted)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(y_test_labels, predicted, average='weighted')
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(y_test_labels, predicted, average='weighted')
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y_test_labels, predicted, average='weighted')
        print('F1 score: %f' % f1)
        # kappa
        kappa = cohen_kappa_score(y_test_labels, predicted)
        print('Cohens kappa: %f' % kappa)
        # confusion matrix
        matrix = confusion_matrix(y_test_labels, predicted)
        print("Confusion Matrix: ", matrix)

        x_axis_labels = ['Original', 'Fake'] # labels for x-axis
        df_cm = pd.DataFrame(matrix, range(2), range(2))
        # plt.figure(figsize=(10,7))
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(df_cm, annot=True, fmt='g',xticklabels=x_axis_labels, yticklabels=x_axis_labels, annot_kws={"size": 16}) # font size

        plt.show()
