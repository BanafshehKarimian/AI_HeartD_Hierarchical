##neural networks
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

file = open("/home/banafshbts/Desktop/hosh/76/Untitled Folder/t1")
file.readline()
data = np.loadtxt(file, delimiter=',')
X_train = data[0:810, 0:12]
Y_train = data[0:810, 13]
X_test = data[810:, 0:12]
Y_test = data[810:, 13]

# X_train, X_test, y_train, y_test = train_test_split(data[0:800, 0:12].reshape(data.shape[1:]).tranpose(), data[0:800, 13], test_size=0.33, #random_state=42)

# X = data[0:810, 0:12]
# y = data[0:810, 13]

# Add noisy features
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2],test_size=.5,random_state=random_state)

objects = []
performance = []
pre = []
rec = []
pres = []
nuralnet = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2, 2), random_state=1)
nuralnet.fit(X_train, Y_train)
accuracy_score(nuralnet.predict(X_test), Y_test)  # 0.9662921348314607
performance.append(accuracy_score(nuralnet.predict(X_test), Y_test))
objects.append('neural\n networks')
# pre.append(precision_score(Y_test, nuralnet.predict(X_test), average='macro'))
rec.append(recall_score(Y_test, nuralnet.predict(X_test), average='macro'))
pres.append(precision_score(Y_test, nuralnet.predict(X_test), average='macro'))
# .................................................................................................#
##Regression+linearSVC+SVC
from sklearn import svm
from sklearn.metrics import accuracy_score

svr_clf = svm.SVR()
svr_clf.fit(X_train, Y_train)
accuracy_score(svr_clf.predict(X_test).round(), Y_test)  # 0.4157303370786517
rec.append(recall_score(Y_test, svr_clf.predict(X_test).round(), average='macro'))
pres.append(precision_score(Y_test, svr_clf.predict(X_test).round(), average='macro'))
performance.append(accuracy_score(svr_clf.predict(X_test).round(), Y_test))
objects.append('Regression')
# pre.append(precision_score(Y_test, svr_clf.predict(X_test), average='macro'))
lin_clf = svm.LinearSVC()
lin_clf.fit(X_train, Y_train)
accuracy_score(lin_clf.predict(X_test).round(), Y_test)  # 0.5056179775280899
rec.append(recall_score(Y_test, lin_clf.predict(X_test).round(), average='macro'))
pres.append(precision_score(Y_test, lin_clf.predict(X_test).round(), average='macro'))
performance.append(accuracy_score(lin_clf.predict(X_test).round(), Y_test))
objects.append('linearSVC')
# pre.append(precision_score(Y_test, lin_clf.predict(X_test), average='macro'))
rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.fit(X_train, Y_train)
accuracy_score(Y_test, rbf_svc.predict(X_test).round())  # 0.4157303370786517
rec.append(recall_score(Y_test, rbf_svc.predict(X_test).round(), average='macro'))
pres.append(precision_score(Y_test, rbf_svc.predict(X_test).round(), average='macro'))
performance.append(accuracy_score(Y_test, rbf_svc.predict(X_test).round()))
objects.append('rbfSVC')
# pre.append(precision_score(Y_test, rbf_svc.predict(X_test), average='macro'))
sigmoid_svc = svm.SVC(kernel='sigmoid')
sigmoid_svc.fit(X_train, Y_train)
accuracy_score(Y_test, sigmoid_svc.predict(X_test).round())  # 0.5617977528089888
rec.append(recall_score(Y_test, sigmoid_svc.predict(X_test).round(), average='macro'))
pres.append(precision_score(Y_test, sigmoid_svc.predict(X_test).round(), average='macro'))
performance.append(accuracy_score(Y_test, sigmoid_svc.predict(X_test).round()))
objects.append('sigmoidSVC')
# pre.append(precision_score(Y_test, sigmoid_svc.predict(X_test), average='macro'))
# .................................................................................................#
## Nearest Centroid Classifier
from sklearn.neighbors.nearest_centroid import NearestCentroid

ncc_clf = NearestCentroid()
ncc_clf.fit(X_train, Y_train)
accuracy_score(ncc_clf.predict(X_test), Y_test)  # 0.5842696629213483
rec.append(recall_score(Y_test, ncc_clf.predict(X_test), average='macro'))
pres.append(precision_score(Y_test, ncc_clf.predict(X_test), average='macro'))
performance.append(accuracy_score(ncc_clf.predict(X_test), Y_test))
objects.append('Nearest \nCentroid\n Classifier')
# pre.append(precision_score(Y_test, ncc_clf.predict(X_test), average='macro'))
# .................................................................................................#
##Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
accuracy_score(gnb.fit(X_train, Y_train).predict(X_test), Y_test)  # 0.5955056179775281
rec.append(recall_score(Y_test, gnb.fit(X_train, Y_train).predict(X_test), average='macro'))
pres.append(precision_score(Y_test, gnb.fit(X_train, Y_train).predict(X_test), average='macro'))
performance.append(accuracy_score(gnb.fit(X_train, Y_train).predict(X_test), Y_test))
objects.append('Gaussian\n Naive \nBayes')
# pre.append(precision_score(Y_test, gnb.fit(X_train, Y_train).predict(X_test), average='macro'))
# .................................................................................................#
##DecisionTreeClassifier
from sklearn import tree

DT_clf = tree.DecisionTreeClassifier()
DT_clf = DT_clf.fit(X_train, Y_train)
accuracy_score(DT_clf.predict(X_test), Y_test)  # 0.6292134831460674
rec.append(recall_score(Y_test, DT_clf.predict(X_test), average='macro'))
pres.append(precision_score(Y_test, DT_clf.predict(X_test), average='macro'))
performance.append(accuracy_score(DT_clf.predict(X_test), Y_test))
# pre.append(precision_score(Y_test, DT_clf.predict(X_test), average='macro'))
objects.append('DecisionTree')
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("Heart D-tree")
DTR_clf = tree.DecisionTreeRegressor()
DTR_clf = DTR_clf.fit(X_train, Y_train)
accuracy_score(DTR_clf.predict(X_test), Y_test)  # 0.6292134831460674
rec.append(recall_score(Y_test, DTR_clf.predict(X_test), average='macro'))
pres.append(precision_score(Y_test, DTR_clf.predict(X_test), average='macro'))
performance.append(accuracy_score(DTR_clf.predict(X_test), Y_test))
objects.append('DecisionTree \nregression')
# pre.append(precision_score(Y_test, DTR_clf.predict(X_test), average='macro'))

# .................................................................................................#
##plotting
import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

y_pos = np.arange(len(objects))
plt.bar(y_pos, pres, .1, alpha=0.5, color='navy', label="precision")
plt.bar(y_pos + 0.1, performance, .1, alpha=0.5, color='green', label="score")
plt.bar(y_pos + 0.2, rec, .1, alpha=0.5, color='purple', label="recall")
# plt.yticks(())
plt.xticks(y_pos, objects)
# plt.yticks(())
plt.legend(loc='best')
# plt.subplots_adjust(left=.25)
# plt.subplots_adjust(top=.95)
# plt.subplots_adjust(bottom=.05)

plt.show()
