
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def run_model(model, X, y, train_size, random_state):
    if model == "svm_rbf":
        clf = svm.SVC(kernel='rbf', probability=True, C=3)
    elif model == "svm_poly":
        clf = svm.SVC(kernel='poly', probability=True, C=3)
    elif model == "svm_linear":
        clf = svm.SVC(kernel='linear', probability=True, C=3)
    elif model == "guass_nb":
        clf = GaussianNB()
    elif model == "bernoulli_nb":
        clf = BernoulliNB()
    elif model == "mlp":
        clf = MLPClassifier(max_iter=1000)
    elif model == "rf":
        clf = RandomForestClassifier(random_state=9252001)
    elif model == "dt":
        clf = DecisionTreeClassifier(random_state=9252001)

    ss = StratifiedShuffleSplit(n_splits=5, train_size=train_size, random_state=random_state)
    y_test_total = []
    y_pred_total = []
    for i, (train_index, test_index) in enumerate(ss.split(X, y)):
        X_train = [X[index] for index in train_index]
        X_test = [X[index] for index in test_index]
        y_train = [y[index] for index in train_index]
        y_test = [y[index] for index in test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_test_total = np.concatenate((y_test_total, y_test))
        y_pred_total = np.concatenate((y_pred_total, y_pred))

    conf = metrics.confusion_matrix(y_test_total, y_pred_total)
    score = metrics.accuracy_score(y_test_total, y_pred_total)

    return score, conf
