import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression, load_iris, make_moons
import pandas as pd
import time
from classe import MLP

def get_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5  # (0.5-1, 0.5-1)
    X[50:100] = np.random.random((50, 2)) / 2  # (0-0.5, 0-0.5)
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]])  # (0-0.5, 0.5-1)
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]])  # (0.5-1, 0-0.5)
    y = np.array([0] * 100 + [1] * 100)
    return X, y

def makemoons():
    X, y = make_moons(n_samples = 200, noise=0.2, random_state=42)
    print(X.shape)
    print(y.shape)
    return X,y

def iris():
    
    iris = load_iris()
    X = iris.data[:, 2:]
    y = (iris.target == 2).astype(np.int64)
    return X, y

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    return acc, prec, rec


def normaliza(X_train, X_test):

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    return X_train_std, X_test_std

def train_test(X, y):
    kf = KFold(n_splits=5)
    param_grid = {'learning_rate': [0.01, 0.001, 0.0001, 0.05, 0.5], 'epochs': [100,500, 1000]}
    best_accuracy = 0.0
    best_params = {}

    acuracy_total = []
    recall_total = []
    precision_total = []
    tempo_total = []

    for v in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=v)
        X_train, X_test = normaliza(X_train, X_test)
        for lr in param_grid['learning_rate']:
            for epochs in param_grid['epochs']:
                avg_accuracy = 0.0

                for train_index, val_index in kf.split(X_train):
                    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                    model = MLP(input_size=X.shape[1], hidden_size=10, output_size=1, learning_rate=lr, epochs=epochs)
                    model.fit(X_train_fold, y_train_fold)
                    y_pred = model.predict(X_val_fold)
                    fold_accuracy = accuracy_score(y_val_fold, y_pred)
                    avg_accuracy += fold_accuracy

                avg_accuracy /= kf.get_n_splits()

                if avg_accuracy > best_accuracy:
                    best_accuracy = avg_accuracy
                    best_params = {'learning_rate': lr, 'epochs': epochs}

        start_time = time.time()
        model = MLP(input_size=X.shape[1], hidden_size=10, output_size=1, learning_rate=best_params['learning_rate'],
                    epochs=best_params['epochs'])
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

        acc, prec, rec = compute_metrics(y_test, y_pred_test)
        acuracy_total.append(acc)
        precision_total.append(prec)
        recall_total.append(rec)

        end_time = time.time()
        tempo_total.append(end_time - start_time)

    print("Melhores parâmetros:", best_params)
    print("Média do tempo de execução:", np.mean(tempo_total))
    print("Desvio padrão do tempo:", np.std(tempo_total))
    print("Tempo total de execução:", sum(tempo_total))

    df_results = pd.DataFrame({
        "acc": acuracy_total,
        "precision": precision_total,
        "recall": recall_total,
        "tempo": tempo_total
    })
    print(df_results.aggregate(
        {'acc': ['mean', 'std'], 'precision': ['mean', 'std'], 'recall': ['mean', 'std'], 'tempo': ['mean', 'std']}))