import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
import time

#load the datasets
train_data = pd.read_csv('X_train.csv')
test_data = pd.read_csv('X_test.csv')


#function to divide the dataset and split it into features and targets
def divideDataset():
    X = train_data.drop(columns=['v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3', 'Id'])
    X = X.iloc[::257].reset_index(drop=True).loc[np.repeat(np.arange(len(X.iloc[::257])), 256)]
    X['t'] = np.repeat(train_data['t'].iloc[::257].values, 256)
    y = train_data.drop(train_data.index[::257]).reset_index(drop=True)
    y = y.drop(columns=['t', 'v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3', 'Id'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = divideDataset()


#function to make and visualize the graphs on screen
def plot_y_yhat(y_val,y_pred, plot_title = "plot"):
    labels = ['x_1','y_1','x_2','y_2','x_3','y_3']
    MAX = 500
    if len(y_val) > MAX:
        idx = np.random.choice(len(y_val),MAX, replace=False)
    else:
        idx = np.arange(len(y_val))
    plt.figure(figsize=(10,10))
    for i in range(6):
        x0 = np.min(y_val[idx,i])
        x1 = np.max(y_val[idx,i])
        plt.subplot(3,2,i+1)
        plt.scatter(y_val[idx,i],y_pred[idx,i])
        plt.xlabel('True '+labels[i])
        plt.ylabel('Predicted '+labels[i])
        plt.plot([x0,x1],[x0,x1],color='red')
        plt.axis('square')
    plt.savefig(plot_title+'.pdf')
    plt.show()


#function with the linear regression model of the problem
def linearRegression(X_train, y_train, X_test, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    result = pipeline.predict(X_test)
    print(root_mean_squared_error(y_test, result))
    plot_y_yhat(y_test.to_numpy(), result)


#linearRegression(X_train, y_train, X_test, y_test)


#function to test and validate the different models and degrees
def val_poly_regression(X_train, y_train, X_val, y_val, regressor=RidgeCV(), degrees=range(1, 15), max_features=None):
    best_rmse = float('inf')
    best_model = None
    best_degree = None
    for degree in degrees:
        start_time = time.time()
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('scaler', StandardScaler()),
            ('model', RidgeCV())
        ])
        pipeline.fit(X_train, y_train)
        result = pipeline.predict(X_val)
        rmse = mean_squared_error(y_val, result)
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline
            best_degree = degree
        print(f"Degree: {degree}, RMSE: {rmse}, Features: {pipeline.named_steps['poly'].n_output_features_}, Time: {time.time() - start_time}s")

    return best_model, best_rmse, best_degree


#function with the polynomial regression model of the problem
def polynomialRegression(X_train, y_train, X_test, y_test):
    sample_size = int(0.01 * len(X_train))
    X_sample_train = X_train.iloc[:sample_size]
    y_sample_train = y_train.iloc[:sample_size]
    X_sample_train, X_sample_val, y_sample_train, y_sample_val = train_test_split(X_sample_train, y_sample_train, test_size=0.2)
    best_model, best_rmse, best_degree = val_poly_regression(X_sample_train, y_sample_train, X_sample_val, y_sample_val, regressor=RidgeCV())
    print(f"Best RMSE: {best_rmse}")

    X_sample_test = X_test.iloc[:sample_size]
    y_sample_test = y_test.iloc[:sample_size]
    pipeline = best_model
    pipeline.fit(X_sample_train, y_sample_train)
    result = best_model.predict(X_sample_test)
    print(f"Test RMSE: {root_mean_squared_error(y_sample_test, result)}")
    plot_y_yhat(y_sample_test.to_numpy(), result)


polynomialRegression(X_train, y_train, X_test, y_test)