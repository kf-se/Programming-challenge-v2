import pandas as pd
import numpy as np
import csv
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_validate, train_test_split
import matplotlib.pyplot as plt
import math

def preprocessing(filename, ev=False):
    # replace missing values with nan
    missing_value_types = ["ooh", "?", "nan", ""]
    df = pd.read_csv(filename, na_values=missing_value_types)
    # print datatypes for each column, object = string
    #print(df.dtypes)

    # find rows with missing values
    index_missing = np.where(df.isnull()==True)[0]
    index = np.concatenate([np.unique(index_missing), np.unique(np.where(df['x8'] > 100000)[0]), np.unique(np.where(df['x7'] < -500))])
    # loop through them and remove them
    for i in index:
        df = df.drop([i], axis=0)

    # find unique values in string columns  
    if ev == False: 
        char2y = {u:i for i, u in enumerate(df['y'])}
        char2x5 = {u:i for i, u in enumerate(df['x5'])}
        char2x6 = {u:i for i, u in enumerate(df['x6'])}
        print(char2y)
        print(char2x5)
        print(char2x6)

    # cast categorical columns to pandas categorical and create a one-hot encoding
    df['x5'] = pd.Categorical(df['x5'])
    dfDummiesX5 = pd.get_dummies(df['x5'], prefix='x5')
    df['x6'] = pd.Categorical(df['x6'])
    dfDummiesX6 = pd.get_dummies(df['x6'], prefix='x6')
    # concatenate to original dataframe
    df = pd.concat([df, dfDummiesX5, dfDummiesX6], axis=1)
    # drop x5 and x6
    df = df.drop('x5', axis=1)
    df = df.drop('x6', axis=1)
    return df

def split_data(df, split, from_start, from_end, random_seed):
    # Create random_seed
    np.random.seed(random_seed)
    df.iloc[np.random.permutation(len(df))]
    # Make y labels into numbers 
    df.y = pd.Categorical(df.y)
    # Add to end of dataframe
    df['y_code'] = df.y.cat.codes

    # Split evenly over classes
    # 183 is smallest number of same class
    start = int(183*split)
    i_bob_s = (np.where(df['y'] == "Bob")[0])[0:start]
    i_bob_e = (np.where(df['y'] == "Bob")[0])[start:183]
    i_atsuto_s = (np.where(df['y'] == "Atsuto")[0])[0:start]
    i_atsuto_e = (np.where(df['y'] == "Atsuto")[0])[start:183]
    i_jorg_s = (np.where(df['y'] == "Jörg")[0])[0:start]
    i_jorg_e = (np.where(df['y'] == "Jörg")[0])[start:183]

    # Make pandas train and test dataset to numpy arrays
    x_train_b = df.iloc[i_bob_s, from_start:from_end]
    x_train_a = df.iloc[i_atsuto_s, from_start:from_end]
    x_train_j = df.iloc[i_jorg_s, from_start:from_end]
    x_train_p = x_train_b.append(x_train_a).append(x_train_j)
    x_train = x_train_p.to_numpy()

    y_train_b = df.iloc[i_bob_s, [-1]]
    y_train_a = df.iloc[i_atsuto_s, [-1]]
    y_train_j = df.iloc[i_jorg_s, [-1]]
    y_train_p = y_train_b.append(y_train_a).append(y_train_j)
    y_train = y_train_p.to_numpy().ravel()

    x_test_b = df.iloc[i_bob_e, from_start:from_end]
    x_test_a = df.iloc[i_atsuto_e, from_start:from_end]
    x_test_j = df.iloc[i_jorg_e, from_start:from_end]
    x_test_p = x_test_b.append(x_test_a).append(x_test_j)
    x_test = x_test_p.to_numpy()

    y_test_b = df.iloc[i_bob_e, [-1]]
    y_test_a = df.iloc[i_atsuto_e, [-1]]
    y_test_j = df.iloc[i_jorg_e, [-1]]
    y_test_p = y_test_b.append(y_test_a).append(y_test_j)
    y_test = y_test_p.to_numpy().ravel()

    c=0
    if c==1:
        # shuffle training dataset
        rng_state = np.random.get_state()
        np.random.shuffle(x_train)
        np.random.set_state(rng_state)
        np.random.shuffle(y_train)
        np.random.set_state(rng_state)
        np.random.shuffle(x_test)
        np.random.set_state(rng_state)
        np.random.shuffle(y_test)
    print("Shapes training data, y_test, y_train, x_test, x_train", y_test.shape, y_train.shape, x_test.shape, x_train.shape)
    print("Training data\n", x_train_p)
    return x_train, x_test, y_train, y_test

def evaluation_dataset(df, start):
    x_val = df.iloc[:, start:]
    return x_val

print("=======================Preprocessing training data======================= ")
filename = 'TrainOnMe.csv'
df = preprocessing(filename)
# Make y into numbers
print("y\n", df.y)
df.y = pd.Categorical(df.y)
# Add to end of dataframe
df['y_code'] = df.y.cat.codes
print(df.head)
print("=======================Preprocessing evaluation data======================= ")
df_ev = preprocessing('EvaluateOnMe.csv', ev=True)
print("=======================Creating evaluation data======================= \n")
X_val = evaluation_dataset(df_ev, start=2)
X_val = X_val.iloc[:, :-9]
print("X_val\n", X_val)
print("=======================Splitting training data======================= ")
# Which columns to use, -1, -6, -8
print(df.describe())
# from_end=-10,-8,-1; from_start=2,3
X_train, X_test, y_train, y_test = split_data(df, 0.75, 3, -10, 100)   
#X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,3:-10], df.iloc[:,-1], random_state=0) 
print("=======================Grid Search or Best Parameters======================= ")
# Grid search
def grid_search(classifier, param_dist, X, y):
    gsc = GridSearchCV(estimator=classifier, param_grid=param_dist)
    grid_results = gsc.fit(X,y)
    best_params = grid_results.best_params_
    print(best_params)
    return best_params

def printit(clf, X_train, X_val, y_train):
    scoring = {'abs_error': 'neg_mean_absolute_error',
               'squared_error': 'neg_mean_squared_error'}
    scores = cross_validate(clf, X, y, cv=10, scoring=scoring, return_train_score=True)
    print("MAE :", abs(scores['test_abs_error'].mean()), "| RMSE :", math.sqrt(abs(scores['test_squared_error'].mean())))
    print("Accuracy:", clf.score(X_test, y_test))
    pred_test = clf.predict(X_test)
    pred_val = clf.predict(X_val)
    #print(X_val)
    #print("Predictions on val", pred_val)
    array = np.empty((pred_val.shape[0]), dtype=object)
    for i, item in enumerate(pred_val):
        if item==0:
            array[i] = "Atsuto"
        elif item==1:
            array[i] = "Bob"
        elif item==2:
            array[i] = "Jörg"
        else:
            print("fuck")
    print(pred_val)
    print(array)
    np.savetxt("8865.txt", array, delimiter='\n', fmt="%s")

# Make X and y for K-crossfold validation
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))
# Create classifyer for grid search
rf = RandomForestClassifier()
# Param distribution to optimize
param_dist_rf = {"max_depth": [1, 2, 3, 4, 5, 6, None],
            "max_features": [1, 2, 3, 4, 5, 6, 'sqrt', 'log2'],
            "min_samples_leaf": [1, 2, 3, 4, 10, 20],
            "criterion": ["friedman_mse", "mse", "mae"],
            "n_estimators": [50, 100, 150, 200]}

param_dist_gdb = {"loss": ["deviance"],
            "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
            "min_samples_split": np.linspace(0.1, 0.5, 12),
            "min_samples_leaf": np.linspace(0.1, 0.5, 12),
            "max_depth":[3,5,8],
            "max_features":["log2","sqrt"],
            "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
            "n_estimators": [10, 20, 30, 40, 50]}
gdb = GradientBoostingClassifier()
#bp = grid_search(gdb, param_dist_gdb, X, y)
#best_gdb = GradientBoostingClassifier(criterion=bp['criterion'], learning_rate=bp['learning_rate'], min_samples_split=bp['min_samples_split'],
#                                       min_samples_leaf=bp['min_samples_leaf'], max_depth=bp['max_depth'], max_features=bp['max_features'],
#                                       subsample=bp['subsample'], n_estimators=bp['n_estimators'])
gdb.fit(X_train, y_train)
#best_rf = RandomForestClassifier(max_depth=best_params['max_depth'], max_features=best_params['max_features'], min_samples_leaf=best_params['min_samples_leaf'],criterion=best_params['criterion'], n_estimators=best_params['n_estimators'])
#best_rf = RandomForestClassifier(max_depth=None, max_features=6, min_samples_leaf=2,criterion='entropy', n_estimators=30)
#best_rf.fit(X_train, y_train)
printit(gdb, X_train, X_val, y_train)

#rf = RandomForestClassifier() #, min_samples_leaf=20,  max_features=7
#rf.fit(X_train, y_train)
#pred = rf.predict(X_test)
#acc_rf = rf.score(X_test, y_test)
#printit(rf, X_train, X_val, y_train)


