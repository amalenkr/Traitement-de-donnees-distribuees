import time
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import multiprocessing as mp
from multiprocessing import Process
from threading import Thread

cpus = mp.cpu_count()

def preprocessing(path):
    data = pd.read_csv(str(path))
    data["homework_target"] = np.where(data.required_car_parking_spaces == 0,0,1)
    data["arrival_date"] = data['arrival_date_day_of_month'].astype(str) + data['arrival_date_month'].astype(str) + data['arrival_date_year'].astype(str)
    data["arrival_date"] = pd.to_datetime(data["arrival_date"], format='%d%B%Y')
    data["reserved_assigned_room_type"] = np.where(data["reserved_room_type"] == data["assigned_room_type"], 1, 0)
    
    Col_to_drop = ["is_canceled", "lead_time", "arrival_date_year", "arrival_date_month", "arrival_date_week_number",
               "arrival_date_day_of_month", "reserved_room_type", "assigned_room_type",
               "required_car_parking_spaces", "reservation_status", "reservation_status_date",
               "agent", "company"]
    data = data.drop(Col_to_drop, axis=1)
    
    num_var = ["stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
           "previous_cancellations", "previous_bookings_not_canceled", "booking_changes",
           "days_in_waiting_list", "adr","total_of_special_requests"]

    cat_var = ["hotel", "meal", "market_segment", "distribution_channel", "is_repeated_guest",
           "country","deposit_type", "customer_type", "reserved_assigned_room_type"]
    
    for col in cat_var:
        data[col] = data[col].astype('category')
        data["homework_target"] = data["homework_target"].astype('category')
    
    X = data.drop("homework_target", axis=1)
    y = data["homework_target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    train = pd.concat([X_train, y_train], axis=1)
    train1 = train.loc[train.homework_target == 1].reset_index(drop=True)
    train0 = train.loc[train.homework_target == 0].reset_index(drop=True)
    
    N = train1.shape[0]
    UnderSampling = train0.sample(n = N, random_state=13)
    BalanceData = pd.concat([train1, UnderSampling], axis=0)
    X_train = BalanceData.drop("homework_target", axis=1)
    y_train = BalanceData["homework_target"]
    
    numeric_transformer = Pipeline(steps=[("imputer",SimpleImputer(strategy="constant", fill_value=0)),("scaler", MaxAbsScaler())])
    categorical_transformer = Pipeline(steps=[
                                    ("imputer", SimpleImputer(strategy="constant", fill_value="NA")),
                                    ("onehot", OneHotEncoder(handle_unknown='ignore'))])
    preproc = ColumnTransformer(transformers=[("num", numeric_transformer, num_var),
                                          ("cat", categorical_transformer, cat_var)])
    
    return preproc, X_train, X_test, y_train, y_test

def classification(name, model):
    global grids, X_train, X_test, y_train, y_test, preproc
    print(name)
    
    pipe = Pipeline(steps=[('preprocessor', preproc), (name, model)])
    clf = GridSearchCV(pipe, grids[name], cv=3)
    clf.fit(X_train, y_train)
    print('Returned hyperparameter: {}'.format(clf.best_params_))
    print('Best classification accuracy in train is: {}'.format(clf.best_score_))
    print('Classification accuracy on test is: {}'.format(clf.score(X_test, y_test)))
    print()

path = "Data/bookings.csv"
    
models = [('knn', KNeighborsClassifier()),
        ("logreg", LogisticRegression(max_iter=10000, random_state=1)),
         ("RF", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)),
         ("svc", LinearSVC(max_iter=5000))]

grids = {"knn" : {'knn__n_neighbors': [1, 2, 3, 4, 5]},
        "logreg" : {'logreg__C': np.logspace(-2, 2, 5, base=2)
                    }, 
         "RF" : {'RF__n_estimators' : np.arange(60,120+1,30),
                 'RF__max_depth': np.arange(7, 13)
                },
         "svc" : {'svc__C': np.logspace(-2, 2, 5, base=2)}
        }
    
preproc, X_train, X_test, y_train, y_test = preprocessing(path)
    
def main():
    
    start = time.time()
    for name, model in models:
        classification(name, model)
    end = time.time()
    print("Series computation: {} secs\n".format(end - start))
    
    start = time.time()
    threads = []
    for name, model in models:
        t = Thread(target=classification, args=(name,model,))
        threads.append(t)
        t.start()
      
    for t in threads: t.join()
    end = time.time()
    print("Multithreading computation: {} secs\n".format(end - start))
    
    
    start = time.time()
    with mp.Pool(cpus) as p:
        p.starmap(classification, models)
        p.close()
        p.join()
    end = time.time()
    print("Multiprocessing computation (with Pool): {} secs\n".format(end - start))
    
    
    start = time.time()
    processes = []
    for name, model in models:
        p = Process(target=classification, args=(name,model,))
        processes.append(p)
        p.start()
      
    for p in processes: p.join()
    end = time.time()
    print("Multiprocessing computation (with Process): {} secs\n".format(end - start))
    
if __name__ == '__main__':
    # Better protect your main function when you use multiprocessing
    main()