import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr

from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.ensemble import GradientBoostingRegressor as gbr

from xgboost import XGBClassifier
from xgboost import XGBRegressor

from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

count = 1


def model_randomforest_classifier(X_train, X_test, y_train, y_test):
    model_name = f'model_{count}_randomforest_classifier'

    model = rfc()
    model.fit(X_train, y_train)
    model.independentcols = independentcols
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    # print(classification_report(y_test, y_pred))
    score = accuracy_score(y_test, y_pred)

    print(f'{model_name} accuracy: {score}')
    joblib.dump(model, f'model/{model_name}.joblib')


def model_gradientboosting_classifier(X_train, X_test, y_train, y_test):
    model_name = f'model_{count}_gradientboosting_classifier'

    model = gbc()
    model.fit(X_train, y_train)
    model.independentcols = independentcols
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    # print(classification_report(y_test, y_pred))
    score = accuracy_score(y_test, y_pred)

    print(f'{model_name} accuracy: {score}')
    joblib.dump(model, f'model/{model_name}.joblib')


def model_xgboost_classifier(X_train, X_test, y_train, y_test):
    model_name = f'model_{count}_xgboost_classifier'

    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    # print(classification_report(y_test, y_pred))
    score = accuracy_score(y_test, y_pred)

    print(f'{model_name} accuracy: {score}')
    joblib.dump(model, f'model/{model_name}.joblib')


def model_lightgbm_classifier(X_train, X_test, y_train, y_test):
    model_name = f'model_{count}_lightgbm_classifier'

    model = LGBMClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    # print(classification_report(y_test, y_pred))
    score = accuracy_score(y_test, y_pred)

    print(f'{model_name} accuracy: {score}')
    joblib.dump(model, f'model/{model_name}.joblib')


def model_randomforest_regressor(X_train, X_test, y_train, y_test):
    model_name = f'model_{count}_randomforest_regressor'

    model = rfr()
    model.fit(X_train, y_train)
    model.independentcols = independentcols

    score = model.score(X_test, y_test)

    print(f'{model_name} accuracy: {score}')
    joblib.dump(model, f'model/{model_name}.joblib')


def model_gradientboosting_regressor(X_train, X_test, y_train, y_test):
    model_name = f'model_{count}_gradientboosting_regressor'

    model = gbr()
    model.fit(X_train, y_train)
    model.independentcols = independentcols

    score = model.score(X_test, y_test)

    print(f'{model_name} accuracy: {score}')
    joblib.dump(model, f'model/{model_name}.joblib')


def model_xgboost_regressor(X_train, X_test, y_train, y_test):
    model_name = f'model_{count}_xgboost_regressor'

    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = model.score(X_test, y_test)

    print(f'{model_name} accuracy: {score}')
    joblib.dump(model, f'model/{model_name}.joblib')


def model_lightgbm_regressor(X_train, X_test, y_train, y_test):
    model_name = f'model_{count}_lightgbm_regressor'

    model = LGBMRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = model.score(X_test, y_test)

    print(f'{model_name} accuracy: {score}')
    joblib.dump(model, f'model/{model_name}.joblib')


if __name__ == '__main__':
    mydf = pd.read_csv('model/datasets/BaseDefault01.csv')

    targetcol = 'default'
    y = mydf[targetcol]

    independentcols_list = [
        ['renda', 'idade', 'etnia', 'sexo', 'casapropria',
            'outrasrendas', 'estadocivil', 'escolaridade'],
        ['renda', 'idade', 'sexo', 'casapropria',
            'outrasrendas', 'estadocivil', 'escolaridade']
    ]

    for independentcols in independentcols_list:
        X = mydf[independentcols]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.3, random_state=25)

        model_randomforest_classifier(X_train, X_test, y_train, y_test)
        model_xgboost_classifier(X_train, X_test, y_train, y_test)
        model_lightgbm_classifier(X_train, X_test, y_train, y_test)
        model_gradientboosting_classifier(X_train, X_test, y_train, y_test)
        model_randomforest_regressor(X_train, X_test, y_train, y_test)
        model_xgboost_regressor(X_train, X_test, y_train, y_test)
        model_lightgbm_regressor(X_train, X_test, y_train, y_test)
        model_gradientboosting_regressor(X_train, X_test, y_train, y_test)

        count += 1
