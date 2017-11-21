import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tabpy_client
from sklearn.neural_network import MLPRegressor

df = pd.read_csv('Boston_Housing.csv')
y = df['MEDV']
X = df.drop(['MEDV'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model2 = MLPRegressor(hidden_layer_sizes=(100,100,64), activation='relu', solver='adam', alpha=0.0001,
                                         batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
                                         power_t=0.5, max_iter=400, shuffle=True, random_state=None,
                                         tol=0.0001, verbose=True, warm_start=False, momentum=0.8,
                                         nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                                         beta_1=0.9, beta_2=0.999, epsilon=1e-08)


model2.fit(X_train, y_train)
print(mean_absolute_error(y_test, model2.predict(X_test)))


def predict_price(_arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8, _arg9, _arg10, _arg11, _arg12):
    import pandas as pd

    d = {'A-CRIM': _arg1, 'B-ZN': _arg2, 'C-INDUS': _arg3, 'D-CHAS': _arg4, 'E-NOX': _arg5, 'F-RM': _arg6,
         'G-AGE': _arg7, 'H-DIS': _arg8, 'I-RAD': _arg9, 'J-TAX': _arg10, 'K-PTRATIO': _arg11, 'L-LSTAT':_arg12}
    # Convert the dictionary to a Pandas Dataframe
    df = pd.DataFrame(data=d)

    df = scaler.transform(df)

    # Use the loaded model to develop predictions for the new data from Tableau
    prices = model2.predict(df)
    return [price for price in prices]


client = tabpy_client.Client('http://localhost:9004')

# Identify and deploy the loan classifier function defined above
client.deploy('predict_price', predict_price,
              'Returns the estimated price of a home based on a variety of factors', override=True)

