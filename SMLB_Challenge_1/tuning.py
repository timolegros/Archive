from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from kerastuner.tuners import BayesianOptimization, Hyperband
from sklearn.preprocessing import MinMaxScaler


def replace_carType(carType):
    try:
        types = ['van', 'regcar', 'sportuv', 'sportcar', 'stwagon', 'truck']
        return types.index(carType)
    except ValueError:
        return carType


def replace_fuelType(fuelType):
    try:
        types = ['cng', 'methanol', 'electric', 'gasoline']
        return types.index(fuelType)
    except ValueError:
        return fuelType


path = 'trainingData.csv'
df = read_csv(path)

for col in df.columns[5:11]:
    df[col] = df[col].apply(replace_carType)
for col in df.columns[11:17]:
    df[col] = df[col].apply(replace_fuelType)

df.drop(['id'], axis=1, inplace=True)
X, y = df.values[:, 1:], df.values[:, 0]

X = X.astype('float32')
y = LabelEncoder().fit_transform(y)


scaler = MinMaxScaler()
arr_scaled = scaler.fit_transform(X)
X = pd.DataFrame(arr_scaled)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

n_features = X_train.shape[1]


def create_model(hp):
    model = Sequential()
    for i in range(hp.Int('layers', 2, 15)):
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=10, max_value=512, step=16),
                        activation=hp.Choice('act_' + str(i), ['relu', 'tanh', 'sigmoid']),
                        kernel_initializer='he_normal', input_shape=(n_features,)))

    model.add(Dense(6, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


tuner = BayesianOptimization(create_model, objective='val_accuracy', max_trials=10000)


tuner.search(X_train, y_train, epochs=40, validation_data=(X_test, y_test))
print(tuner.results_summary)


hyper_parameters = tuner.get_best_hyperparameters()[:5]

for i in hyper_parameters:
    print(i.values, '\n')

# tuner = Hyperband(create_model, objective='val_accuracy', max_epochs=40, hyperband_iterations=2)
# make a prediction
# row = [5.1,3.5,1.4,0.2]
# yhat = model.predict([row])
# print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))

# TODO: run the Bayesian and Hyperband tuner then research KNN and SVM to see if they are feasible

