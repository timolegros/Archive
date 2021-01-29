from pandas import read_csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization


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

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
arr_scaled = scaler.fit_transform(X)
X = pd.DataFrame(arr_scaled)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

n_features = X_train.shape[1]


def create_model(firstLayer, secondLayer):
    model = Sequential()
    # model.add(BatchNormalization())
    model.add(Dense(firstLayer, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    model.add(Dense(secondLayer, activation='relu', kernel_initializer='he_normal'))

    model.add(Dense(6, activation='softmax'))

    return model


model = create_model(10, 8)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=0)

loss, acc = model.evaluate(X_test, y_test, verbose=0)

print(f'Test Accuracy: {acc}\tLayers: {(10, 8)}')
