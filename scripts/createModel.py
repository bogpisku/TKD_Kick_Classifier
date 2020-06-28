import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

# Data load and split
data = pd.read_csv("../DataSets/50_kicks/linear_acceleration/full_dataset.csv")

# Normalize input
input_to_normalize = data.iloc[:, 4:-1]
# create scaler
scaler = MinMaxScaler()
# fit scaler on data
scaler.fit(input_to_normalize)
# apply transform
normalized = scaler.transform(input_to_normalize)
# Replace input data

normalized_data = data
normalized_data.iloc[:, 4:-1] = normalized


X = normalized_data.iloc[:, 0:-1]
Y = normalized_data.iloc[:, -1]

best_model = None
best_avg = None
winning_times = 0

while True:
    # multiclass Classificator
    model = Sequential()
    model.add(Dense(32, input_shape=(24,), activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    ##Separathe the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # numeric categorization
    num_classes = 4
    Y_train = to_categorical(Y_train, num_classes)
    Y_test = to_categorical(Y_test, num_classes)

    history = model.fit(X_train, Y_train, epochs=65, batch_size=2, verbose=1, validation_split=0.2)
    # Evaluate Model
    test_results = model.evaluate(X_test, Y_test, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

    Y_pred = model.predict(X_test)

    matrix = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))
    matrixNorm = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1), normalize="true")

    matrixAvg = (np.sum(np.diag(matrixNorm)) / num_classes)

    # Insert first model
    if best_avg is None and best_model is None:
        best_avg = matrixAvg
        best_model = model
    # Compare new avg and save new model if is better
    elif matrixAvg > best_avg:
        best_avg = matrixAvg
        best_model = model
        winning_times = 0
    else:
        winning_times += 1
        print("Winning times : ",winning_times)
        if winning_times >= 50:
            # Plot results
            # Plot training & validation accuracy values
            fig = plt.figure(figsize=(12, 9))
            plt.subplot(111)
            plt.plot(history.history['accuracy'], lw=2)
            plt.plot(history.history['val_accuracy'], lw=2)
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid(linestyle="--")
            # plt.show()
            plt.savefig("plots/bestModel.png")

            joblib.dump(scaler, '../models/scaler.pkl')

            model_json = model.to_json()
            with open("../models/model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("models/model.h5")
            print("Saved model to disk")
            break
