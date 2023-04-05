from hyperopt import fmin, tpe, hp


# Define the model
def build_model(input_size, hidden_size, num_classes, learning_rate):
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(hidden_size, input_shape=(input_size,), activation="relu")
    )
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


# Define the search space for the hyperparameters
space = hp.choice(
    "classifier_type",
    [
        {
            "type": "nn",
            "hidden_size": hp.quniform("hidden_size", 100, 1000, 100),
            "learning_rate": hp.loguniform(
                "learning_rate", np.log(0.0001), np.log(0.1)
            ),
        },
        {
            "type": "svm",
            "C": hp.uniform("C", 0, 10),
            "gamma": hp.uniform("gamma", 0, 10),
        },
    ],
)


# Define the objective function to minimize
def objective(params):
    if params["type"] == "nn":
        model = build_model(
            input_size, params["hidden_size"], num_classes, params["learning_rate"]
        )
    else:
        model = SVM(C=params["C"], gamma=params["gamma"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=0)

    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return {"loss": -accuracy, "status": STATUS_OK}


# Use the Tree of Parzen Estimators (TPE) algorithm to search for the best set of hyperparameters
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)
print(best)
