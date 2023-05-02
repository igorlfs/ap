from keras.api._v2 import keras as KerasAPI


def driver(hidden: int, rate: float, grad: str, batch: int | None):
    # Get data
    mnist = KerasAPI.datasets.mnist
    # Split
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Model
    model = KerasAPI.models.Sequential(
        [
            KerasAPI.layers.Flatten(input_shape=(28, 28)),
            KerasAPI.layers.Dense(hidden, activation="relu"),
            KerasAPI.layers.Dense(10),
        ]
    )
    loss = KerasAPI.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = KerasAPI.optimizers.SGD(learning_rate=rate)
    metrics = ["accuracy"]
    # Compile
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # Training loop
    match grad:
        case "gd":
            model.fit(x_train, y_train, batch_size=len(x_train), epochs=100)
        case "sgd":
            # unfortunately this doesn't work
            # model.fit(x_train, y_train, batch_size=1, epochs=3)
            pass
        case "mini":
            model.fit(x_train, y_train, batch_size=batch, epochs=5)
    # Evaluate
    print(model.evaluate(x_test, y_test))
