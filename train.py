import os
import time
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from data import fetch_data
from model import create_model

BIDIRECTIONAL = False
    
def build_model():
    N_STEPS = 50
    FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
    ### model parameters
    N_LAYERS = 2
    CELL = LSTM
    UNITS = 256
    DROPOUT = 0.4
    LOSS = "huber_loss"
    OPTIMIZER = "adam"
    # build the model
    model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS, dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
    return model
    
def train_model(ticker, LOOKUP_STEP):
    SCALE = True
    scale_str = f"sc-{int(SCALE)}"
    SHUFFLE = True
    shuffle_str = f"sh-{int(SHUFFLE)}"
    SPLIT_BY_DATE = False
    split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
    # test ratio size, 0.2 is 20%
    TEST_SIZE = 0.2
    date_now = time.strftime("%Y-%m-%d")

    ### training parameters
    BATCH_SIZE = 64
    EPOCHS = 1

    model_name = f"{date_now}_{ticker}_steps{LOOKUP_STEP}"
    if BIDIRECTIONAL:
        model_name += "-b"

    # create these folders if they does not exist
    if not os.path.isdir("results"):
        os.mkdir("results")
    if not os.path.isdir("logs"):
        os.mkdir("logs")

    model = build_model()

    # load the data
    data = fetch_data(ticker,LOOKUP_STEP)

    # some tensorflow callbacks
    checkpointer = ModelCheckpoint(os.path.join("results", f"{model_name}.h5"), save_weights_only=True, save_best_only=True, verbose=1)

    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    
    ## training
    history = model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]),
                        callbacks=[checkpointer, tensorboard],
                        verbose=1)

if __name__ == "__main__":
    ticker = input("Stock Code: ")
    LOOKUP_STEP = input("Duration: ")
    train_model(ticker, LOOKUP_STEP)
    print("\nTraining completed. Data saved in < results >.")