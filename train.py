import os
import time
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
from data import fetch_data
from model import create_model
from params import *
    
def build_model():
    # build the model
    model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS, dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
    return model
    
def train_model(ticker, LOOKUP_STEP):
    model_name = f"{date_now}_{ticker}_steps{LOOKUP_STEP}"
    if BIDIRECTIONAL:
        model_name += "-b"

    # create folders if !exists
    if not os.path.isdir("results"):
        os.mkdir("results")
    if not os.path.isdir("logs"):
        os.mkdir("logs")

    model = build_model()

    # load the data
    data = fetch_data(ticker,LOOKUP_STEP)

    # tensorflow callbacks
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
    LOOKUP_STEP = int(input("Duration: "))
    train_model(ticker, LOOKUP_STEP)
    print("\nTraining completed. Data saved in < results >.")