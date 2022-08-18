import os
import numpy as np
import time

import matplotlib.pyplot as plt

from data import fetch_data
from train import train_model, build_model

N_STEPS = 50
# scale feature columns & output price
SCALE = True
BIDIRECTIONAL = False
LOSS = "huber_loss"

def plot_graph(test_df,LOOKUP_STEP):
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()
    
    
def get_final_df(model, data,LOOKUP_STEP):
    # if predicted future price is higher than the current, 
    # future price - current price = buy profit
    buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    # if the predicted future price is lower than the current price,
    # then the current price - true future price
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    y_pred = model.predict(X_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    test_df = data["test_df"]

    test_df[f"adjclose_{LOOKUP_STEP}"] = y_pred
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test
    test_df.sort_index(inplace=True)
    final_df = test_df
    final_df["buy_profit"] = list(map(buy_profit, 
                                    final_df["adjclose"], 
                                    final_df[f"adjclose_{LOOKUP_STEP}"], 
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    final_df["sell_profit"] = list(map(sell_profit, 
                                    final_df["adjclose"], 
                                    final_df[f"adjclose_{LOOKUP_STEP}"], 
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    return final_df

def predict(model, data):
    last_sequence = data["last_sequence"][-N_STEPS:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    prediction = model.predict(last_sequence)
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price

def result(ticker,LOOKUP_STEP):
    date_now = time.strftime("%Y-%m-%d")
    model_name = f"{date_now}_{ticker}_steps{LOOKUP_STEP}"

    path_to_file = f"results/{model_name}.h5"
    if os.path.exists(path_to_file) != True:
        train_model(ticker, LOOKUP_STEP)

    model = build_model()
    model_path = os.path.join("results", model_name) + ".h5"
    model.load_weights(model_path)

    data = fetch_data(ticker,LOOKUP_STEP)

    loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    # calculate the mean absolute error (inverse scaling)
    if SCALE:
        mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
    else:
        mean_absolute_error = mae

    # get the final dataframe for the testing set
    final_df = get_final_df(model, data,LOOKUP_STEP)

    # predict the future price
    future_price = predict(model, data)

    accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)
    #buy & sell profit
    total_buy_profit  = final_df["buy_profit"].sum()
    total_sell_profit = final_df["sell_profit"].sum()
    total_profit = total_buy_profit + total_sell_profit
    profit_per_trade = total_profit / len(final_df)
    fp = (f"{future_price:.2f}$")
    # printing metrics
    # print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
    # print(f"{LOSS} loss:", loss)
    # print("Mean Absolute Error:", mean_absolute_error)
    # print("Accuracy score:", accuracy_score)
    # print("Total buy profit:", total_buy_profit)
    # print("Total sell profit:", total_sell_profit)
    # print("Total profit:", total_profit)
    # print("Profit per trade:", profit_per_trade)
    return fp

if __name__ == "__main__":
    ticker = input("Stock Code:")
    LOOKUP_STEP = int(input("Duration: "))
    result(ticker,LOOKUP_STEP)
    data = fetch_data(ticker,LOOKUP_STEP)
    print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
    print(f"{LOSS} loss:", loss)
    print("Mean Absolute Error:", mean_absolute_error)
    print("Accuracy score:", accuracy_score)
    print("Total buy profit:", total_buy_profit)
    print("Total sell profit:", total_sell_profit)
    print("Total profit:", total_profit)
    print("Profit per trade:", profit_per_trade)
    test_df = data["test_df"]
    plot_graph(test_df,LOOKUP_STEP)