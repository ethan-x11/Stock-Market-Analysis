import os
import yaml
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from Stock import result
with open('./config.yaml', 'r') as fd:
    opts = yaml.safe_load(fd)

app = Flask(__name__)
CORS(app)

if opts['colab-mode']:
    app.debug = True
    app.secret_key = 'development key'  

# base = "index1.html" if (opts['web-version'] == 1) else "index2.html"

@app.get("/") # route
def index_get():
    return "Stock Price Prediction"
    # return render_template("index.html")

@app.post("/predict")
def predict():
    stock_name = request.get_json().get("name")
    duration = request.get_json().get("duration")
    if (int(duration) <= 60 and int(duration) >= 7):
        response = result(stock_name ,int(duration))
        price = response[0]
        graph = response[1]
        resp ={"price": price, "graph": graph}
        print(f"Future Price for {stock_name}: {response}\n")
    else:
        resp = {"price":"Invalid Time-Frame"}
    return jsonify(resp)

if __name__ == "__main__":
    if opts['colab-mode']:
        app.run(debug=True, use_reloader=False, host='0.0.0.0', port='80')
    else:
        app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 80)))