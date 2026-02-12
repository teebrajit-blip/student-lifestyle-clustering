from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
scaler = pickle.load(open("model/scaler.pkl", "rb"))
model = pickle.load(open("model/kmeans_model.pkl", "rb"))

cluster_labels = {
    0: "📱 Entertainment Heavy - High screen time, less study",
    1: "📚 High Achiever - Studies a lot, low screen time",
    2: "⚖️ Balanced Lifestyle - Healthy routine",   
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    study = float(request.form['study'])
    sleep = float(request.form['sleep'])
    screen = float(request.form['screen'])

    data = np.array([[study, sleep, screen]])
    data_scaled = scaler.transform(data)

    cluster = model.predict(data_scaled)[0]
    result = cluster_labels[cluster]

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
