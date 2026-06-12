from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open("loan_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:

        features = np.array([[
            float(request.form["Gender"]),
            float(request.form["Married"]),
            float(request.form["Dependents"]),
            float(request.form["Education"]),
            float(request.form["Self_Employed"]),
            float(request.form["ApplicantIncome"]),
            float(request.form["CoapplicantIncome"]),
            float(request.form["LoanAmount"]),
            float(request.form["Loan_Amount_Term"]),
            float(request.form["Credit_History"]),
            float(request.form["Property_Area"])
        ]])

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "✅ Loan Approved"
            message = "Congratulations! The loan is likely to be approved."
            color = "approved"
        else:
            result = "❌ Loan Rejected"
            message = "The loan may not be approved."
            color = "rejected"

        return render_template(
            "index.html",
            prediction_text=result,
            prediction_message=message,
            color=color
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction_text="⚠️ Error",
            prediction_message=str(e),
            color="rejected"
        )

if __name__ == "__main__":
    app.run(debug=True)