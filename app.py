from flask import Flask, render_template, request, jsonify
import pandas as pd
import mysql.connector
import pickle

app=Flask(__name__)
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Naty@2911",
        database="traffic_prediction"
    )

with open('traffic_model.pkl','rb') as model_file:
    model=pickle.load(model_file)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    location=request.form.get("location")
    time_of_day=request.form.get("time")
    weather=request.form.get("weather")

    connection=connect_to_db()
    cursor=connection.cursor()
    query="""SELECT vehicles FROM traffic_data 
     WHERE TIME(datetime)= %s AND junction=%s """
    
    cursor.execute(query,(time_of_day,location))
    result=cursor.fetchone()
    cursor.close()
    connection.close()
    if result:
        vehicle_count=result[0]
        return jsonify({"status":"success", "prediction":f"Predicted vehicle count: {vehicle_count}"})
    else:
        try:
            datetime=pd.to_datetime(f"2024-01-01{time_of_day}")
            hour=datetime.hour
            day_of_week=datetime.dayofweek
            month=datetime.month

            input_data=pd.DataFrame([[hour,day_of_week,month,location]],columns=['hour','day_of_week','month','Junction'])
            prediction=model.predict(input_data)
            return jsonify({"status":"success","prediction":f"Predicted vehicle count:{prediction[0]}"})
        except Exception as e:
            return jsonify({"status":"error","message":str(e)})
        

if __name__=="__main__":
    app.run(debug=True)