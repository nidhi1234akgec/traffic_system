import pandas as pd
import mysql.connector

connection=mysql.connector.connect(
    host="localhost",
    user="root",
    password="Naty@2911",
    database="traffic_prediction"
)

cursor=connection.cursor()
file_path="C:\\Users\\HP\\Downloads\\traffic_system\\trafficdata.csv"
data=pd.read_csv(file_path)

'''print("data columns:", data.columns.tolist())
print(data.head)'''

for _, row in data.iterrows():
    cursor.execute(""" INSERT INTO traffic_data( datetime, junction,vehicles,data_id)VALUES(%s,%s,%s,%s)""",(row['DateTime'],row['Junction'],row['Vehicles'],row['ID']))
connection.commit()
cursor.close()
connection.close()
print("data imported successfully")