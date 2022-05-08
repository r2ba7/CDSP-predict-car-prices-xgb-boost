import numpy as np
import pandas as pd
#Import Flask modules
from flask import Flask, request, jsonify, render_template

#Import pickle to save our regression model
import pickle 

#Initialize Flask and set the template folder to "template"
app = Flask(__name__)

with open('model/labelencoder.pickle', 'rb') as handle:
    le_dict = pickle.load(handle)
model = pickle.load(open('model/model.pkl','rb'))
SC = pickle.load(open('model/scaler.pkl','rb'))
ohe = pickle.load(open('model/onehotencoder.pkl','rb'))


# def extract_client_from_request(req_data):
#     client = []
#     client.append(req_data['name'])
#     client.append(int(req_data['km_driven']))
#     client.append(req_data['fuel'])
#     client.append(req_data['seller_type'])
#     client.append(req_data['transmission'])
#     client.append(req_data['owner'])
#     client.append(float(req_data['mileage']))
#     client.append(req_data['transmission'])
#     client.append(float(req_data['max_power']))
#     client.append(int(req_data['seats']))
#     client.append(float(req_data['torque_unit']))
#     client.append(int(req_data['torque_rpm']))
#     client.append(int(req_data['cars_age']))
#     return client

@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    #client = [extract_client_from_request(request.json)]
    d = request.form.to_dict()
    keys=['name','km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage','engine',
    'transmission','max_power', 'seats', 'torque_unit', 'torque_rpm', 'cars_age']
    tmp_d = {x : float(d[x]) if x in ['mileage', 'engine', 'cars_age', 'torque_rpm', 'torque_unit', 'seats', 'max_power','km_driven'] else d[x] for x in keys}
    df = pd.DataFrame([tmp_d])
    # Could've created a function here.
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    onehotencoded_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'name']
    num_cols = [col for col in df.columns if df[col].dtype != 'object']
    for col in cat_cols:
        df[col] = le_dict[col].transform(df[col])
    df2 = df[onehotencoded_cols].copy()
    df2 = ohe.transform(df2).toarray()
    df.drop(onehotencoded_cols, axis=1, inplace=True)
    df[num_cols] = SC.transform(df[num_cols])
    df = df.to_numpy()
    new_df = np.concatenate((df, df2), axis=1)
    del df, df2
    prediction = model.predict(new_df)[0]
    output = int(prediction)
    print(output)
    return render_template('index.html', prediction_text='Car Price is: {}'.format(output))

#Run app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="5000")