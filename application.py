import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# Load the model
Model = pk.load(open('model.pkl', 'rb'))

st.header('Car Prediction ML Model')


car = pd.read_csv('Cardetails.csv')



def get_brandname(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

car['name'] = car['name'].apply(get_brandname)



name = st.selectbox('Select the brand you want', car['name'].unique())
year = st.slider('Car Manufactured Year',1994,2020)
km_driven = st.slider('Distance driven',1.000000e+00,2.360457e+06)
fuel= st.selectbox('Fuel type', car['fuel'].unique())
seller_type = st.selectbox('Type of the seller', car['seller_type'].unique())

transmission = st.selectbox('Transmission type', car['transmission'].unique())
owner = st.selectbox('Type of owner', car['owner'].unique())

mileage = st.slider('car mileage', 10, 40, step=1)
engine = st.slider('Engine', 624, 3604, step=1)
max_power = st.slider('Max Power', 0, 400, step=1)
seats = st.slider('No. of seats', 2, 14, step=1)

if st.button("Predict"):
    
    input_datamodel = pd.DataFrame([[name, year, km_driven, fuel, seller_type,  transmission, owner,mileage, engine, max_power, seats]],
                                   columns=[ 'name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])

    
    input_datamodel['name'].replace(
        ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
         'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
         'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], inplace=True)

    
    input_datamodel['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_datamodel['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_datamodel['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_datamodel['owner'].replace(
        ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
        [1, 2, 3, 4, 5], inplace=True)

    
    
    st.write(input_datamodel)


    


    
    car_price = Model.predict(input_datamodel)
    st.markdown('Car price can be: ' + str(car_price[0]))
# import pandas as pd
# import numpy as np
# import pickle as pk
# import streamlit as st

# # Load the model
# Model = pk.load(open('model.pkl', 'rb'))

# st.header('Car Prediction ML Model')

# car = pd.read_csv('Cardetails.csv')

# def get_brandname(car_name):
#     car_name = car_name.split(' ')[0]
#     return car_name.strip()

# car['name'] = car['name'].apply(get_brandname)

# name = st.selectbox('Select the brand you want', car['name'].unique())
# year = st.slider('Car Manufactured Year', 1994, 2019)
# km_driven = st.slider('Distance driven', 11, 2000000)
# fuel = st.selectbox('Fuel type', car['fuel'].unique())
# seller_type = st.selectbox('Type of the seller', car['seller_type'].unique())
# transmission = st.selectbox('Transmission type', car['transmission'].unique())
# owner = st.selectbox('Type of owner', car['owner'].unique())
# mileage = st.slider('Car mileage', 10, 40, step=1)
# engine = st.slider('Engine', 700, 5000, step=1)
# max_power = st.slider('Max Power', 0, 200, step=1)
# seats = st.slider('No. of seats', 5, 10, step=1)

# if st.button("Predict"):
#     # Manually encode categorical features
#     name_mapping = {'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5, 'Ford': 6, 'Renault': 7, 'Mahindra': 8, 'Tata': 9,
#                     'Chevrolet': 10, 'Datsun': 11, 'Jeep': 12, 'Mercedes-Benz': 13, 'Mitsubishi': 14, 'Audi': 15, 'Volkswagen': 16,
#                     'BMW': 17, 'Nissan': 18, 'Lexus': 19, 'Jaguar': 20, 'Land': 21, 'MG': 22, 'Volvo': 23, 'Daewoo': 24, 'Kia': 25,
#                     'Fiat': 26, 'Force': 27, 'Ambassador': 28, 'Ashok': 29, 'Isuzu': 30, 'Opel': 31}
#     transmission_mapping = {'Manual': 1, 'Automatic': 2}
#     seller_type_mapping = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}
#     fuel_mapping = {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4}
#     owner_mapping = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5}
    
#     name_encoded = name_mapping[name]
#     transmission_encoded = transmission_mapping[transmission]
#     seller_type_encoded = seller_type_mapping[seller_type]
#     fuel_encoded = fuel_mapping[fuel]
#     owner_encoded = owner_mapping[owner]
    
#     # Scale numerical features (assuming MinMaxScaler was used during training)
#     year_scaled = (year - 1994) / (2019 - 1994)
#     km_driven_scaled = (km_driven - 11) / (2000000 - 11)
#     mileage_scaled = (mileage - 10) / (40 - 10)
#     engine_scaled = (engine - 700) / (5000 - 700)
#     max_power_scaled = max_power / 200
#     seats_scaled = (seats - 5) / (10 - 5)
    
#     # Create input data for prediction
#     input_data = [[name_encoded, year_scaled, km_driven_scaled, fuel_encoded, seller_type_encoded, transmission_encoded,
#                    owner_encoded, mileage_scaled, engine_scaled, max_power_scaled, seats_scaled]]
    
#     input_datamodel = pd.DataFrame(input_data, columns=['name', 'year', 'km_driven', 'fuel', 'seller_type',
#                                                          'transmission', 'owner', 'mileage', 'engine',
#                                                          'max_power', 'seats'])

#     # Make prediction
#     car_price = Model.predict(input_datamodel)
#     st.markdown('Car price can be: ' + str(car_price[0]))
