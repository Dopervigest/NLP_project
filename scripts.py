'''
This file contains all the scripts for future import
'''

import json
import dateparser

# Loads and structures the test data
def load_test_data(file_name='test_data.json'):
    with open(file_name) as json_file:
        loaded_dict = json.load(json_file)

    user_requests = loaded_dict['request']
    required_data = loaded_dict['data']

    return user_requests, required_data


# Uses NER model on n rows of data and prints outputs with true values 
def test_n_samples(model, user_requests, required_data, n=5):
    assert n > 0, "n must be greater than 0"
    for idx in range(len(user_requests)):
        name, departure, destination, date = model.extract_flight_details(user_requests[idx])
    
        true_name, true_departure, true_destination, true_date = required_data[idx].values()
    
        # Displaying extracted information
        print(f"Name: {name}, True_Name: {true_name}")
        print(f"Departure: {departure}, True_Departure: {true_departure}")
        print(f"Destination: {destination}, True_Destination: {true_destination}")
        print(f"Date: {date}, True_Date: {true_date}")
        print("---------------------------------------------")
    
        if idx == n-1:
            break


# Converts date to standard format
def convert_to_standard_date(date_string):
    parsed_date = dateparser.parse(date_string)
    if parsed_date:
        return parsed_date.strftime('%d-%m-%Y')
    else:
        return "Unspecified"
