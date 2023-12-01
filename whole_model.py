import pandas as pd

from ner_models import FlairModel
from ir_model import IRModel

class FlightBookingSystem:
    def __init__(self, available_flights):       
        self.ner_model = FlairModel()
        self.ir_model = IRModel()
        
        self.available_flights = available_flights


    def run(self):
        self.user_data = {}
        self.model_state = 'Neutral'
        
        print("Model: Hello! I'm here to help you with flight booking.")
        while True:                
            user_input = input("User: ").capitalize()

            self.intent = self.predict_intent(user_input.strip())
            
            if self.intent == "thank":
                print("Model: Happy to help!")
                continue

            elif self.intent == "end_conversation":
                print("Model: Goodbye!")
                break

            elif self.model_state == 'Neutral' and self.intent == "book_flight":
                self.model_state = 'Booking'
                self.user_data['name'], self.user_data['departure'], self.user_data['destination'], self.user_data['date'] = self.extract_entities(user_input)

                answer = self.check_data()
                
                if answer:
                    continue
                else:
                    break
                        
            elif self.model_state == 'Booking':
                new_data = self.extract_entities(user_input)
                self.update_data(new_data)
                answer = self.check_data()
                if answer:
                    continue
                else:
                    break

            else:
                print("Model: I'm not sure what you mean. I'm designed only to book flight tickets. Could you please rephrase?")
                
    def predict_intent(self, user_input):
        intent = self.ir_model.predict(user_input)
        return intent

    def extract_entities(self, user_input):
        name, departure, destination, date = self.ner_model.extract_flight_details(user_input)
        return name, departure, destination, date
    
    def update_data(self, new_data):
        flg=None
        if self.user_data['departure'] != 'Unspecified' and self.user_data['destination'] == 'Unspecified':
            flg=True

        for idx, key in enumerate(list(self.user_data.keys())):
            if self.user_data[key] == 'Unspecified':
                self.user_data[key] = new_data[idx]

        if self.user_data['destination'] == 'Unspecified' and flg:
            self.user_data['destination'] = new_data[1]



    def check_data(self):
        is_name = self.user_data['name'] != 'Unspecified'
        is_departure = self.user_data['departure'] != 'Unspecified'
        is_destination = self.user_data['destination'] != 'Unspecified'
        is_date = self.user_data['date'] != 'Unspecified'

        if is_name and is_departure and is_destination and is_date:
            is_available = self.check_flight_availability()
            if is_available:
                return self.final_check()     
                
            else:
                print("Model: Sorry, the requested flight is not available.")
                return False

        elif is_departure and is_destination and is_date:
            is_available = self.check_flight_availability()
            if is_available:
                self.request_name()
                return True
                                   
            else:
                print("Model: Sorry, the requested flight is not available.")
                return False
                    
                
        elif is_departure and is_destination:
            answer = self.get_list_of_flights()
            if answer:
                print(f'Model: We have this flight available for these dates: {answer}')
                self.request_date()
                date = input('User: ')
                if date.strip() in answer:
                    self.user_data['date'] = date
                    self.check_data()
                else: 
                    self.update_data(self.extract_entities(date))
                    self.check_data()
                    return True
            else:
                print(f'Model: Sorry, the requested flight is not available.')
                return False
                
        elif not is_name and not is_departure and not is_destination and not is_date:
            self.request_all()
            return True
        
        elif not is_destination and not is_departure:
            self.request_departure_destination()
            return True
        
        elif not is_destination:
            self.request_destination()
            return True

        elif not is_departure:
            self.request_departure()
            return True
            
        elif not is_name:
            self.request_name()
            return True
            
        elif not is_date:
            self.request_date()
            return True

        else:
            self.request_all()
            return True


    def request_name(self):
        print("Model: Please provide your full name: ")
    def request_departure(self):
        print("Model: Please provide the city of departure: ")
    def request_destination(self):
        print("Model: Please provide the city of destination: ")
    def request_date(self):
        print("Model: Please provide the date of the flight: ")
    def request_departure_destination(self):
        print("Model: Please provide the city of departure and the city of destination: ")
    def request_all(self):
        print("Model: Please provide your name, city of departure, city of destination, and date of flight: ")
            


    def check_flight_availability(self):
        condition = ((self.available_flights['Departure'] == self.user_data['departure']) &
                     (self.available_flights['Destination'] == self.user_data['destination']) &
                     (self.available_flights['Date'] == self.user_data['date']) &
                     (self.available_flights['Available seats'] > 0))
        if self.available_flights.loc[condition].shape[0] != 0:
            return True
        else:
            return False

    def get_list_of_flights(self):
        condition = ((self.available_flights['Departure'] == self.user_data['departure']) &
                     (self.available_flights['Destination'] == self.user_data['destination']) &
                     (self.available_flights['Available seats'] > 0))
        if self.available_flights.loc[condition].shape[0] != 0:
            return list(self.available_flights.loc[condition, 'Date'])
        else:
            return False
    
    def book_flight(self):
        condition = ((self.available_flights['Departure'] == self.user_data['departure']) &
                     (self.available_flights['Destination'] == self.user_data['destination']) &
                     (self.available_flights['Date'] == self.user_data['date']))
        self.available_flights.loc[condition, 'Available seats'] -= 1
        self.available_flights.loc[condition, 'Passengers'] += f'{self.user_data["name"]}, '
        print('Model: The requested flight is successfully booked! Thank you for using our service!')


    def final_check(self):
        print("Model: Let's check the data one last time to be sure.")
        print(f"Model: Your name is {self.user_data['name']}, the city of departure is {self.user_data['departure']}, the destination is {self.user_data['destination']}, and the date is {self.user_data['date']}, right? Answer only 'Yes' or 'No'.")
        answer = input('User: ')
        if 'yes' in answer.lower():
            self.book_flight()
        else:
            self.model_state = 'Neutral' 
            print("Model: Then please repeat your request and state your name, city of departure, city of destination and date of flight.")
            return True
