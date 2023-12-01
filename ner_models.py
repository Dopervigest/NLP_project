'''
This file contains all the NER models from jupyter notebooks 2-3
'''

import re
import spacy

import datefinder

from flair.data import Sentence
from flair.models import SequenceTagger

from scripts import convert_to_standard_date

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Baseline regex-based method 
class RegexModel():
    def __init__(self):
        # Regular expressions to extract information
        self.pattern_name = r'[A-Z][a-z]+\s[A-Z][a-z]+' # gets first name and last name
        self.pattern_departure = r'from\s([A-Z][a-z]+)' # gets capitalized noun after word 'from'
        self.pattern_destination = r'to\s([A-Z][a-z]+)' # gets capitalized noun after word 'to' 
        self.pattern_date = r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s\d{4}\b'

    def extract_flight_details(self, user_request):
        # Initializing base variables
        name = 'Unspecified'
        departure = 'Unspecified'
        destination = 'Unspecified'
        date = 'Unspecified'
    
        # Extracting information
        match_name = re.search(self.pattern_name, user_request)
        if match_name:
            name = match_name.group()
    
        match_departure = re.search(self.pattern_departure, user_request)
        if match_departure:
            departure = match_departure.group(1)
    
        match_destination = re.search(self.pattern_destination, user_request)
        if match_destination:
            destination = match_destination.group(1)
    
        match_date = re.search(self.pattern_date, user_request)
        if match_date:
            date = match_date.group()
            date = convert_to_standard_date(date)
    
        return name, departure, destination, date

# Huggingface base BERT model
class HfBERTModel():
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")        
        self.model = pipeline("ner", model=model, tokenizer=tokenizer)

    def extract_flight_details(self, user_request):
        # Initializing base variables
        name = 'Unspecified'
        departure = 'Unspecified'
        destination = 'Unspecified'
        date = 'Unspecified'
        
        ner_results = self.model(user_request)
        
        departure_end = None
        destination_end = None
        
        for entity in ner_results:           
            if entity['entity'] == 'B-PER' or entity['entity'] == 'I-PER':
                if name == 'Unspecified':
                    name = entity['word']
                else:
                    name += " " + entity['word']
                    
            elif entity['entity'] == 'B-LOC' or entity['entity'] == 'I-LOC':
                if departure == 'Unspecified':
                    if ' from ' in user_request[:entity['start']]:
                        departure = entity['word']
                        departure_end = entity['end']
                    elif ' to ' in user_request[:entity['start']]:
                        destination = entity['word']
                        destination_end = entity['end']
                        
                elif departure_end is not None and entity['start'] == departure_end+1:
                        departure += ' '+ entity['word']
                
                elif destination == 'Unspecified':
                    destination = entity['word']
                    destination_end = entity['end']

                elif destination_end is not None and entity['start'] == destination_end+1:
                        destination += ' '+ entity['word']
                else:
                    pass

        matches = datefinder.find_dates(user_request)
        match = next(matches, None)
        date = match.strftime("%d-%m-%Y") if match else 'Unspecified'
            
            
        return name.title(), departure.title(), destination.title(), date



# Huggingface base BERT Uncased model
class HfBERTUncasedModel():
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")        
        self.model = pipeline("ner", model=model, tokenizer=tokenizer)

    def extract_flight_details(self, user_request):
        # Initializing base variables
        name = 'Unspecified'
        departure = 'Unspecified'
        destination = 'Unspecified'
        date = 'Unspecified'
        
        ner_results = self.model(user_request)
        
        departure_end = None
        destination_end = None
        
        for entity in ner_results:           
            if entity['entity'] == 'B-PER' or entity['entity'] == 'I-PER':
                if name == 'Unspecified':
                    name = entity['word']
                else:
                    name += " " + entity['word']
                    
            elif entity['entity'] == 'B-LOC' or entity['entity'] == 'I-LOC':
                if departure == 'Unspecified':
                    if ' from ' in user_request[:entity['start']]:
                        departure = entity['word']
                        departure_end = entity['end']
                    elif ' to ' in user_request[:entity['start']]:
                        destination = entity['word']
                        destination_end = entity['end']
                        
                elif departure_end is not None and entity['start'] == departure_end+1:
                        departure += ' '+ entity['word']
                
                elif destination == 'Unspecified':
                    destination = entity['word']
                    destination_end = entity['end']

                elif destination_end is not None and entity['start'] == destination_end+1:
                        destination += ' '+ entity['word']
                else:
                    pass

        matches = datefinder.find_dates(user_request)
        match = next(matches, None)
        date = match.strftime("%d-%m-%Y") if match else 'Unspecified'
            
        return name.title(), departure.title(), destination.title(), date





# Huggingface RoBERTa model
class HfRoBERTaModel():
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
        model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
        self.model = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        
    def extract_flight_details(self, user_request):
        # Initializing base variables
        name = 'Unspecified'
        departure = 'Unspecified'
        destination = 'Unspecified'
        date = 'Unspecified'
        
        ner_results = self.model(user_request)

        for entity in ner_results:
            if entity['entity_group'] == 'PER':
                if name == 'Unspecified':
                    name = entity['word']
                    
            elif entity['entity_group'] in ['LOC']:
                if departure == 'Unspecified':
                    if ' from ' in user_request[:entity['start']]:
                        departure = entity['word']
                    elif ' to ' in user_request[:entity['start']]:
                        destination = entity['word']
                elif destination == 'Unspecified':
                    destination = entity['word']
        
        matches = datefinder.find_dates(user_request)
        match = next(matches, None)
        date = match.strftime("%d-%m-%Y") if match else 'Unspecified'
        
        return name.title().strip(), departure.title().strip(), destination.title().strip(), date



# Huggingface Electra model with large discriminator
class HfElectraModel():
    def __init__(self):
        self.model = pipeline("ner", model="dbmdz/electra-large-discriminator-finetuned-conll03-english", grouped_entities=True)

    def extract_flight_details(self, user_request):
        # Initializing base variables
        name = 'Unspecified'
        departure = 'Unspecified'
        destination = 'Unspecified'
        date = 'Unspecified'
        
        ner_results = self.model(user_request)

        for entity in ner_results:
            if entity['entity_group'] == 'PER':
                if name == 'Unspecified':
                    name = entity['word']
                    
            elif entity['entity_group'] in ['LOC']:
                if departure == 'Unspecified':
                    if ' from ' in user_request[:entity['start']]:
                        departure = entity['word']
                    elif ' to ' in user_request[:entity['start']]:
                        destination = entity['word']
                elif destination == 'Unspecified':
                    destination = entity['word']

        matches = datefinder.find_dates(user_request)
        match = next(matches, None)
        date = match.strftime("%d-%m-%Y") if match else 'Unspecified'
            
        return name.title(), departure.title(), destination.title(), date

# standard SpaCy pipeline
class SpacyLgModel():
    def __init__(self):
        self.model = spacy.load("en_core_web_lg")
        
    def extract_flight_details(self, user_request):
        sentence = self.model(user_request)
         
        name = 'Unspecified'
        departure = 'Unspecified'
        destination = 'Unspecified'
        date = 'Unspecified'
         
        for entity in sentence.ents:
            if entity.label_ == 'PERSON' and name == 'Unspecified':
                name = entity.text
            elif entity.label_ in ['NORP', 'GPE', 'LOC']:
                if departure == 'Unspecified':
                    if ' from ' in user_request[:user_request.index(entity.text)]:
                        departure = entity.text
                    elif ' to ' in user_request[:user_request.index(entity.text)]:
                        destination = entity.text
                    else: departure = entity.text
                elif destination == 'Unspecified':
                    destination = entity.text 
            elif entity.label_ == "DATE":  # Recognize date
                date = entity.text
                if 'the ' in date:
                    date = date.replace('the ', '')
                if ' of ' in date:
                    date = date.replace(' of ', ' ')
                if 'around' in date:
                    date = date.replace('around', '')
                date = date.strip()
                date = convert_to_standard_date(date)
                    
        return name.title(), departure.title(), destination.title(), date


# SpaCy pipeline with RoBERTa model 
class SpacyTrfModel():
    def __init__(self):
        self.model = spacy.load("en_core_web_trf")
        
    def extract_flight_details(self, user_request):
        sentence = self.model(user_request)
         
        name = 'Unspecified'
        departure = 'Unspecified'
        destination = 'Unspecified'
        date = 'Unspecified'
         
        for entity in sentence.ents:
            if entity.label_ == 'PERSON' and name == 'Unspecified':
                name = entity.text
            elif entity.label_ in ['NORP', 'GPE', 'LOC']:
                if departure == 'Unspecified':
                    if ' from ' in user_request[:user_request.index(entity.text)]:
                        departure = entity.text
                    elif ' to ' in user_request[:user_request.index(entity.text)]:
                        destination = entity.text
                    else: departure = entity.text
                elif destination == 'Unspecified':
                    destination = entity.text 
            elif entity.label_ == "DATE":  # Recognize date
                date = entity.text
                if 'the ' in date:
                    date = date.replace('the ', '')
                if ' of ' in date:
                    date = date.replace(' of ', ' ')
                if 'around' in date:
                    date = date.replace('around', '')
                date = date.strip()
                date = convert_to_standard_date(date)
                    
        return name.title(), departure.title(), destination.title(), date



class FlairModel():
    def __init__(self):
        self.model = SequenceTagger.load("flair/ner-english-ontonotes-large")
        
    def extract_flight_details(self, user_request):
        sentence = Sentence(user_request)
        
        # predict NER tags
        self.model.predict(sentence)
        
        # Initializing base variables
        name = 'Unspecified'
        departure = 'Unspecified'
        destination = 'Unspecified'
        date = 'Unspecified'
        
        # Extract named entities and their labels
        for entity in sentence.get_spans('ner'):
            if entity.tag == 'PERSON':
                name = entity.text
            elif entity.tag in ['GPE', 'LOC']: 
                if departure == 'Unspecified':
                    if ' from ' in user_request[:entity.start_position]:
                        departure = entity.text
                    elif ' to ' in user_request[:entity.start_position]:
                        destination = entity.text
                    else: departure = entity.text
                elif destination == 'Unspecified':
                    destination = entity.text
                        
            elif entity.tag == 'DATE':
                date = entity.text
                if 'the ' in date:
                    date = date.replace('the ', '')
                if ' of ' in date:
                    date = date.replace(' of ', ' ')
                if 'around' in date:
                    date = date.replace('around', '')
                date = convert_to_standard_date(date)
                    
        return name.title(), departure.title(), destination.title(), date
