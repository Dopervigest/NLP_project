import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IRModel():
    def __init__(self):
        model_path = './best_trained_model/'
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)

    def predict(self, user_input):
        inputs = self.tokenizer(user_input, return_tensors='pt').to(device)

        # Perform inference using the model
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
        
        # Map predicted class to the respective intent
        intents = {
            0: "book_flight",
            1: "misc",
            2: "thank",
            3: "end_conversation"
        }
        
        predicted_intent = intents[predicted_class]
        return predicted_intent
