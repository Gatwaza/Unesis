import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataset import MedicalDataset
from preprocess import preprocess_data

def load_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model

def predict(model, tokenizer, input_text, lab_results):
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    inputs['lab_results'] = torch.tensor(lab_results).unsqueeze(0)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
    
    return predictions

if __name__ == '__main__':
    model_path = 'fine-tuned-model'  # Path to the fine-tuned model
    model = load_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained('allenai/llama-3-7B')

    # Example input
    input_text = "Patient has a history of hypertension and exhibits symptoms of fatigue."  # Replace with actual input text
    lab_results = [5.0, 140, 7.4]  # Replace with actual normalized lab results

    prediction = predict(model, tokenizer, input_text, lab_results)
    print(f'Predicted disease class: {prediction.item()}')
