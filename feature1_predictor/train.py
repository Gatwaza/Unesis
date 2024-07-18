import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from dataset import MedicalDataset
from preprocess import preprocess_data

def main():
    # Load and preprocess data
    train_dataset_path = '/mnt/data/heart_data.csv'  # Path to your training dataset
    
    tokenized_texts, normalized_lab_results, labels = preprocess_data(train_dataset_path)
    
    dataset = MedicalDataset(tokenized_texts, normalized_lab_results, labels)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained('allenai/llama-3-7B', num_labels=len(set(labels)))
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,  # For simplicity, use the same dataset for eval (change as needed)
    )
    
    # Train the model
    trainer.train()
    
    # Save the fine-tuned model
    model.save_pretrained('fine-tuned-model')

if __name__ == '__main__':
    main()
