# Use the official PyTorch image as base image
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements and code files
COPY requirements.txt .
COPY train.py .
COPY preprocess.py .
COPY dataset.py .
COPY predict.py .
COPY preprocessed/ preprocessed/
COPY fine-tuned-model/ fine-tuned-model/

# Install dependencies
RUN pip install -r requirements.txt

# Set entrypoint for prediction script by default
ENTRYPOINT ["python", "predict.py"]
