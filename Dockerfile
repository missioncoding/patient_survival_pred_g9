# Pull Python base image
FROM python:3.10

# Specify working directory
WORKDIR /patient_survival_pred_g9

# Add requirements file
ADD requirements.txt .

# Add model pickle file instead of .whl, /patient_survival_pred_g9/
ADD xgboost-model.pkl .  

# Update pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Copy application files
ADD /app/* ./app/

# Expose port for application
EXPOSE 8001

# Start FastAPI application
CMD ["python", "app/main.py"]
