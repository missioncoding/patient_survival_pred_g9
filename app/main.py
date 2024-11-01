import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import gradio
from fastapi import FastAPI, Request, Response

import random
import numpy as np
import pandas as pd
import gradio
import joblib
# from titanic_model.processing.data_manager import load_dataset, load_pipeline
# from titanic_model import __version__ as _version
# from titanic_model.config.core import config
# from sklearn.model_selection import train_test_split
# from titanic_model.predict import make_prediction

# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# FastAPI object
app = FastAPI()

save_file_name = "xgboost-model.pkl"
model = joblib.load(save_file_name)


def predict_death_event(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, 
                        high_blood_pressure, platelets, serum_creatinine, serum_sodium, 
                        sex, smoking, time):
    # Organize the input as a numpy array and reshape it for the model
    user_input = np.array([age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, 
                           high_blood_pressure, platelets, serum_creatinine, serum_sodium, 
                           sex, smoking, time]).reshape(1, -1)
    
    pred = model.predict(user_input)
    return "Death Event" if pred[0] == 1 else "No Death Event"

# Gradio interface to generate UI link
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,
                         inputs = [
        gradio.Slider(0, 100, step=1, label="Age"),  # Numerical
        gradio.Radio([0, 1], label="Anaemia"),   # Categorical (0 = No, 1 = Yes)
        gradio.Slider(23, 7861, step=1, label="Creatinine Phosphokinase"),  # Numerical
        gradio.Radio([0, 1], label="Diabetes"),  # Categorical (0 = No, 1 = Yes)
        gradio.Slider(10, 80, step=1, label="Ejection Fraction"),           # Numerical
        gradio.Radio([0, 1], label="High Blood Pressure"),              # Categorical
        gradio.Slider(25000, 850000, step=1000, label="Platelets"),         # Numerical
        gradio.Slider(0.5, 10.0, step=0.1, label="Serum Creatinine"),       # Numerical
        gradio.Slider(110, 150, step=1, label="Serum Sodium"),              # Numerical
        gradio.Radio([0, 1], label="Sex"),                              # Categorical (0 = Female, 1 = Male)
        gradio.Radio([0, 1], label="Smoking"),                          # Categorical (0 = No, 1 = Yes)
        gradio.Slider(1, 300, step=1, label="Time")                      # Numerical
    ],
                         outputs = "text",
                         title = title,
                         description = description,
                         allow_flagging='never')



# # UI - Input components
# in_Pid = gradio.Textbox(lines=1, placeholder=None, value="79", label='Passenger Id')
# in_Pclass = gradio.Radio(['1', '2', '3'], type="value", label='Passenger class')
# in_Pname = gradio.Textbox(lines=1, placeholder=None, value="Caldwell, Master. Alden Gates", label='Passenger Name')
# in_sex = gradio.Radio(["Male", "Female"], type="value", label='Gender')
# in_age = gradio.Textbox(lines=1, placeholder=None, value="14", label='Age of the passenger in yrs')
# in_sibsp = gradio.Textbox(lines=1, placeholder=None, value="0", label='No. of siblings/spouse of the passenger aboard')
# in_parch = gradio.Textbox(lines=1, placeholder=None, value="2", label='No. of parents/children of the passenger aboard')
# in_ticket = gradio.Textbox(lines=1, placeholder=None, value="248738", label='Ticket number')
# in_cabin = gradio.Textbox(lines=1, placeholder=None, value="A5", label='Cabin number')
# in_embarked = gradio.Radio(["Southampton", "Cherbourg", "Queenstown"], type="value", label='Port of Embarkation')
# in_fare = gradio.Textbox(lines=1, placeholder=None, value="29", label='Passenger fare')

# # UI - Output component
# out_label = gradio.Textbox(type="text", label='Prediction', elem_id="out_textbox")

# # Label prediction function
# def get_output_label(in_Pid, in_Pclass, in_Pname, in_sex, in_age, in_sibsp, in_parch, in_ticket, in_cabin, in_embarked, in_fare):
    
#     input_df = pd.DataFrame({"PassengerId": [in_Pid], 
#                              "Pclass": [int(in_Pclass)], 
#                              "Name": [in_Pname],
#                              "Sex": [in_sex.lower()], 
#                              "Age": [float(in_age)], 
#                              "SibSp": [int(in_sibsp)],
#                              "Parch": [int(in_parch)], 
#                              "Ticket": [in_ticket], 
#                              "Cabin": [in_cabin],
#                              "Embarked": [in_embarked[0]], 
#                              "Fare": [float(in_fare)]})
    
#     result = make_prediction(input_data=input_df.replace({np.nan: None}))["predictions"]
#     label = "Survive" if result[0]==1 else "Not Survive"
#     return label


# # Create Gradio interface object
# iface = gradio.Interface(fn = get_output_label,
#                          inputs = [in_Pid, in_Pclass, in_Pname, in_sex, in_age, in_sibsp, in_parch, in_ticket, in_cabin, in_embarked, in_fare],
#                          outputs = [out_label],
#                          title="Titanic Survival Prediction API  ⛴",
#                          description="Predictive model that answers the question: “What sort of people were more likely to survive?”",
#                          allow_flagging='never'
#                          )

# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gradio.mount_gradio_app(app, iface, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
