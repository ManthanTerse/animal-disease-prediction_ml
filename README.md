# Animal Disease Prediction

This project is a Flask-based machine learning web application that predicts possible animal diseases from symptom and health-related inputs. It combines data preprocessing, feature engineering, a scikit-learn classification pipeline and a simple frontend for user interaction.

## Project Overview

The application allows a user to:

- enter animal details such as type, breed, age, weight, and gender
- provide symptoms and clinical indicators
- receive a predicted disease from a trained machine learning model
- view additional health context such as severity level, temperature category, and heart-rate category
- submit messages through a contact form
- use a rule-based chatbot for navigation and basic symptom guidance

## Animals and Data Coverage

The dataset used in this project contains:

- 8 animal types: Dog, Cat, Cow, Sheep, Pig, Horse, Rabbit, and Goat
- about 1500 records
- 22 columns including symptoms, vitals, and target disease labels
- around 140 disease classes

Main input columns:

- `Animal_Type`
- `Breed`
- `Age`
- `Gender`
- `Weight`
- `Symptom_1` to `Symptom_4`
- `Duration`
- symptom flags like `Vomiting`, `Diarrhea`, `Coughing`, `Skin_Lesions`
- `Body_Temperature`
- `Heart_Rate`

Target column:

- `Disease_Prediction`

## Machine Learning Workflow

### 1. Preprocessing

`preprocess_utils.py` standardizes raw input before training and inference:

- extracts numeric values from text like `4 days` or `39.2 C`
- converts yes/no style inputs into consistent values
- normalizes text fields using title case
- creates `Duration_Days` from `Duration`

### 2. Feature Engineering

`feature_engineering.py` adds interpretable health features:

- `Temperature_Category`: `Low`, `Normal`, `High`, or `Unknown`
- `Heart_Rate_Category`: `Low`, `Normal`, `High`, or `Unknown`
- `Severity_Level`: `Mild`, `Moderate`, or `Severe`

Severity is calculated from:

- symptom count across the yes/no clinical flags
- illness duration
- abnormal temperature
- abnormal heart rate

### 3. Model Training

`train_model.py` builds a scikit-learn pipeline with:

- `FunctionTransformer(preprocess_df)`
- `ColumnTransformer`
- numeric imputation with `SimpleImputer(strategy="median")`
- categorical imputation with `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`
- `RandomForestClassifier(random_state=42)`

The trained pipeline is saved to `model.pkl` using `joblib`.

## Web Application Flow

The Flask app in `main.py` provides these main routes:

- `/` and `/home`: landing page
- `/prediction`: disease prediction form
- `/predict`: form submission and model inference
- `/contact`: contact form that stores messages in `data/contact_messages.csv`
- `/guide_chatbot`: rule-based chatbot endpoint returning JSON responses

Prediction flow:

1. the user fills the form on the prediction page
2. Flask collects and formats the request data
3. the app builds a one-row pandas DataFrame
4. preprocessing and feature engineering are used to derive result insights
5. the saved ML pipeline predicts the disease label
6. the result page shows the prediction with severity, temperature, and heart-rate categories

## Tech Stack

- Python
- Flask
- pandas
- NumPy
- scikit-learn
- joblib
- HTML
- CSS

## Project Structure

```text
Animal_disease_prediction -/
|-- main.py
|-- train_model.py
|-- preprocess_utils.py
|-- feature_engineering.py
|-- model.pkl
|-- README.md
|-- Architecture/
|   |-- pipeline.drawio
|   |-- explain.drawio
|-- requirements.txt
|-- data/
|   |-- dataset.csv
|   |-- output.csv
|   |-- contact_messages.csv
|-- templates/
|   |-- base.html
|   |-- home.html
|   |-- prediction.html
|   |-- contact.html
|-- static/
|   |-- styles.css
|   |-- bg.png
|-- model_train&test.ipynb
|-- visuals.ipynb
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Flask app

```bash
python main.py
```

### 3. Open in browser

Visit:

```text
http://127.0.0.1:5000/
```

## Model Retraining

To retrain the model:

```bash
python train_model.py
```

Note:

- this project already includes a pretrained `model.pkl`
- if you retrain locally, confirm the dataset path in `train_model.py` matches your local file location

## Documentation Files

- `pipeline.drawio` contains the earlier pipeline-level diagram
- `explain.drawio` contains a deeper visual explanation of the full project, including data flow, preprocessing, feature engineering, training, Flask routing, and output generation

## Current Scope and Notes

- the ML dataset covers multiple animal types
- the chatbot logic in `main.py` is more focused on dog and cat guidance language
- the app is intended for educational and project demonstration purposes
- predictions should not be treated as a replacement for professional veterinary diagnosis

## Author

Mini project on animal disease prediction using Machine learning and Flask.
