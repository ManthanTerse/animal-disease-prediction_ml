## Project Overview

This folder contains a Flask-based animal disease prediction web application. The project combines:

- A machine learning disease prediction pipeline
- A web interface for pet/animal symptom entry
- A contact form that stores messages in CSV format
- An OpenAI-powered veterinary guidance chatbot
- Jupyter notebooks for training and visual analysis

The application is designed as an educational and decision-support tool for early awareness, not as a replacement for licensed veterinary diagnosis.

## Main Technologies Used

- Python
- Flask
- pandas
- numpy
- scikit-learn
- joblib
- gunicorn
- OpenAI Python SDK
- Bootstrap 5
- Bootstrap Icons
- Jinja2 templates

## Folder Contents

### Core Application Files

`main.py`
- Main Flask application entry point
- Defines routes for home, contact, prediction, model inference, and chatbot
- Loads environment variables from `.env` or `main.env`
- Loads the trained model from `model.pkl`

`preprocess_utils.py`
- Shared preprocessing and feature-engineering helpers
- Extracts numeric values from text fields like `102 F` or `3 days`
- Standardizes text and yes/no fields
- Builds temperature, heart rate, and severity categories

`train_model.py`
- Training script for the disease prediction model
- Reads `dataset.csv`
- Applies preprocessing
- Trains a `RandomForestClassifier`
- Saves the trained model and feature columns to `model.pkl`

### Data and Model Files

`dataset.csv`
- Main dataset used for training
- Referenced in project notes as 1500 rows and 22 columns
- Contains animal profile, symptom, and disease prediction data

`model.pkl`
- Saved trained model bundle
- Stores:
  - trained pipeline/model
  - feature column list used during prediction

`output.csv`
- Additional generated data/output file in the project

`contact_messages.csv`
- Stores messages submitted from the contact form
- Appends timestamp, name, email, and message

### Notebook Files

`model_train&test.ipynb`
- Notebook for model training and testing experiments

`visuals.ipynb`
- Notebook for data exploration or visual analysis

### Configuration and Deployment Files

`.env`
- Stores environment variables such as API keys and model settings

`.gitignore`
- Git ignore rules

`requirements.txt`
- Python dependencies list

`Procfile`
- Deployment process definition, typically for platforms like Heroku/Render

`runtime.txt`
- Python runtime version reference for deployment

### Templates

`templates/base.html`
- Shared layout
- Navbar, footer, Bootstrap links, font imports and page blocks

`templates/home.html`
- Landing page
- Project intro, features, pet care tips and chatbot UI

`templates/prediction.html`
- Disease prediction form
- Displays prediction result and severity details

`templates/contact.html`
- Contact page with message form and project contact details

### Static Files

`static/styles.css`
- Active main stylesheet used by `base.html`
- Controls layout, cards, navbar, form styling, result box, chatbot section and responsive behavior

`static/style.css`
- Older or alternate stylesheet present in the folder
- Not linked from `base.html` currently

`static/bg.png`
- Background image used by `static/styles.css`

### Generated / Cache Files

`__pycache__/`
- Python bytecode cache directory

`.ipynb_checkpoints/`
- Jupyter notebook checkpoint files

`.virtual_documents/`
- IDE/Jupyter-related generated helper files

## Application Features

### 1. Disease Prediction

Users can enter:

- Animal type
- Breed
- Age
- Gender
- Weight
- Up to 4 symptoms
- Duration
- Body temperature
- Heart rate
- Several yes/no symptom flags

The app then:

1. Formats the input
2. Builds a pandas DataFrame
3. Derives temperature and heart-rate categories
4. Calculates severity level
5. Reorders columns to match training features
6. Predicts disease using the trained model
7. Shows the result on the prediction page

### 2. Severity Assessment

The project adds a rule-based severity layer using:

- Duration in days
- Temperature category
- Heart rate category
- Appetite loss
- Vomiting
- Diarrhea
- Coughing
- Labored breathing
- Lameness
- Skin lesions
- Nasal discharge
- Eye discharge

Severity output:

- Mild
- Moderate
- Severe

### 3. Contact Form

The contact page:

- Accepts name, email, and message
- Validates that fields are filled
- Saves submissions into `contact_messages.csv`
- Adds a timestamp for each message

### 4. Veterinary Chatbot

The chatbot on the home page:

- Sends messages to the `/chatbot` route using JavaScript `fetch`
- Uses the OpenAI Responses API through the `openai` package
- Keeps a small conversation history
- Restricts responses to animal and pet healthcare guidance
- Warns that it is not a substitute for a veterinarian
- Encourages emergency vet care for severe warning signs

## Flask Routes in `main.py`

`/`
- Renders `home.html`

`/home`
- Renders `home.html`

`/contact`
- `GET`: shows contact form
- `POST`: validates and stores contact message

`/prediction`
- Shows the prediction form

`/predict`
- Accepts prediction form submission
- Runs model inference
- Renders prediction result

`/chatbot`
- Accepts JSON POST requests
- Returns chatbot reply or error as JSON

## ML Pipeline Summary

Training flow in `train_model.py`:

1. Load `dataset.csv`
2. Preprocess with `preprocess_df`
3. Split features and target `Disease_Prediction`
4. Apply:
   - median imputation for numeric columns
   - most-frequent imputation + one-hot encoding for categorical columns
5. Train `RandomForestClassifier`
6. Evaluate accuracy and classification report
7. Save model bundle to `model.pkl`

Prediction flow in `main.py`:

1. Read form input
2. Normalize user text values
3. Compute derived metrics
4. Reindex input by saved feature columns
5. Run `model.predict(...)`
6. Render result in `prediction.html`

## Important Functions

From `preprocess_utils.py`:

`extract_number(value)`
- Extracts numeric value from mixed text

`clean_yes_no(value)`
- Normalizes yes/no values

`clean_text(value)`
- Standardizes text capitalization

`categorize_temperature(value)`
- Maps numeric temperature to `Low`, `Normal`, or `High`

`categorize_heart_rate(value)`
- Maps numeric heart rate to `Low`, `Normal`, or `High`

`calculate_severity_level(row)`
- Builds severity score and returns `Mild`, `Moderate`, or `Severe`

`preprocess_df(df)`
- Main shared preprocessing function for training and transformation

From `main.py`:

`load_local_env(env_path)`
- Loads `.env`-style key/value pairs into environment variables

`save_message(name, email, message)`
- Saves contact form data into CSV

`get_openai_client()`
- Initializes OpenAI client using `OPENAI_API_KEY`

`build_chat_history(history, user_message)`
- Cleans and truncates recent chat history

## Environment Variables Used

`OPENAI_API_KEY`
- Required for chatbot functionality

`OPENAI_MODEL`
- Optional
- Defaults to `gpt-5-mini`

`PORT`
- Optional app port
- Defaults to `5000`

## Dependencies from `requirements.txt`

- `flask`
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `gunicorn`
- `openai`

## Frontend Structure

The frontend uses:

- Jinja templates for page rendering
- Bootstrap 5 for layout and components
- Bootstrap Icons for iconography
- Google Fonts (`Poppins`)
- Custom CSS in `static/styles.css`
- Vanilla JavaScript for:
  - chatbot messaging
  - prediction submit spinner

## Notes About Current Structure

- `static/styles.css` is the stylesheet currently in use
- `static/style.css` appears to be an older or alternate stylesheet
- The prediction form field is named `vomitting` in HTML, while backend maps it to `Vomiting`
- Flash messages are shown in the shared base template
- The model file is large and should usually not be manually edited

