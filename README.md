# Survival Prediction App

This is a [Streamlit]([https://streamlit.io/](https://zinc-phosphide-survival-predictor-yggducdjmzss8dkvnzsahs.streamlit.app/)) web application for predicting **survival (1 = Survived, 0 = Died)** after exposure to zinc phosphide or similar toxic agents.  
It uses a trained Random Forest model based on experimental data.

## Features

- Enter data manually or upload a data file.
- Predict the outcome (Survived or Died) using a trained model.
- Download predictions as a CSV file.
- Simple, clear interface, and easy to deploy globally.

## Project Structure

- `app.py` – Streamlit application code.
- `model.pkl` – Pre-trained Random Forest model.
- `requirements.txt` – List of required Python packages.

## Running Locally

To run the app locally, clone the repository, install dependencies, and start Streamlit:

```bash
git clone https://github.com/YOUR_USERNAME/zinc-phosphide-survival-predictor.git
cd zinc-phosphide-survival-predictor
pip install -r requirements.txt
streamlit run app.py
