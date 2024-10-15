# forest_fire_predictor.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Function to load data
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df

# Function to handle missing values
def handle_missing_values(df):
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Fill numeric columns with the mean
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

    # Fill categorical columns with the most frequent value
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

    return df

# Function to preprocess data
def preprocess_data(df):
    # Handle missing values
    df = handle_missing_values(df)

    # Encode categorical columns if any
    labelencoder = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = labelencoder.fit_transform(df[column])

    return df

# Function to train a multi-output model
def train_models(df):
    target_columns_reg = ['Fire Size (hectares)', 'Fire Duration (hours)', 'Suppression Cost ($)']  # Regression targets
    target_column_cls = 'Fire Occurrence'  # Classification target

    # Check if the target columns exist in the DataFrame
    for target_column in target_columns_reg + [target_column_cls]:
        if target_column not in df.columns:
            st.error(f"Target column '{target_column}' not found in the dataset.")
            return None, None, None

    X = df.drop(target_columns_reg + [target_column_cls], axis=1)  # Features (remove target variables)
    y_reg = df[target_columns_reg]  # Regression targets
    y_cls = df[target_column_cls]  # Classification target

    # Train-test split
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.3, random_state=42)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.3, random_state=42)

    # Train a MultiOutput Random Forest model for regression
    regressor = MultiOutputRegressor(RandomForestRegressor())
    regressor.fit(X_train, y_train_reg)

    # Train a Random Forest model for classification
    classifier = RandomForestClassifier()
    classifier.fit(X_train_cls, y_train_cls)

    return regressor, classifier

# Function to predict based on input data
def predict_fire(regressor, classifier, input_data):
    # Regression predictions
    predictions_reg = regressor.predict([input_data])[0]

    # Classification prediction
    prediction_cls = classifier.predict([input_data])[0]

    return predictions_reg, prediction_cls

# Streamlit app
def main():
    st.title("Wildfire Prediction App")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

    if uploaded_file:
        # Load and preprocess the data
        df = load_data(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

        # Preprocess the data (includes handling missing values)
        df = preprocess_data(df)

        # Train the models
        regressor, classifier = train_models(df)

        if regressor and classifier:
            st.write("Models trained successfully.")

            # Get input for prediction, including vegetation and region as word options
            st.write("Input feature values for prediction:")

            input_data = []

            # Assuming 'vegetation' and 'region' are columns in the dataset
            vegetation_options = ['Forest', 'Shrubland', 'Grassland']  # Example options
            region_options = ['North', 'South', 'East', 'West']  # Example options

            # Dropdown for vegetation
            vegetation = st.selectbox("Select Vegetation Type", vegetation_options)
            input_data.append(vegetation_options.index(vegetation))  # Convert vegetation to numerical index

            # Dropdown for region
            region = st.selectbox("Select Region", region_options)
            input_data.append(region_options.index(region))  # Convert region to numerical index

            # Provide user input fields for the remaining features
            for col in df.drop(['Fire Size (hectares)', 'Fire Duration (hours)', 'Suppression Cost ($)', 'Fire Occurrence', 'Vegetation Type', 'Region'], axis=1).columns:
                val = st.number_input(f"Input {col}", value=0.0)
                input_data.append(val)

            # User input for Fire Occurrence (Yes/No)
            fire_occurrence = st.selectbox("Did a fire occur?", ["Yes", "No"])
            input_data.append(1 if fire_occurrence == "Yes" else 0)

            # Make predictions
            if st.button("Predict Fire Characteristics"):
                predictions_reg, prediction_cls = predict_fire(regressor, classifier, input_data)
                fire_size, fire_duration, suppression_cost = predictions_reg

                # Display the predictions
                st.write("### Predictions:")
                st.write(f"**Fire Size (hectares):** {fire_size:.2f}")
                st.write(f"**Fire Duration (hours):** {fire_duration:.2f}")
                st.write(f"**Suppression Cost ($):** ${suppression_cost:,.2f}")
                st.write(f"**Fire Occurrence Prediction:** {'Yes' if prediction_cls == 1 else 'No'}")

if __name__ == "__main__":
    main()
