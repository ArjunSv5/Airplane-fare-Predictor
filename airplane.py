import streamlit as st
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ----------- Load and preprocess dataset -------------------
@st.cache_data
def load_data():
    """
    Loads and preprocesses the airline dataset from a CSV file.
    Note: For this to work, 'Clean_Dataset.csv' should be in the same directory.
    """
    # Using a relative path is better for portability and deployment
    df = pd.read_csv('Clean_Dataset.csv', nrows=50000)

    # Define mappings for categorical features
    airline_map = {'Air_India': 0, 'AirAsia': 1, 'GO_FIRST': 2, 'Indigo': 3, 'SpiceJet': 4, 'Vistara': 5}
    location_map = {'Bangalore': 0, 'Chennai': 1, 'Delhi': 2, 'Hyderabad': 3, 'Kolkata': 4, 'Mumbai': 5}
    time_map = {'Afternoon': 0, 'Early_Morning': 1, 'Evening': 2, 'Late_Night': 3, 'Morning': 4, 'Night': 5}
    stops_map = {'zero': 0, 'one': 1, 'two_or_more': 2}
    class_map = {'Economy': 0, 'Business': 1}

    # Apply mappings
    df['airline'] = df['airline'].replace(airline_map).astype('int64')
    df['source_city'] = df['source_city'].replace(location_map).astype('int64')
    df['destination_city'] = df['destination_city'].replace(location_map).astype('int64')
    df['departure_time'] = df['departure_time'].replace(time_map).astype('int64')
    df['arrival_time'] = df['arrival_time'].replace(time_map).astype('int64')
    df['stops'] = df['stops'].replace(stops_map).astype('int64')
    df['class'] = df['class'].replace(class_map).astype('int64')
    df['duration'] = (df['duration'] * 60).astype(int)
    
    # Drop the 'flight' column as it's not needed for prediction
    df.drop('flight', axis=1, inplace=True)

    # Scale the 'duration' feature
    scale = StandardScaler()
    df['duration'] = scale.fit_transform(df[['duration']])

    return df, airline_map, location_map, time_map, stops_map, class_map, scale

# --------- Load data and train models ------------------
# This section runs once when the script starts
df, airline_map, location_map, time_map, stops_map, class_map, scaler = load_data()

X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train XGBoost Regressor
model_xgb = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1, random_state=42)
model_xgb.fit(X_train, y_train)

# Initialize and train Random Forest Regressor
model_rf = rfr(n_estimators=1000, n_jobs=-1, random_state=42)
model_rf.fit(X_train, y_train)

# --- EFFICIENT MODEL COMPARISON ---
# This is done only once, not on every button click.
r2_rf = r2_score(y_test, model_rf.predict(X_test))
r2_xgb = r2_score(y_test, model_xgb.predict(X_test))

better_model_name = "Random Forest Regressor" if r2_rf > r2_xgb else "XGBoost Regressor"


# ---------- Front-end Starts Here ------------------------
st.title("✈️ Airline Ticket Price Predictor")
st.markdown("Fill in the travel details to get an estimated ticket price.")

# --- User Inputs ---
airline = st.selectbox("Airline", list(airline_map.keys()))
source = st.selectbox("Source City", list(location_map.keys()))
destination = st.selectbox("Destination City", list(location_map.keys()))
departure = st.selectbox("Departure Time", list(time_map.keys()))
arrival = st.selectbox("Arrival Time", list(time_map.keys()))
stops = st.selectbox("Number of Stops", list(stops_map.keys()))
ticket_class = st.radio("Class", list(class_map.keys()))
duration = st.number_input("Flight Duration (in hours)", min_value=0.1, max_value=50.0, value=2.0, step=0.1)
days_left = st.number_input("Days Left for Departure", min_value=1, max_value=50, value=15, step=1)

# --- Prediction Logic ---
if st.button("Predict Price"):
    # Create a DataFrame from user inputs
    user_input = pd.DataFrame({
        'airline': [airline_map[airline]],
        'source_city': [location_map[source]],
        'departure_time': [time_map[departure]],
        'stops': [stops_map[stops]],
        'arrival_time': [time_map[arrival]],
        'destination_city': [location_map[destination]],
        'class': [class_map[ticket_class]],
        'duration': [int(duration * 60)], # Convert hours to minutes
        'days_left': [days_left]
    })

    # Scale the duration feature just like the training data
    user_input['duration'] = scaler.transform(user_input[['duration']])

    # --- CRITICAL FIX: Reorder columns to match training data ---
    user_input = user_input[X_train.columns]

    # Make predictions with both models
    pred_xgb = model_xgb.predict(user_input)[0]
    pred_rf = model_rf.predict(user_input)[0]

    # Select the prediction from the better model (determined earlier)
    final_price = pred_rf if better_model_name == "Random Forest Regressor" else pred_xgb

    st.subheader(f"🎯 Estimated Price: ₹ {int(final_price):,}")
    st.caption(f"Prediction based on best model: {better_model_name} (R² on test data: {max(r2_rf, r2_xgb):.2f})")