## ✈️ Airline Ticket Price Predictor

This repository contains the code for a Streamlit web application that predicts airline ticket prices based on various travel details. The application utilizes two machine learning models, XGBoost Regressor and Random Forest Regressor, to provide accurate price estimations.

### ✨ Features

  * **User-Friendly Interface:** Built with Streamlit for an intuitive and interactive user experience.
  * **Machine Learning Models:** Employs both XGBoost and Random Forest algorithms for robust price prediction.
  * **Automatic Model Selection:** The application automatically identifies and uses the better-performing model (based on R² score) for predictions.
  * **Comprehensive Input Fields:** Allows users to input details such as airline, source/destination cities, departure/arrival times, number of stops, class, flight duration, and days left until departure.
  * **Real-time Predictions:** Get instant price estimations as you adjust the travel parameters.

### 🚀 How to Run

To run this application locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required libraries:**

    ```bash
    pip install streamlit pandas numpy xgboost scikit-learn
    ```

4.  **Download the dataset:**
    Ensure you have the `Clean_Dataset.csv` file in the same directory as the `airplane.py` script. You can typically find similar airline datasets on platforms like Kaggle.

5.  **Run the Streamlit application:**

    ```bash
    streamlit run airplane.py
    ```

    This will open the application in your web browser.

### 🛠️ Project Structure

  * `airplane.py`: The main Python script containing the Streamlit application code, data loading, preprocessing, model training, and prediction logic.
  * `Clean_Dataset.csv`: (To be placed by the user) The dataset used for training the machine learning models. *Note: The provided `airplane.py` expects this file to be in the same directory.*

### 🧠 Models Used

The application trains and utilizes two regression models:

  * **XGBoost Regressor:** A highly efficient and flexible gradient boosting framework.
  * **Random Forest Regressor:** An ensemble learning method that builds multiple decision trees.

The model with the higher R² score on the test set is automatically selected for making predictions, ensuring the most accurate possible estimation.

### 📊 Data Preprocessing

The `load_data` function in `airplane.py` performs the following preprocessing steps:

  * **Categorical Feature Mapping:** Converts categorical features like `airline`, `source_city`, `destination_city`, `departure_time`, `arrival_time`, `stops`, and `class` into numerical representations using predefined mappings.
  * **Duration Conversion:** Converts flight duration from hours to minutes.
  * **Feature Scaling:** Scales the `duration` feature using `StandardScaler` to ensure it contributes appropriately to the model training.
  * **Column Dropping:** Removes the `flight` column as it's not relevant for price prediction.

### 🤝 Contributing

Feel free to fork this repository, suggest improvements, or open issues. Contributions are welcome\!

### 📄 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).
