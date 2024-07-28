import streamlit as st
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.title('Ridge Regression Real-Time Interface')

# Upload dataset
data_file = st.file_uploader('Upload Dataset', type=['csv', 'xlsx'])

if data_file:
    # Load dataset
    if data_file.name.endswith('.csv'):
        data = pd.read_csv(data_file)
    else:
        data = pd.read_excel(data_file)
    
    st.write('Dataset Preview:', data.head())
    
    # Select features and target
    features = st.multiselect('Select Features', options=data.columns)
    target = st.selectbox('Select Target', options=data.columns)
    
    if features and target:
        X = data[features]
        y = data[target]

        # Split the data
        test_size = st.slider('Test Size', 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Set ridge regression parameters
        alpha = st.slider('Alpha (Regularization Strength)', 0.01, 10.0, 1.0)
        
        # Train the model
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        st.write(f'Mean Squared Error: {mse}')
        
        # Plot predictions
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Ridge Regression Predictions')
        st.pyplot(fig)

        # Option to download the model (pickle file)
        import pickle
        model_file = st.text_input('Enter a filename to save the model:')
        if st.button('Save Model'):
            with open(f'{model_file}.pkl', 'wb') as f:
                pickle.dump(model, f)
            st.write(f'Model saved as {model_file}.pkl')
