import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title of the app
st.title('Weather Data Analysis')

# Input fields
location = st.text_input('Enter location:')
start_date = st.date_input('Start date:')
end_date = st.date_input('End date:')

# Button to trigger analysis
if st.button('Analyze'):
    # Placeholder for real data fetching and processing
    st.write(f'Fetching weather data for {location} from {start_date} to {end_date}...')
    
    # Example data
    data = pd.DataFrame({
        'Date': pd.date_range(start=start_date, end=end_date),
        'Temperature': [20 + i for i in range((end_date - start_date).days + 1)]
    })
    
    # Display data
    st.write(data)
    
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Temperature'])
    ax.set_title('Temperature over Time')
    st.pyplot(fig)