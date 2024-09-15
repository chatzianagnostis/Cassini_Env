import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pmdarima import auto_arima  # Library for automatic ARIMA parameter selection
import data

# Create a dictionary to store the data for each year
data_by_year = {
    2020: pd.DataFrame(data.data_2020),
    2021: pd.DataFrame(data.data_2021),
    2022: pd.DataFrame(data.data_2022),
    2023: pd.DataFrame(data.data_2023),
    2024: pd.DataFrame(data.data_2024),
}

def clean_data(data_by_year):
    all_years_data = []
    for year, df in data_by_year.items():
        df['Year'] = year  # Add the year to the DataFrame
        all_years_data.append(df)  # Collect all DataFrames

    # Combine all DataFrames into one
    combined_data = pd.concat(all_years_data, ignore_index=True)

    # Convert "Year" and "Month" into a single datetime column
    combined_data['Date'] = pd.to_datetime(combined_data['Year'].astype(str) + '-' + combined_data['Month'], format='%Y-%B')

    # Set "Date" as the index
    combined_data.set_index('Date', inplace=True)

    # Drop the "Month" and "Year" columns, and remove columns with NaN values
    combined_data.drop(columns=['Month', 'Year'], inplace=True)
    combined_data.dropna(axis=1, inplace=True)  # Drop columns with NaN values

    return combined_data

def auto_train_arima(city_data):
    # Automatically find the best ARIMA parameters using auto_arima
    model = auto_arima(city_data, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
    return model

def forecast_next_quarter(model, steps=3):
    # Forecast the next 3 months
    forecast = model.predict(n_periods=steps)
    return forecast

def plot_results(city, train_data, forecast):
    # Create a plot to show the train data and predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data, label="Training Data", color="blue")
    
    # Create a time range for the forecast
    forecast_index = pd.date_range(start=train_data.index[-1] + pd.DateOffset(1), periods=len(forecast), freq='M')
    plt.plot(forecast_index, forecast, label="Forecast", color="red", linestyle='--')

    # Connect the last training point to the first forecast point with a dashed line
    plt.plot([train_data.index[-1], forecast_index[0]], [train_data.iloc[-1], forecast[0]], color="blue", linestyle='--')

    # Format the x-axis to display the months and years vertically
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Show month and year
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Set the locator to months

    # Rotate the date labels to 90 degrees for vertical orientation
    plt.xticks(rotation=90)

    plt.title(f"Forecast vs Training Data for {city}")
    plt.xlabel("Date")
    plt.ylabel("Tourists")
    plt.legend()
    plt.grid(True)

    # Save the plot as an image
    plt.savefig(f"{city}_forecast.png")

    plt.show()


def split_train_test(city_data):
    # Split the data into training (up to May 2024) and test set (June, July, August 2024)
    train_data = city_data[:'2024-05-31']  # Training data up to May 2024
    test_data = city_data['2024-06-01':'2024-08-31']  # Test data for June, July, August 2024
    return train_data, test_data

def main():
    # Step 1: Clean and prepare the time series data
    time_series_data = clean_data(data_by_year)

    # Step 2: Loop over all cities (e.g., Athens, Thessaloniki)
    cities = time_series_data.columns

    for city in cities:
        print(f"Training ARIMA for city: {city}")
        city_data = time_series_data[city]  # Extract data for the current city

        # Split data into training and test sets
        train_data, test_data = split_train_test(city_data)

        # Automatically train an ARIMA model on the training data
        model = auto_train_arima(train_data)

        # Forecast the next quarter (3 months)
        forecast = forecast_next_quarter(model)

        # Plot the results with training data and forecasted data
        plot_results(city, train_data, forecast)


if __name__ == "__main__":
    main()
