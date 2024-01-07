""""
Demand Forecasting Model from Real Time Demand Forecasting Model
"""

#Import modules
import pandas as pd
import statsmodels.api as sm
from joblib import dump

# Load the dataset
def load_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path, encoding='latin1')
    return dataset

# Clean the dataset
def clean_dataset(dataset):
    dataset_cleaned = dataset.copy()
    dataset_cleaned.drop(['Product Description', 'Order Zipcode'], axis=1)
    dataset_cleaned['Customer Lname'].fillna(value='Unknown', inplace=True)
    dataset_cleaned['Customer Zipcode'].fillna(value='Unknown', inplace=True)
    return dataset_cleaned

# Transform the cleaned dataset
def transform_dataset(dataset_cleaned):
    dataset_transformed = dataset_cleaned.copy()
    dataset_transformed['order date (DateOrders)'] = pd.to_datetime(dataset_transformed['order date (DateOrders)'])
    return dataset_transformed

# Feature engineering function
def feature_engineering(dataset_transformed):
    data_trans = dataset_transformed.copy()
    # For order date
    data_trans['order_year'] = data_trans['order date (DateOrders)'].dt.year
    data_trans['order_month'] = data_trans['order date (DateOrders)'].dt.month
    # Aggregate the values to match the months in the dataset
    aggregated_data = data_trans.groupby(['order_year', 'order_month'])['Order Item Quantity'].sum().reset_index()
    # Remove the last year due to having only 1 month
    aggregated_data = aggregated_data[aggregated_data['order_year'] != 2018]
    # Create a proper datetime index for the aggregated data
    aggregated_data['date'] = pd.to_datetime(aggregated_data['order_year'].astype(str) + '-' + aggregated_data['order_month'].astype(str))
    # Sort the data by date to ensure it's in chronological order
    aggregated_data = aggregated_data.sort_values('date')
    # The final series for forecasting
    y = aggregated_data.set_index('date')['Order Item Quantity']
    return y

# Split the dataset
def split_dataset(y):
    # Determine the split point, usually a percentage of the total data
    split_point = int(len(y) * 0.7)  # For instance, 70% for training, 30% for testing
    # Split the data
    y_train = y[:split_point]
    y_test = y[split_point:]
    return y_train, y_test

# Train the model function
def train_model(y_train):
    sarima_model = sm.tsa.statespace.SARIMAX(y_train,
                                            order=(1, 0, 0),
                                            seasonal_order=(0, 0, 1, 12), #Currect best fit after auto arima
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
    model = sarima_model.fit()
    return model

# Prepare the input data for prediction
def input_data(year, month):
    input_data = pd.DataFrame(
        {
            'order_year': [year],
            'order_month': [month]
        }
    )
    input_data = pd.DataFrame(input_data, columns=['order_year', 'order_month'])
    return year, month

# Predict on the test set
def make_predictions(model, year, month):
    input_data_time = pd.to_datetime(f"{year}-{month}-01") 
    months_ahead = (input_data_time.year - 2017) * 12 + input_data_time.month - 12 
    
    start = len(model.model.endog)
    end = start + months_ahead - 1
    
    predictions = model.predict(start=start, end=end, dynamic=False)
    predictions = predictions.iloc[-1]
    return predictions

# Model Training Process
# Loading the dataset
dataset = load_dataset(r'/home/keembo/pacmann_project/data/DataCoSupplyChainDataset.csv')

# Cleaning the dataset and Transforming the dataset
dataset_cleaned = clean_dataset(dataset)
dataset_transformed = transform_dataset(dataset_cleaned)

# Feature engineering the dataset
y = feature_engineering(dataset_transformed)

# Splitting the dataset
y_train, y_test = split_dataset(y)

# Training the model
model = train_model(y_train)

# Saving the model
dump(model, r'model/demand_forecasting_model.pkl')


