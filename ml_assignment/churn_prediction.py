# import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix , classification_report , accuracy_score
import pickle


# fetch the data from source
def data_ingestion(file_path): 
    data = pd.read_csv(file_path)
    
    return data

# preprocess the data (convert the data to numerical format)
def data_preprocessing(data):
    
    df = data
    # Clean the Data
    df = df.drop_duplicates()
    df = df.dropna(subset=['CustomerID'])
    df.loc[:, 'CastCustomerID'] = df['CustomerID'].astype(int).astype(str)
    df = df.drop(columns='CustomerID')
    df.rename(columns = {'CastCustomerID':'CustomerID'} , inplace=True)

    df.to_csv('./Artifacts/online_retail_data_cleaned.csv')

    # Transform the cleaned data .
    cdf = df

    cdf['InvoiceDateTime'] = pd.to_datetime(cdf['InvoiceDate'])
    cdf['Amount'] = cdf['Quantity'] * cdf['UnitPrice']
    cdf.drop(columns = ['Description','InvoiceDate','Country','Quantity',	'UnitPrice'],axis = 1 , inplace=True)

    cdf = cdf.groupby('CustomerID') \
        .agg({'InvoiceNo':'nunique' , 'StockCode':'nunique' , 'Amount':'sum' , 'InvoiceDateTime': ['min', 'max'] })
    cdf.columns = ['unique_invoiceno_count', 'unique_stockcode_count', 'total_amount', 'invoice_min_time', 'invoice_max_time']
    cdf['active_days'] = (cdf['invoice_max_time']-cdf['invoice_min_time']).dt.days
    cdf.drop(columns = ['invoice_min_time'	, 'invoice_max_time'],axis = 1 , inplace=True)
    cdf = cdf.reset_index()

    cdf.to_csv('./Artifacts/online_retail_data_transformed.csv')

    # Scale the transformed features data and Define the Churn Score and Churn Status .

    scaled_cdf = pd.DataFrame()

    scaled_cdf['CustomerID'] = cdf['CustomerID']
    scaled_cdf['invoice_count'] = (cdf['unique_invoiceno_count'] - cdf['unique_invoiceno_count'].min()) / (cdf['unique_invoiceno_count'].max() - cdf['unique_invoiceno_count'].min())
    scaled_cdf['stockcode_count'] = (cdf['unique_stockcode_count'] - cdf['unique_stockcode_count'].min()) / (cdf['unique_stockcode_count'].max() - cdf['unique_stockcode_count'].min())
    scaled_cdf['total_amount'] = (cdf['total_amount'] - cdf['total_amount'].min()) / (cdf['total_amount'].max() - cdf['total_amount'].min())
    scaled_cdf['active_days'] = (cdf['active_days'] - cdf['active_days'].min()) / (cdf['active_days'].max() - cdf['active_days'].min())

    scaled_cdf['churn_score'] = 1 - (0.3 * scaled_cdf['total_amount'] + 0.3 * scaled_cdf['invoice_count'] + 0.2 * scaled_cdf['stockcode_count'] + 0.2 * scaled_cdf['active_days'])

    scaled_cdf['churn_status'] = scaled_cdf['churn_score'].apply(lambda x:1 if x>=0.95 else 0)

    # Add Churn Score and Churn Status as target column to transformed data .
    cdf['Churn_Score'] = scaled_cdf['churn_score']
    cdf['Churn_Status'] = scaled_cdf['churn_status']
    

    cdf.to_csv('./Artifacts/online_retail_data_churn.csv')

    return cdf

    

# select the desired model to train the data 
def model_training(data):

    cdf = data 

    X = cdf[['unique_invoiceno_count',	'unique_stockcode_count', 'total_amount',	'active_days']]
    Y = cdf['Churn_Status'].values

    x_train , x_test , y_train , y_test = train_test_split(X,Y,random_state=42)

    scale = MinMaxScaler()
    x_train_scaled = scale.fit_transform(x_train)
    x_test_scaled = scale.transform(x_test)

    lr = LogisticRegression()
    lr.fit(x_train_scaled,y_train)
    y_test_predict = lr.predict(x_test_scaled)

    return y_test , y_test_predict , lr , scale


   

    
    

    
# evaluate the trained model 
def model_evaluation(y_actual , y_predict):
    
    accuracy = accuracy_score(y_actual, y_predict)
    conf_matrix = confusion_matrix(y_actual, y_predict)
    class_report = classification_report(y_actual, y_predict)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

    
# save the model to disk
def save_model(model):
    
    with open('./Artifacts/customer_churn_model.pkl',  'wb') as f:
        pickle.dump(model, f)

def save_scaler(model):
    
    with open('./Artifacts/customer_churn_scaler.pkl',  'wb') as f:
        pickle.dump(model, f)

    
    

# create the work flow
def main():
    file_path = "./Data/online_retail_data.csv"
    data = data_ingestion(file_path)
    preprocessed_data = data_preprocessing(data)
    actual_test_target_data , predicted_test_target_data , model , scaler = model_training(preprocessed_data)
    model_evaluation(actual_test_target_data , predicted_test_target_data)
    save_model(model)
    save_scaler(scaler)
if __name__ == "__main__":
    main()