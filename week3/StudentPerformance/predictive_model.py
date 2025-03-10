# import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix , classification_report
import pickle

# fetch the data from source
def data_ingestion(file_path): 
    data = pd.read_csv(file_path)
   
    return data

# preprocess the data (convert the data to numerical format)
def data_preprocessing(data):
    global numerical_columns , categorical_columns , target_column
    numerical_columns = ['Age','Grades',	'Attendance',	'TimeSpentOnHomework']
    categorical_columns = ['Gender',	'SocioeconomicStatus'	,'ClassParticipation']
    target_column = ['AcademicPerformanceStatus']
    
    pd.set_option('future.no_silent_downcasting', True)

    data_categorical_encoed = pd.get_dummies(data , columns = categorical_columns).replace({False:0 , True:1})
    data_categorical_encoed['AcademicPerformanceStatus'] = data_categorical_encoed['AcademicPerformanceStatus'].map({'Pass':1,'Fail':0})
    data_encoded = data_categorical_encoed.drop(['StudentID'] , axis = 1)
    
    return data_encoded

# select the desired model to train the data 
def model_training(data):

    X = data.drop(columns=['AcademicPerformanceStatus'] , axis = 1)
    Y = data[['AcademicPerformanceStatus']]

    x_train , x_test , y_train , y_test = train_test_split(X,Y , stratify=Y , random_state=42)
    
    y_train , y_test = y_train.values.ravel() , y_test.values.ravel()

    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both the train and test data
    x_train_scaled = scaler.fit_transform(x_train)  # Fit and transform on the training set
    x_test_scaled = scaler.transform(x_test) 

    lr = LogisticRegression() 

    lr.fit(x_train_scaled ,y_train)

    y_test_pred = lr.predict(x_test_scaled)
    
    print("Feature Importance : Student Performance Status")
    feature_importance_df = pd.DataFrame({
    'Feature': list(x_train.columns),
    'Importance': list(lr.coef_.reshape(-1,1).flatten())  # Take the mean across all classes (multiclass case)
    }).sort_values(by='Importance' , ascending = False)


    print(feature_importance_df)

    print('')

    return y_test , y_test_pred , lr
    

    
# evaluate the trained model 
def model_evaluation(actial_target_data , predicted_target_data):
    
    eval_metrics = classification_report(actial_target_data , predicted_target_data,zero_division=0)

    print("Model Evaluation Metrics: \n  Classification Report : \n" , eval_metrics)

    print('===============================================================')

    

# save the model to disk
def save_model(model):
    
    with open('./Models/student_performance_model.pkl',  'wb') as f:
        pickle.dump(model, f)

    
    

# create the work flow
def main():
    file_path = "./Data/student_performance_data.csv"
    data = data_ingestion(file_path)
    preprocessed_data = data_preprocessing(data)
    actual_test_target_data , predicted_test_target_data , model = model_training(preprocessed_data)
    model_evaluation(actual_test_target_data , predicted_test_target_data)
    save_model(model)

if __name__ == "__main__":
    main()