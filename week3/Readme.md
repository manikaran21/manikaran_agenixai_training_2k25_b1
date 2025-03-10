# Final Report Of Projects

## Customer Churn

  ### Insights from Descriptive Analytics

  - All numerical and categorical independently variables are uniformly distributed aganist target varibale churn status .


 ### Insights from Predictive Analytics
  
  - After observing the multiple features aganist target variable.
  
  ```
    Feature Importance : Churn Status
                        Feature  Importance
  1          AverageCallDuration    0.040305
  11    PaymentMethod_CreditCard    0.037433
  0                          Age    0.024067
  5               Location_Rural    0.023636
  8               PlanType_Basic    0.021307
  6            Location_Suburban   -0.007937
  4               MonthlyCharges   -0.009986
  7               Location_Urban   -0.016067
  9             PlanType_Premium   -0.021307
  3                NumberOfCalls   -0.033533
  10  PaymentMethod_BankTransfer   -0.037433
  2                    DataUsage   -0.061258

  Model Evaluation Metrics: 

  Classification Report :

                precision    recall  f1-score   support

            0       0.78      1.00      0.88       196
            1       0.00      0.00      0.00        54
      accuracy                           0.78       250
    macro avg       0.39      0.50      0.44       250 
  weighted avg       0.61      0.78      0.69       250
```

### Conclusion :

- From above for each attribute we have some importance .
- Positive importance(coefficient) means that value of proportion increases the churn .
- Negative importance(coefficient) means that value of proportion decreses the churn .
    

## Student Performance Status

  ### Insights from Descriptive Analytics

  - All numerical and categorical independently variables are uniformly distributed aganist target varibale except grade attribute.
  - If grade is greater than equal to 60 then pass else fail .
  


  ### Insights from Predictive Analytics
  
  - After observing the multiple features aganist target variable.


  ### Example Code Block for Descriptive Analytics:
  ```
   Feature Importance : Student Performance Status
                       Feature  Importance
1                       Grades    6.969695
2                   Attendance    0.095626
11   ClassParticipation_Medium    0.072332
6     SocioeconomicStatus_High    0.068290
8   SocioeconomicStatus_Middle    0.029787
4                Gender_Female    0.024684
3          TimeSpentOnHomework    0.013428
10      ClassParticipation_Low   -0.001144
5                  Gender_Male   -0.024684
9      ClassParticipation_High   -0.077975
7      SocioeconomicStatus_Low   -0.093049
0                          Age   -0.181354

Model Evaluation Metrics:
  Classification Report :
               precision    recall  f1-score   support

           0       0.98      0.88      0.92        49
           1       0.97      1.00      0.98       201

    accuracy                           0.97       250
   macro avg       0.97      0.94      0.95       250
weighted avg       0.97      0.97      0.97       250

```

### Conclusion :
  
  - From above for each attribute we have some importance .
  - Positive importance(coefficient) means that value of proportion increases the pass percentage .
  - Negative importance(coefficient) means that value of proportion increases the fail percentage .


## Energy Consumption

  ### Insights from Descriptive Analytics

  - All numerical and categorical independently variables are uniformly distributed aganist target varibale except building size.
  - If building size increases energy consumption increases .


  ### Insights from Predictive Analytics
  
  - After observing the multiple features aganist target variable.
  
  ### Example Code Block for Descriptive Analytics:
  ```
    Feature Importance : Energy Consumption
                        Feature  Importance
0                  BuildingSize  139.303216
1                   BuildingAge   13.058449
2                   Temperature   11.107966
3                      Humidity    6.731978
8           InsulationType_Poor    3.447937
4              SupplierLeadTime    2.510865
11   RenewableEnergySource_Wind    2.210202
6           InsulationType_Fair    1.017869
7           InsulationType_Good    0.430256
10  RenewableEnergySource_Solar   -0.798790
9    RenewableEnergySource_None   -1.308856
5      InsulationType_Excellent   -5.154407
7           InsulationType_Good    0.430256
10  RenewableEnergySource_Solar   -0.798790
9    RenewableEnergySource_None   -1.308856
7           InsulationType_Good    0.430256
10  RenewableEnergySource_Solar   -0.798790
7           InsulationType_Good    0.430256
7           InsulationType_Good    0.430256
10  RenewableEnergySource_Solar   -0.798790
7           InsulationType_Good    0.430256
10  RenewableEnergySource_Solar   -0.798790
9    RenewableEnergySource_None   -1.308856
7           InsulationType_Good    0.430256
10  RenewableEnergySource_Solar   -0.798790
10  RenewableEnergySource_Solar   -0.798790
9    RenewableEnergySource_None   -1.308856
9    RenewableEnergySource_None   -1.308856
5      InsulationType_Excellent   -5.154407
Model Evaluation Mertrics:
R2 Score :
0.7093593964386382
```

### Conclusion :
  
  - From above for each attribute we have some importance .
  - Positive importance(coefficient) means that value of proportion increases in Energy Consumption .
  - Negative importance(coefficient) means that value of proportion decreases the Energy Consumption .