import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# Load data
cleaned_df = pd.read_csv('./Artifacts/online_retail_data_cleaned.csv')
churn_df = pd.read_csv('./Artifacts/online_retail_data_churn.csv')

# Preprocessing
cleaned_df['InvoiceDateTime'] = pd.to_datetime(cleaned_df['InvoiceDate'])
cleaned_df['Amount'] = cleaned_df['Quantity'] * cleaned_df['UnitPrice']

# Sidebar
st.sidebar.title("User Selection")

# Get unique customers
users = cleaned_df['CustomerID'].unique()
selected_user = st.sidebar.selectbox("Select a user", users)

# Filter data for selected user
user_df = cleaned_df[cleaned_df['CustomerID'] == selected_user]
user_featured_df = churn_df[churn_df['CustomerID'] == selected_user][['unique_invoiceno_count','unique_stockcode_count','total_amount','active_days']].reset_index().drop(columns='index',axis=1)

with open('./Artifacts/customer_churn_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('./Artifacts/customer_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Step 5: Predict using the model

scaled_data = scaler.transform(user_featured_df)
predictions = model.predict(scaled_data)
churn_score = model.predict_proba(scaled_data)

user_df['year_month'] = user_df['InvoiceDateTime'].dt.to_period('M').astype(str)

# Group by month and sum amount
monthly_totals = user_df.groupby('year_month')['Amount'].sum().reset_index()

# Create plot with markers and Y-value labels

fig = px.line(
    monthly_totals,
    x='year_month',
    y='Amount',
    # title='Monthly Total Amount',
    labels={'year_month': 'Month', 'Amount': 'Total Amount'},
    markers=True  # Show dots at each data point
)

# Add Y-values as labels
fig.update_traces(text=monthly_totals['Amount'].round(2), textposition='top center')

# Rotate x-axis labels for readability
fig.update_layout(xaxis_tickangle=-45)

# Main content
st.title("User Analytics Dashboard")
st.subheader(f"Analytics Report for User ID: {selected_user}")

st.markdown("###### Monthly Total Amount : ")
# Show plot in Streamlit
st.plotly_chart(fig, use_container_width=True)



# **Horizontal Bar Chart: Group by StockCode**
# Group by StockCode and calculate the total amount and count
stockcode_data = user_df.groupby('StockCode').agg({
    'Amount': 'sum',
    'Quantity': 'sum',
    'StockCode': 'count'  # This will give you the count of rows per StockCode
}).rename(columns={'StockCode': 'Count','Amount':'Total_Amount','Qunantity':'Total_Quantity'}).reset_index().sort_values('Total_Amount', ascending=False)


st.markdown("###### Stock Code Info : ")
st.dataframe(stockcode_data) 


st.markdown("###### User Optimized Info : ")

st.dataframe(user_featured_df) 






st.subheader("User Churn Status")
st.info(f'Predicted Churn Score : {churn_score[0][1]}')




if churn_score[0][1] >= 0.95 :
    st.error('High Risk of Churn')
elif churn_score[0][1] >= 0.75 :
    st.warning('At Risk of Churn')
else:
    st.success('Not Likely to Churn')

