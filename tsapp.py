import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import lightgbm as lgb
import altair as alt

# Load data from CSV file
df = pd.read_csv('CaseStudy_FraudIdentification.csv')


# Convert categorical variables to numeric using one-hot encoding
data = pd.get_dummies(df, columns=['EDUCATION', 'MARRIAGE','Gender'], prefix=['edu', 'mar','gen'])

# Separate features and target variable
X = data.drop('default payment next month', axis=1)
y = data['default payment next month']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LightGBM classifier model
lgb_model = lgb.LGBMClassifier(learning_rate=0.25,max_depth=-5,random_state=42)

# Train model on training data
lgb_model.fit(X_train,y_train,eval_set=[(X_test,y_test),(X_train,y_train)],eval_metric='logloss')

# Define function to display dashboard
def dashboard():
    # Display header
    st.write('# Credit Default Prediction Dashboard')
    
    # Display data summary
    st.write('## Data Summary')
    st.write(data.describe())
    
    # Display feature importance
    st.write('## Feature Importance')
    feature_importance = pd.Series(lgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(feature_importance)
    
    # Display model performance metrics
    st.write('## Model Performance')
    y_pred = lgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    st.write('Accuracy:', accuracy)
    st.write('Precision:', precision)
    st.write('Recall:', recall)

        # Display prediction form
    st.write('## Prediction')
    form = st.form(key='prediction_form')
    limit_bal = form.number_input('Limit Balance', min_value=0, max_value=1000000, step=1000)
    age = form.number_input('Age', min_value=18, max_value=100, step=1)
    pay_1 = form.selectbox('Payment Status (September)', options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    pay_2 = form.selectbox('Payment Status (August)', options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    pay_3 = form.selectbox('Payment Status (July)', options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    bill_amt1 = form.number_input('Bill Amount (September)', min_value=0, max_value=1000000, step=1000)
    bill_amt2 = form.number_input('Bill Amount (August)', min_value=0, max_value=1000000, step=1000)
    bill_amt3 = form.number_input('Bill Amount (July)', min_value=0, max_value=1000000, step=1000)
    form_submit = form.form_submit_button('Predict')

    # Execute prediction if form is submitted
    if form_submit:
        input_data = pd.DataFrame({
            'LIMIT_BAL': [limit_bal],
            'AGE': [age],
            'PAY_1': [pay_1],
            'PAY_2': [pay_2],
            'PAY_3': [pay_3],
            'BILL_AMT1': [bill_amt1],
            'BILL_AMT2': [bill_amt2],
            'BILL_AMT3': [bill_amt3]
        })

        prediction = lgb_model.predict(input_data)[0]

        if prediction == 0:
            st.write('Prediction: No Default')
        else:
            st.write('Prediction: Default')

# Run dashboard
if __name__ == '__main__':
    dashboard()
