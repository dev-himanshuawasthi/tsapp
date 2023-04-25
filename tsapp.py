import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import lightgbm as lgb

# Load data from CSV file
data = pd.read_csv('CaseStudy_FraudIdentification.csv')

# Separate features and target variable
X = data.drop('default', axis=1)
y = data['default']
X = data.drop('default payment next month', axis=1)
y = data['default payment next month']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LightGBM classifier model
lgb_model = lgb.LGBMClassifier(learning_rate=0.25,max_depth=-5,random_state=42)

# Train model on training data
lgb_model.fit(X_train,y_train,eval_set=[(X_test,y_test),(X_train,y_train)],
          verbose=20,eval_metric='logloss')

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

# Run dashboard
if __name__ == '__main__':
    dashboard()
