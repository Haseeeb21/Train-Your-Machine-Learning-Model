import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import pickle
import os

from streamlit_option_menu import option_menu

# Loading CSV or Excel file
def load_data(file):
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        data = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format")
    return data

# Displaying dataframe
def display_dataframe(data):
    st.write(data)

# Training model
def train_model(data, target_col, test_size, classifier_name):
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    if classifier_name == 'Logistic Regression':
        model = LogisticRegression()
    elif classifier_name == 'Random Forest':
        model = RandomForestClassifier()
    elif classifier_name == 'SVM':
        model = SVC()
    elif classifier_name == 'KNN':
        model = KNeighborsClassifier()
    elif classifier_name == 'Decision Tree':
        model = DecisionTreeClassifier()
    elif classifier_name == 'SGD':
        model = SGDClassifier()
    elif classifier_name == 'MLP':
        model = MLPClassifier()
    else:
        raise ValueError("Classifier not supported")
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

# Save model to file
def save_model_to_file(model, model_filename, model_extension):
    if model_extension == "Joblib":
        joblib.dump(model, model_filename)
    else:
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)


def main():
    
    with st.sidebar:
        selected = option_menu(
            menu_title = "Menu",
            options = ["Home", "Classification", "Regression"],
            icons = ["house", "🧐", "😶‍🌫️"],
            menu_icon = [],
            default_index = 0            
        )
    
    if selected == "Home":
        home()
    
    if selected == "Classification":
        classification()
    
    if selected == "Regression":
        regression()
    

def home():
    st.title("Wellcome")
    
    st.header("Train and Save Machine Learning Model", divider="gray")

    st.markdown(
        """
        Train and Save your Machine Learning Model.
        
        ### Classification
        
        To train Classification Models select Classification tab from the sidebar.
        
        
        ### Regression
        
        To train Regression Models select Classification tab from the sidebar.
        
        
        ### Note that
        
        You can upload `.csv` or `.xlsx` files. The columns / features in the files should be encoded i.e. should be integers.
        
        The target column should also be encoded in integer values for **`Classification`** and float values for **`Regression`**.
        
        """
        
    )




# Streamlit UI
def classification():
    st.title("Machine Learning Model Trainer")
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Dataframe:")
        display_dataframe(data)
        
        # default_target_col = 'target'  # Replace with your default column name
        
        target_col = st.selectbox("Select Target Column", data.columns, key='target_col')
        test_size = st.slider("Select Test Size", 0.05, 0.80, 0.2, key='test_size')
        
        classifier_name = st.selectbox("Select Classifier", ["Logistic Regression", "Random Forest", "SVM", "KNN", "Decision Tree", "SGD", "MLP"], key='classifier_name')
        
        if st.button("Train Model"):
            model, accuracy = train_model(data, target_col, test_size, classifier_name)
            st.session_state['model'] = model  # Store the trained model in session state
            st.session_state['accuracy'] = accuracy  # Store the accuracy in session state
            st.session_state['trained'] = True

    if 'trained' in st.session_state:
        st.write(f"Model Accuracy: {st.session_state['accuracy'] * 100:.2f}%")
        
        model_filename = st.text_input("Enter model filename", classifier_name, key='model_filename')
        model_extension = st.selectbox("Select model extension", ["Joblib", "Pickle"], key='model_extension')
        
        if st.button("Save Model"):
            file_extension = '.joblib' if model_extension == 'Joblib' else '.pkl'
            full_model_filename = f"{model_filename}{file_extension}"
            save_model_to_file(st.session_state['model'], full_model_filename, model_extension)
            with open(full_model_filename, 'rb') as f:
                st.download_button(label="Download Model", data=f, file_name=full_model_filename, mime='application/octet-stream')


def regression():
    pass


if __name__ == "__main__":
    main()
