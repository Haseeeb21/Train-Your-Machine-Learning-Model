import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib
import pickle
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


# Training classification model
def train_CL_model(data, target_col, test_size, classifier_name):
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


# Training regression model
def train_RG_model(data, target_col, test_size, regressor_name):
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if regressor_name == 'Linear Regression':
        model = LinearRegression()
    elif regressor_name == 'Ridge Regression':
        model = Ridge()
    elif regressor_name == 'Lasso Regression':
        model = Lasso()
    elif regressor_name == 'Elastic Net':
        model = ElasticNet()
    elif regressor_name == 'Decision Tree':
        model = DecisionTreeRegressor()
    elif regressor_name == 'Random Forest':
        model = RandomForestRegressor()
    elif regressor_name == 'Gradient Boosting':
        model = GradientBoostingRegressor()
    elif regressor_name == 'SVM':
        model = SVR()
    elif regressor_name == 'KNN':
        model = KNeighborsRegressor()
    elif regressor_name == 'MLP':
        model = MLPRegressor()
    else:
        raise ValueError("Regressor not supported")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2


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
            menu_title="Menu",
            options=["Home", "Classification", "Regression"],
            icons=["house", "🧐", "😶‍🌫️"],
            menu_icon=[],
            default_index=0
        )

    # Clear session state when switching tabs
    if 'last_selected' not in st.session_state or st.session_state.last_selected != selected:
        st.session_state.last_selected = selected
        if selected == "Classification":
            st.session_state['clf_trained'] = False
        elif selected == "Regression":
            st.session_state['reg_trained'] = False

    if selected == "Home":
        home()

    if selected == "Classification":
        classification()

    if selected == "Regression":
        regression()


def home():
    st.title("Welcome")

    st.header("Train and Save Machine Learning Model", divider="gray")

    st.markdown(
        """
        Train and Save your Machine Learning Model.

        ### Classification

        To train Classification Models select Classification tab from the sidebar.

        ### Regression

        To train Regression Models select Regression tab from the sidebar.

        ### Note

        You can upload `.csv` or `.xlsx` files. The columns / features in the files should be encoded i.e. should be integers.

        The target column should also be encoded in integer values for **`Classification`** and float / int values for **`Regression`**.
        """
    )


def classification():
    st.title("Classification Machine Learning Model Trainer")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], key='clf_file_uploader')

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Dataframe:")
        display_dataframe(data)

        target_col = st.selectbox("Select Target Column", data.columns, key='clf_target_col')
        test_size = st.slider("Select Test Size", 0.05, 0.80, 0.2, key='clf_test_size')

        classifier_name = st.selectbox("Select Classifier",
                                       ["Logistic Regression", "Random Forest", "SVM", "KNN", "Decision Tree", "SGD",
                                        "MLP"], key='clf_classifier_name')

        if st.button("Train Model", key='clf_train_button'):
            model, accuracy = train_CL_model(data, target_col, test_size, classifier_name)
            st.session_state['clf_model'] = model
            st.session_state['clf_accuracy'] = accuracy
            st.session_state['clf_trained'] = True

    if st.session_state.get('clf_trained', False):
        st.write(f"Model Accuracy: {st.session_state['clf_accuracy'] * 100:.2f}%")

        model_filename = st.text_input("Enter model filename", classifier_name, key='clf_model_filename')
        model_extension = st.selectbox("Select model extension", ["Joblib", "Pickle"], key='clf_model_extension')

        if st.button("Save Model", key='clf_save_button'):
            file_extension = '.joblib' if model_extension == 'Joblib' else '.pkl'
            full_model_filename = f"{model_filename}{file_extension}"
            save_model_to_file(st.session_state['clf_model'], full_model_filename, model_extension)
            with open(full_model_filename, 'rb') as f:
                st.download_button(label="Download Model", data=f, file_name=full_model_filename,
                                   mime='application/octet-stream')


def regression():
    st.title("Regression Machine Learning Model Trainer")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], key='reg_file_uploader')

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Dataframe:")
        display_dataframe(data)

        target_col = st.selectbox("Select Target Column", data.columns, key='reg_target_col')
        test_size = st.slider("Select Test Size", 0.05, 0.80, 0.2, key='reg_test_size')

        regressor_name = st.selectbox("Select Regressor",
                                      ["Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net",
                                       "Decision Tree", "Random Forest", "Gradient Boosting", "SVM", "KNN", "MLP"],
                                      key='reg_regressor_name')

        if st.button("Train Model", key='reg_train_button'):
            model, mse, r2 = train_RG_model(data, target_col, test_size, regressor_name)
            st.session_state['reg_model'] = model
            st.session_state['reg_mse'] = mse
            st.session_state['reg_r2'] = r2
            st.session_state['reg_trained'] = True

    if st.session_state.get('reg_trained', False):
        st.write(f"Mean Squared Error: {st.session_state['reg_mse']}")
        st.write(f"R-squared: {st.session_state['reg_r2']:.2f}")

        model_filename = st.text_input("Enter model filename", regressor_name, key='reg_model_filename')
        model_extension = st.selectbox("Select model extension", ["Joblib", "Pickle"], key='reg_model_extension')

        if st.button("Save Model", key='reg_save_button'):
            file_extension = '.joblib' if model_extension == 'Joblib' else '.pkl'
            full_model_filename = f"{model_filename}{file_extension}"
            save_model_to_file(st.session_state['reg_model'], full_model_filename, model_extension)
            with open(full_model_filename, 'rb') as f:
                st.download_button(label="Download Model", data=f, file_name=full_model_filename,
                                   mime='application/octet-stream')


if __name__ == "__main__":
    main()
