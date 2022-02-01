import joblib
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
iris = load_iris()


def create_model():
    global iris
    X, Y = iris.data, iris.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, Y_train)
    joblib.dump(model, 'model.joblib')


def prediction(sepal_length, sepal_width, petal_length, petal_width):
    classifier = joblib.load('model.joblib')
    lst = [[sepal_length, sepal_width, petal_length, petal_width]]
    p = iris.target_names[classifier.predict(lst)]
    return 'Iris ' + p[0]


def main():
    st.title('Iris Flower Prediction', anchor='center')
    st.image('Photo.jpg', width=600, caption='Iris Flowers Images')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sepal_length = st.text_input('Sepal Length', '0.0')
    with col2:
        sepal_width = st.text_input('Sepal Width', '0.0')
    with col3:
        petal_length = st.text_input('Petal Length', '0.0')
    with col4:
        petal_width = st.text_input('Petal Width', '0.0')
    result = ''
    if st.button('Predict'):
        result = prediction(sepal_length, sepal_width, petal_length, petal_width)
    st.header('The Type Of Flower is {}'.format(result.title()))


main()

