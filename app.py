import streamlit as st
import pickle
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os

# Load dataset
iris = load_iris()
X, Y = iris.data, iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

MODEL_PATH = 'model.pkl'

# Train and save model if not exists
if not os.path.exists(MODEL_PATH):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, Y_train)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

# Load model from file
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


def prediction(sepal_length, sepal_width, petal_length, petal_width):
    lst = [[sepal_length, sepal_width, petal_length, petal_width]]
    p = iris.target_names[model.predict(lst)]
    return 'Iris ' + p[0]


def main():
    st.title('Iris Flowers', anchor='center')
    st.image('Photo.jpg', width=705, caption='Iris Flowers Images')

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
        result = prediction(float(sepal_length), float(sepal_width), float(petal_length), float(petal_width))

    st.header('The Type Of Flower is {}'.format(result.title()))


main()
