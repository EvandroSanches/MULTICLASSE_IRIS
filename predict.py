import numpy as np
from keras.models import load_model

def predict(previsores):
    classes = ['Iris-setosa','Iris-versicolor','Iris-virginica']
    modelo = load_model('Modelo.0.1')
    resultado = modelo.predict(previsores)
    resultado = classes[np.argmax(resultado)]
    return resultado

# Informar Array ou csv com os seguintes atribustos
# sepal_length, sepal_width, petal_length, petal_width
#Exemplo

dados = np.array([[6.5,3.2,5.2,2.3]])
result = predict(dados)
print(result)

