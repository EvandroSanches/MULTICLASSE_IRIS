import keras.optimizers.schedules
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

dados = pd.read_csv('iris.csv')

previsores = dados.drop('class', axis=1)
target = dados['class']

#Tratando dados da variavel target
labelencoder = LabelEncoder()
target = labelencoder.fit_transform(target)
target_dummy = to_categorical(target)




def CriaModelo():
    modelo = Sequential()

    modelo.add(Dense(units=4, activation='leaky_relu', input_dim=4))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(units=3, activation='softmax'))

    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.024,
        decay_steps=2800,
        decay_rate=0.018
    )

    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_scheduler, clipvalue=0.5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return modelo

def avalia_treinamento():
    modelo = KerasClassifier(build_fn=CriaModelo, epochs=400, batch_size=10)

    resultados = cross_val_score(estimator=modelo, X=previsores, y=target_dummy, cv=10, scoring='accuracy')
    print(resultados)
    plt.bar(range(0,10),resultados)
    plt.xlabel('Épocas')
    plt.ylabel('Taxa de Acerto')
    plt.title('Histórico de Treinamento\n'+'Média de Acertos:'+str(resultados.mean())+'\nDesvio Padrão:'+str(resultados.std()))
    plt.show()

def gera_modelo():
    modelo = CriaModelo()
    resultado = modelo.fit(previsores, target_dummy, epochs=400, batch_size=10)
    modelo.save('Modelo.0.1')

gera_modelo()
