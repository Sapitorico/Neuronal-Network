import tensorflow as tf
import numpy as np

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheint = np.array([-40, 14, 32, 46.4, 59, 71.6, 100.4], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='mean_squared_error'
)

print("comienzo de entrenameinto...")
historial = modelo.fit(celsius, fahrenheint, epochs=10000, verbose=False)
print("modelo entrenado!")

print("predicciones ")
resultado = modelo.predict([7567])
print("Resultado  " + str(resultado) + " fahrenheit!")

print("variables internas del modelo")
print(capa.get_weights())
