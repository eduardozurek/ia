# Redes Neuronales Artificiales con ScikitLearn

**Eduardo Zurek, Ph.D.**  
Profesor Titular — ezurek@uninorte.edu.co  
Departamento de Ingeniería de Sistemas  
Universidad del Norte

> Versión en Markdown de la presentación «00_0_ANN_Introducción_Sklearn». Se conserva el orden de sus 13 diapositivas, con las figuras originales y las fórmulas en LaTeX.

---

## 1. Redes Neuronales Artificiales con ScikitLearn

Clase Modelo.

---

## 2. ¿Qué es una Red Neuronal Artificial?

Es una técnica biomimética (inspirada en sistemas biológicos) de aprendizaje computacional. Las neuronas artificiales se inspiran fundamentalmente en las neuronas multipolares.

---

## 3. ¿Cómo funciona una Neurona biológica?

Las neuronas multipolares comúnmente reciben señales a través de las dendritas y propagan señales a través del axón y posteriormente a través de las terminales axonales que hacen sinapsis con las dendritas de otras neuronas

La forma más común de transmitir información a través del espacio que existe entre una terminal axonal y una dendrita es mediante la difusión de unas sustancias químicas llamadas neurotransmisores, este fenómeno se conoce como sinapsis química

---

## 4. Neurotransmisores y nivel de activación

Los neurotransmisores abren compuertas moleculares en las dendritas que permiten el flujo de partículas cargadas eléctricamente llamadas iones.

Los iones crean una diferencia de voltaje entre el interior del cuerpo de la neurona y el medio ambiente externo, esa diferencia de voltaje se llama nivel de activación o potencial de membrana de la neurona

---

## 5. Nivel de activación y potencial de acción

Las señales que llegan a través de las dendritas se combinan dentro de la neurona

En muchos modelos computacionales se supone que la combinación es una suma ponderada, pero también es posible describir la relación usando funciones lógicas booleanas

Una neurona propaga una señal eléctrica a través de su axón cuando el nivel de activación en su membrana es mayor que un umbral (propio de cada tipo de neurona)

La descarga eléctrica de la neurona se llama potencial de acción

---

## 6. Neurona Multipolar

Fuente: scientificstock.com

![Neurona multipolar](imagenes/neurona-multipolar.png)

---

## 7. Neurona Artificial

![Neurona artificial: entradas, pesos, suma, sesgo, activación y salida](imagenes/perceptron.png)

---

## 8. Descenso del Gradiente

Tenemos un conjunto de datos, y suponemos que ese conjunto de datos es explicado por una hipótesis.

![Conjunto de puntos y recta que representa la hipótesis lineal](imagenes/puntos-y-linea.png)

Para este caso proponemos la hipótesis:

$$
h(x) = w_1 x + w_0
$$

Esta hipótesis implica un error respecto al valor verdadero que se está estimando. Por conveniencia matemática usamos el error cuadrático:

$$
e^2 = (y-h)^2
$$

$$
\frac{d e^2}{d h} = -2(y-h) = 2(h-y)
$$

Esto nos muestra la relación del error y $h$.

![Error cuadrático y dirección de descenso hacia el mínimo](imagenes/error-cuadratico.png)

---

## 9. Red Neuronal Artificial Multicapa

![Red neuronal multicapa con entrada, capas ocultas y salida](imagenes/red-neuronal-multicapa.png)

La capa de entrada no tiene neuronas; $E_1$ y $E_2$ son para indicar. En todas las neuronas va implícita una entrada de sesgo. El peso de un arco que conecte una neurona $i$ con una neurona $j$ se denota por $w_{j,i}$.

---

## 10. Implementación en Scikit-Learn

Para el desarrollo de los ejemplos usaremos Python y Scikit-Learn.

- [Python](https://www.python.org): lenguaje usado en los ejemplos.
- [Scikit-Learn](https://scikit-learn.org): herramientas para análisis de datos y aprendizaje automático.
- [Jupyter](https://jupyter.org): cuadernos para presentar los ejemplos.

Existen recursos en la nube que facilitan el desarrollo con Python, Scikit-Learn y Jupyter:

- [Google Colab](https://colab.research.google.com), que se usará en esta presentación.
- [Microsoft Azure Notebooks](https://notebooks.azure.com).
- [Proyecto Jupyter: Try Jupyter](https://jupyter.org/try).

*Se conservan los enlaces de la presentación original.*

---

## 11. Scikit-Learn - Ejemplo

Usaremos un **Perceptrón Multicapa**.

- [Redes neuronales supervisadas](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).
- [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).

Para usar correctamente la función debemos entender los elementos importantes.

### Entradas y salidas

`X`, `y`.

### Parámetros

- Número de elementos en las capas ocultas: `hidden_layer_sizes`.
- Función de activación: `activation`.

### Etapas

1. Configuración: `clf = MLPClassifier(...)`.
2. Entrenamiento: `clf.fit(X, y)`.
3. Predicción: `clf.predict` / `clf.predict_proba`.

---

## 12. Scikit-Learn - Ejemplo

En [este ejemplo](https://github.com/eduardozurek/ia/blob/main/_008_redes_neuronales_artificiales/00_1_RNA_SKLearn_EZ.ipynb) entrenaremos una Red Neuronal Artificial que aprenda de los datos que siguen una distribución como la siguiente

![Distribución de los datos del ejemplo de clasificación](imagenes/estatura-peso.png)

---

## 13. Scikit-Learn - Otros ejemplos

- [Reconocimiento de dígitos escritos](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html).
- [Clasificación de lirios (plantas)](https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html).
