# S1-CP-Proy1
Regresion Lineal fundamentos

## Proyecto general
El cuadernillo Proyecto.ipynb tiene una descripcion detallada de los pasos sugeridos por el proyecto.

## Regresion por gradiente en descenso
La libreria *custom_regression.py* tiene la clase *linear_regression*.

La clase *linear_regression* recibe los siguientes parametros:
* **X**: Matriz de variables independientes, array numpy de dimension $m\times n$, donde $m$ es el numero de filas y $n$ es la dimension de la matriz.
* **y**: Matriz de variables dependientes, array numpy de dimension $m\times k$ donde $m$ es el numero de filas y $k$ es la dimension de las variables dependientes.
* **fit_intercept**: Flag que indica si se debe calcular un intercepto. Si es True, se agregara internamente a las variables dependientes una columna de unos. Si es False, no se agrega dicha columna. El valor por defecto es True.
* **n_epochs**: Numero de corridas que efectuara el gradiente en descenso sobre todas las observaciones. Los metodos se encuentran vectorizados por lo que una epoch corresponde a una operacion vectorial de calculo de gradiente, prediccion y error. Valor por defeto es 2000.
* **alpha**: Step size de cada actualizacion efectuada por el algoritmo gradiente en descenso. Valor por defecto es 0.01
* **print_every_iter**: Indica cuantas cada cuantas iteraciones presentara en pantalla el error de la iteracion. Valor por defecto 100.


La clase *linear_regression* cuenta con los siguientes atributos:

* *X_*: Atributo interno de la matriz de variables dependientes. Si el objeto es inicializado con fit_intercept=True se agrega una columna de unos.
* *y_*: Atributo interno de la matriz de variables independientes.
* *coef_*: Matriz $A$, de dimension $n\times k$, correspondiente a la matriz del modelo lineal $y=XA$
* *y_pred*: Aplicacion del modelo lineal a a X_ usando coef_


La clase *linear_regression* cuenta con los metodos:

* **fit()**: Aplica gradiente en descenso para encontrar los coeficientes de la matriz $A$ de un modelo lineal de la forma $y=XA$. Internamente, actualiza los atributos coef_ y y_pred_
* **predict(X)**: Aplica el modelo lineal a un conjunto de observaciones $X$ utilizando el valor del atributo coef_
* **sqr_err()**: Calcula el error correspondiente al valor del atributo coef_.
