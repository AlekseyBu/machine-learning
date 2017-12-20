Пусть задано пространство объектов X и множество возможных ответов 
![equation](https://latex.codecogs.com/gif.latex?$&space;Y&space;=&space;\mathbb{R}&space;$). 
Существует неизвестная зависимость ![equation](https://latex.codecogs.com/gif.latex?$y^*:X&space;\rightarrow&space;Y$), значения которой известны только на объектах обучающией выборки ![equation](https://latex.codecogs.com/gif.latex?$&space;X^l&space;=&space;(x_i\&amp;amp;amp;amp;amp;amp;space;,\&amp;amp;amp;amp;amp;amp;space;y_i)^l_{i=1},\&amp;amp;amp;amp;amp;amp;space;y_i&space;=&space;y^*(x_i)&space;$). Требуется построить алгоритм ![equation](https://latex.codecogs.com/gif.latex?$&space;a:\&amp;amp;amp;amp;amp;amp;space;X\rightarrow&space;Y&space;$ ) , аппроксимирующий неизвестную зависимость  ![equation](https://latex.codecogs.com/gif.latex?$y^*$) . Предполагается, что на множестве X задана метрика ![equation](https://latex.codecogs.com/gif.latex?\rho(x,x^')).
В первом подходе используется формула Надарая-Ватсона: 
![equation](https://latex.codecogs.com/gif.latex?a_h(x;X^l)&space;=&space;\frac{\sum_{i=1}^{l}&space;y_i\omega_i(x)}{\sum_{i=1}^{l}&space;\omega_i(x)}&space;=&space;\frac{\sum_{i=1}^{l}&space;y_iK\left(\frac{\rho(x,x_i)}{h}&space;\right&space;)}{\sum_{i=1}^{l}&space;K\left(\frac{\rho(x,x_i)}{h}&space;\right&space;)})

Однако, данный подход слишком чувствителен к выбросам. 
Отсюда идея: домножать веса на коэффициенты  
![equation](https://latex.codecogs.com/gif.latex?$&space;\delta_t&space;=\bar{K}(\hat{\varepsilon_t})&space;$) , 
где  ![equation](https://latex.codecogs.com/gif.latex?$&space;\hat{\varepsilon_t}=&space;\|&space;\hat{y_t}&space;-&space;y_t&space;\|&space;$) . 
Такой процесс называется локально взвешенным сглаживанием (Lowess): 
![equation](https://latex.codecogs.com/gif.latex?a(x_t;&space;X\setminus\{&space;x_t\})&space;=&space;\frac{&space;\sum_{i=1,&space;i\neq&space;t&space;}^{m}&space;{y_i&space;\delta_i&space;K\left(&space;\frac{\rho(x_i,x_t)}{h(x_t)}\right)}&space;}&space;{\sum_{i=1,&space;i\neq&space;t&space;}^{m}&space;{y_i&space;K\left(&space;\frac{\rho(x_i,x_t)}{h(x_t)}\right)}&space;}) 
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/regression/lowess/img/Nadaray_Lowess_difference.JPG)

На данном графике видно, что lowess сглаживает лучше (при параметре ширины окна h = 0.5)

![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/regression/lowess/img/Lowess_1.JPG)