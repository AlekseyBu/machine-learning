Пусть задано пространство объектов X и множество возможных ответов $ Y = \mathbb{R} $. Существует неизвестная зависимость $y^*:X \rightarrow Y$, значения которой известны только на объектах обучающией выборки $ X^l = (x_i\ ,\ y_i)^l_{i=1},\  y_i = y^*(x_i) $. Требуется построить алгоритм $ a:\ X\rightarrow Y $, аппроксимирующий неизвестную зависимость $y^*$. Предполагается, что на множестве X задана метрика.
В первом подходе используется формула Надарая-Ватсона: 
$a_h(x;X^l) = \frac{\sum_{i=1}^{l} y_i\omega_i(x)}{\sum_{i=1}^{l} \omega_i(x)} = \frac{\sum_{i=1}^{l} y_iK\left(\frac{\rho(x,x_i)}{h} \right )}{\sum_{i=1}^{l} K\left(\frac{\rho(x,x_i)}{h} \right )}$
Однако, данный подход слишком чувствителен к выбросам. Отсюда идея: домножать веса на коэффициенты  $ \delta_t =\bar{K}(\hat{\varepsilon_t}) $, где  $ \hat{\varepsilon_t}= \| \hat{y_t} - y_t \| $. Такой процесс называется локально взвешенным сглаживанием (Lowess): 
$ a(x_t; X\setminus\{ x_t\}) = \frac{ \sum_{i=1, i\neq t }^{m} {y_i \delta_i K\left( \frac{\rho(x_i,x_t)}{h(x_t)}\right)} } {\sum_{i=1, i\neq t }^{m} {y_i K\left( \frac{\rho(x_i,x_t)}{h(x_t)}\right)} } $

![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/regression/lowess/img/Nadaray_Lowess_difference.JPG)

На данном графике видно, что lowess сглаживает лучше (при параметре ширины окна h = 0.5)

![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/regression/lowess/img/Lowess_1.JPG)