����� ������ ������������ �������� X � ��������� ��������� ������� 
![equation](https://latex.codecogs.com/gif.latex?$&space;Y&space;=&space;\mathbb{R}&space;$). 
���������� ����������� ����������� ![equation](https://latex.codecogs.com/gif.latex?$y^*:X&space;\rightarrow&space;Y$), �������� ������� �������� ������ �� �������� ���������� ������� ![equation](https://latex.codecogs.com/gif.latex?$&space;X^l&space;=&space;(x_i\&amp;amp;amp;amp;amp;amp;space;,\&amp;amp;amp;amp;amp;amp;space;y_i)^l_{i=1},\&amp;amp;amp;amp;amp;amp;space;y_i&space;=&space;y^*(x_i)&space;$). ��������� ��������� �������� ![equation](https://latex.codecogs.com/gif.latex?$&space;a:\&amp;amp;amp;amp;amp;amp;space;X\rightarrow&space;Y&space;$ ) , ���������������� ����������� �����������  ![equation](https://latex.codecogs.com/gif.latex?$y^*$) . ��������������, ��� �� ��������� X ������ ������� ![equation](https://latex.codecogs.com/gif.latex?\rho(x,x^')).
� ������ ������� ������������ ������� �������-�������: 
![equation](https://latex.codecogs.com/gif.latex?a_h(x;X^l)&space;=&space;\frac{\sum_{i=1}^{l}&space;y_i\omega_i(x)}{\sum_{i=1}^{l}&space;\omega_i(x)}&space;=&space;\frac{\sum_{i=1}^{l}&space;y_iK\left(\frac{\rho(x,x_i)}{h}&space;\right&space;)}{\sum_{i=1}^{l}&space;K\left(\frac{\rho(x,x_i)}{h}&space;\right&space;)})

������, ������ ������ ������� ������������ � ��������. 
������ ����: ��������� ���� �� ������������  
![equation](https://latex.codecogs.com/gif.latex?$&space;\delta_t&space;=\bar{K}(\hat{\varepsilon_t})&space;$) , 
���  ![equation](https://latex.codecogs.com/gif.latex?$&space;\hat{\varepsilon_t}=&space;\|&space;\hat{y_t}&space;-&space;y_t&space;\|&space;$) . 
����� ������� ���������� �������� ���������� ������������ (Lowess): 
![equation](https://latex.codecogs.com/gif.latex?a(x_t;&space;X\setminus\{&space;x_t\})&space;=&space;\frac{&space;\sum_{i=1,&space;i\neq&space;t&space;}^{m}&space;{y_i&space;\delta_i&space;K\left(&space;\frac{\rho(x_i,x_t)}{h(x_t)}\right)}&space;}&space;{\sum_{i=1,&space;i\neq&space;t&space;}^{m}&space;{y_i&space;K\left(&space;\frac{\rho(x_i,x_t)}{h(x_t)}\right)}&space;}) 
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/regression/lowess/img/Nadaray_Lowess_difference.JPG)

�� ������ ������� �����, ��� lowess ���������� ����� (��� ��������� ������ ���� h = 0.5)

![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/regression/lowess/img/Lowess_1.JPG)