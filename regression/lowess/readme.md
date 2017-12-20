����� ������ ������������ �������� X � ��������� ��������� ������� $ Y = \mathbb{R} $. ���������� ����������� ����������� $y^*:X \rightarrow Y$, �������� ������� �������� ������ �� �������� ���������� ������� $ X^l = (x_i\ ,\ y_i)^l_{i=1},\  y_i = y^*(x_i) $. ��������� ��������� �������� $ a:\ X\rightarrow Y $, ���������������� ����������� ����������� $y^*$. ��������������, ��� �� ��������� X ������ �������.
� ������ ������� ������������ ������� �������-�������: 
$a_h(x;X^l) = \frac{\sum_{i=1}^{l} y_i\omega_i(x)}{\sum_{i=1}^{l} \omega_i(x)} = \frac{\sum_{i=1}^{l} y_iK\left(\frac{\rho(x,x_i)}{h} \right )}{\sum_{i=1}^{l} K\left(\frac{\rho(x,x_i)}{h} \right )}$
������, ������ ������ ������� ������������ � ��������. ������ ����: ��������� ���� �� ������������  $ \delta_t =\bar{K}(\hat{\varepsilon_t}) $, ���  $ \hat{\varepsilon_t}= \| \hat{y_t} - y_t \| $. ����� ������� ���������� �������� ���������� ������������ (Lowess): 
$ a(x_t; X\setminus\{ x_t\}) = \frac{ \sum_{i=1, i\neq t }^{m} {y_i \delta_i K\left( \frac{\rho(x_i,x_t)}{h(x_t)}\right)} } {\sum_{i=1, i\neq t }^{m} {y_i K\left( \frac{\rho(x_i,x_t)}{h(x_t)}\right)} } $

![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/regression/lowess/img/Nadaray_Lowess_difference.JPG)

�� ������ ������� �����, ��� lowess ���������� ����� (��� ��������� ������ ���� h = 0.5)

![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/regression/lowess/img/Lowess_1.JPG)