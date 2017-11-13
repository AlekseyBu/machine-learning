# <center> FRiS-СТОЛП </center>

**Алгоритм FRiS-СТОЛП (FRiS-STOLP)** - алгоритм отбора эталонных объектов для метрического классификатора на основе FRiS-функции.
Одной из основных проблем, возникающих при решении задачи классификации каких-либо объектов, является проблема выбора меры схожести. Чаще всего в этой роли выступает расстояние, однако в некоторых задачах удается добиться более достоверного результата, используя более специфичные меры. 

**FRiS-функция** — мера схожести двух объектов относительно некоторого третьего объекта. В отличие от классических мер расстояния, эта функция позволяет не просто сказать, похожи объекты друг на друга или нет, но и уточнить ответ на вопрос «по сравнению с чем?». Такой подход позволяет учитывать большее число факторов при классификации.

На рисунке ниже приведён пример случая, когда FRiS функция, как мера сходства, работает лучше, чем обычная метрика:

![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/img/fris-func_example.png)

Объекты «красные квадраты» и «синие круги» образуют два класса. Рассматриваемый объект «желтый шестиугольник» располагается ближе к классу «синих кругов», но судя по структуре классов он является типичным представителем «красных квадратов». В большинстве подобных случаев функция конкурентного сходства будет работать корректно.


###### Пример работы алгоритма **FRiS-STOLP**:
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/img/iris_05_01.png)
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/img/iris_07_01.png)
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/img/iris_quality.png)

Качество классификации на тестовых данных (желтые точки) при полной выборке = 0.93, при эталонах = 0.80

![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/img/iris_09_01.png)
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/img/wine_05_01.png)
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/img/wine_07_01.png)

![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/img/wine_quality_1.png)

Качество классификации на тестовых данных (желтые точки) при полной выборке = 0.77, при эталонах = 0.70

![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/img/wine_05_02.png)
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/img/wine_07_02.png)

![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/img/wine_quality_2.png)

Качество классификации на тестовых данных (желтые точки) при полной выборке = 0.99, при эталонах = 0.60


Алгоритм FRiS-STOLP создаёт в процессе работы сокращенное описание обучающей выборки. Это позволяет сократить расход памяти, избавиться от ошибок и выбросов, содержащихся в ней, но при этом сохранить информацию, необходимую для дальнейшего распознавания новых объектов.



### Ссылки

1. Алгоритм FRiS-STOLP http://www.machinelearning.ru/wiki/index.php?title=%D0%90%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_FRiS-%D0%A1%D0%A2%D0%9E%D0%9B%D0%9F
2. FRiS-функция  http://www.machinelearning.ru/wiki/index.php?title=FRiS-%D1%84%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D1%8F