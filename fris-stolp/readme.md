**Алгоритм FRiS-СТОЛП (FRiS-STOLP)** - алгоритм отбора эталонных объектов для метрического классификатора на основе FRiS-функции.
Одной из основных проблем, возникающих при решении задачи классификации каких-либо объектов,
является проблема выбора меры схожести. 
Чаще всего в этой роли выступает расстояние,
однако в некоторых задачах удается добиться более достоверного результата,
используя более специфичные меры. 
**FRiS-функция** — мера схожести двух объектов относительно некоторого третьего объекта. 
В отличие от классических мер расстояния, эта функция позволяет не просто сказать, похожи объекты друг на друга или нет, но и уточнить ответ на вопрос «по сравнению с чем?».
Такой подход позволяет учитывать большее число факторов при классификации.


######Пример работы алгоритма **FRiS-STOLP**:
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/iris_05_01.png)
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/iris_07_01.png)
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/iris_09_01.png)
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/wine_05_01.png)
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/wine_07_01.png)
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/wine_09_01.png)
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/wine_05_02.png)
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/wine_07_02.png)
![](https://raw.githubusercontent.com/elvinayakubova/machine-learning/master/fris-stolp/wine_09_02.png)

Алгоритм FRiS-STOLP создаёт в процессе работы сокращенное описание обучающей выборки. Это позволяет сократить расход памяти, избавиться от ошибок и выбросов, содержащихся в ней, но при этом сохранить информацию, необходимую для дальнейшего распознавания новых объектов.


