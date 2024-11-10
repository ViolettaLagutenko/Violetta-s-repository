#Задание 1
# импорт пакетов
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
from scipy.stats import mannwhitneyu 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
plt.style.use('ggplot')
from matplotlib.pyplot import figure
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (24,16)
v = None
# чтение данных
df = pd.read_csv('train.csv')
df.info()
#проверим пропущенные значения
df.isnull().sum()
#отбор числовых колонок
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
print(numeric_cols)
# отбор нечисловых колонок 
# отбор нечислловых колонок для train
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
print(numeric_cols)
print(non_numeric_cols)
cols = df.columns
colors = ['#FFC0CB', '#008000'] 
a = sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colors))
for i, col in enumerate(df.columns):
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))
    if i>=10:
        break
# отбросить признак с большим количеством пропущенных данных cabin
df.drop(columns = 'Cabin', axis = 1, inplace = True)
print(df)
# заменить в возрасте пропуски на медианное значение 
med = df['Age'].median()
print(med)
df['Age'] = df['Age'].fillna(med)
# удалить строки в Embarked
df.dropna(inplace = True)
df.isnull().sum()
#PassengerId — идентификатор пассажира
#Survived — погиб (0) или выжил (1)
#Pclass — класс билета: первый (1), второй (2) или третий (3)
#Name — имя пассажира
#Sex — пол
#Age — возраст
#SibSp — количество братьев и сестер или супругов (siblings and spouses) на борту
#Parch — количество родителей и детей (parents and children) на борту
#Ticket — номер билета
#Fare — стоимость билета
#Embarked — порт посадки (C — Шербур; Q — Квинстаун; S — Саутгемптон)

#категориальные переменные 
# применим one-hot encoding к переменной Sex (пол) с помощью функции pd.get_dummies()
pd.get_dummies(df['Sex']).head(3)
# вновь скачаем датафрейм с единственным столбцом Sex
previous = pd.read_csv('train.csv')[['Sex']]
previous.head()
# закодируем переменную через 0 и 1
pd.get_dummies(previous['Sex'], dtype = int).head(3)
# удалим первый столбец, он избыточен
sex = pd.get_dummies(df['Sex'], drop_first = True)
sex.head(3)
# закодируем переменные чере 0 и 1 Pclass и Embarked.
embarked = pd.get_dummies(df['Embarked'], drop_first = True)
pclass = pd.get_dummies(df['Pclass'], drop_first = True)
df = pd.concat([df, pclass, sex, embarked], axis = 1)
df.head(3)
#отбор признаков
# применим метод .drop() к соответствующим столбцам
df.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Ticket', 'Embarked'], axis = 1, inplace = True)
df.head(3)
# нормализация данных
# импортируем класс StandardScaler
from sklearn.preprocessing import StandardScaler
# создадим объект этого класса
scaler = StandardScaler()
# выберем те столбцы, которые мы хотим масштабировать
cols_to_scale = ['Age', 'Fare']
# рассчитаем среднее арифметическое и СКО для масштабирования данных
scaler.fit(df[cols_to_scale])
# применим их
df[cols_to_scale] = scaler.transform(df[cols_to_scale])
# посмотрим на результат
df.head(3)
df.columns
#Название переменных 2 и 3 (второй и третий классы) выражены числами, а не строками (их выдает отсутствие кавычек в коде ниже). Так быть не должно.
#Преобразуем эти переменные в тип str через функцию map()
df.columns = df.columns.map(str)
df.columns
#разделим обущающую выборку на признаки и целевую переменную
# поместим в X_df все кроме столбца Survived
x_df = df.drop('Survived', axis = 1)
# столбец 'Survived' станет нашей целевой переменной (y_df)
y_df = df['Survived']
x_df.head(3)
# обучение модели логистической регрессии
# импортируем логистическую регрессию из модуля linear_model библиотеки sklearn
from sklearn.linear_model import LogisticRegression
 
# создадим объект этого класса и запишем его в переменную model
model = LogisticRegression()
 
# обучим нашу модель
model.fit(x_df, y_df)
# прогнозируем
y_pred_df = model.predict(x_df)
# построим матрицу ошибок
from sklearn.metrics import confusion_matrix
 
# передадим ей фактические и прогнозные значения
conf_matrix = confusion_matrix(y_df, y_pred_df)
 
# преобразуем в датафрейм
conf_matrix_df = pd.DataFrame(conf_matrix)
conf_matrix_df
# рассчитаем метрику accuracy вручную
round((480 + 237)/(480 + 237 + 69 + 103), 3)
#0,807 - На обучающей выборке наша модель показала результат в 80,7%.
#Задание 2

Статистика на отчищенных данных 
# импорт пакетов
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
from scipy import stats
from scipy.stats import mannwhitneyu 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
plt.style.use('ggplot')
from matplotlib.pyplot import figure
plt.style.use('ggplot')    # стиль графиков
from google.colab import drive
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (24,16)
v = None
# чтение данных
df = pd.read_csv('train.csv')
df.info()
#проверим пропущенные значения
df.isnull().sum()
#отбор числовых колонок
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
print(numeric_cols)
# отбор нечисловых колонок 
# отбор нечислловых колонок для train
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
print(numeric_cols)
print(non_numeric_cols)
cols = df.columns
colors = ['#FFC0CB', '#008000'] 
a = sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colors))
for i, col in enumerate(df.columns):
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))
    if i>=10:
        break
# отбросить признак с большим количеством пропущенных данных cabin
df.drop(columns = 'Cabin', axis = 1, inplace = True)
print(df)
# заменить в возрасте пропуски на медианное значение 
med = df['Age'].median()
print(med)
df['Age'] = df['Age'].fillna(med)
# удалить строки в Embarked
df.dropna(inplace = True)
df.isnull().sum()
#PassengerId — идентификатор пассажира
#Survived — погиб (0) или выжил (1)
#Pclass — класс билета: первый (1), второй (2) или третий (3)
#Name — имя пассажира
#Sex — пол
#Age — возраст
#SibSp — количество братьев и сестер или супругов (siblings and spouses) на борту
#Parch — количество родителей и детей (parents and children) на борту
#Ticket — номер билета
#Fare — стоимость билета
#Embarked — порт посадки (C — Шербур; Q — Квинстаун; S — Саутгемптон)
df.dtypes
# среднее значение возраста
mean_age = df['Age'].mean()
print(mean_age)
# 29 лет
# среднее значение количества братьев и сестер или супругов на борту
mean_SibSp = df['SibSp'].mean()
print(mean_SibSp)
# 0,5
# среднее значение стоимости билета
mean_Fare = df['Fare'].mean()
print(mean_Fare)
# 32,09
# описательные статистики
df.describe()
# Построим гистограмму по возрасту отдельно для мужчин и женщин
with sns.axes_style('whitegrid'):
    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x='Age', hue='Sex')
# связь пола и класса
pd.crosstab(df['Sex'], df['Pclass'], margins=True)
# каков максимальный возраст среди пассажиров определенного пола для каждого класса
pd.pivot_table(
    df, 
    values='Age', 
    index='Sex',
    columns='Pclass', 
    aggfunc=np.max
)
df.groupby('Sex')['PassengerId'].count()

