import matplotlib         # ВИДЕО № 14
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выболок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
#  %matplotlib inline ------ в обычном питоне не использовать - а только в ноутбуке юпител и другом

# project
# загрузка выборок и обучающей и тестовой
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация (нормализация)  входных данных
x_train = x_train / 255
x_test = x_test / 255

# преобразование выходных значений в векторы по категориям
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# в керас на входе каждого сверточного слоя ожидается 4-х мерный тензор х_ это коллекции (тренировки и тестовой)  ахис - это какую ось нам нужно добавитьб т.е добавили 4-ю ось
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

print( x_train.shape )

# отображение первых 25 изображений
#plt.figure(figsize=(10,5))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.imshow(x_train[i], cmap=plt.cm.binary)
#plt.show()

# формирование модели НС и вывод её структуры в консоль
model = keras.Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),         # функция активации первого скрытого слоя нейронов РЕЛУ
    Dense(10, activation='softmax')         # функция активации выходного слоя нейронов СофтМакс
])
print(model.summary())      # вывод структцры НС в консоль

# компиляция НС с оптимизацией по Adam и критериям - категориальная кросс-энтропия
model.compile(optimizer='adam',                 # оптимизация по Адам
              loss='categorical_crossentropy',  # loss - функция потерь, выбрали эту ф.потерь (критерий качества)
              metrics=['accuracy'])
                                                # т.к. у нас задача классификаци
                                                # да и выходные нейроны имеют функцию активации СофтМакс
                                                # метрика - процент ошибок, т.е. accuracy - это точность (на тест.выборке 0.9984 - это почти 100%)

# запуск процесса обучения: 80% - обучающая выборка, 20% - выборка валидации
#model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2) # ЗДЕСЬ ДРУГАЯ ОБУЧАЛКА
his = model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
                                    # batch_size - после каждых 32 изображений
                                    # мы будем корректировать весовые коэфициенты
                                    # эпох у нас 5
                                    # validation_split - делает разбивку обучающей выборки на обучающую и проверочную
                                    # у нас показатель 80/20

# подаём на вход тестовую выборку
model.evaluate(x_test, y_test_cat)

##################
# вывод графика - переменная history и класс библиотеки .history['loss']
#plt.plot(his.history['loss'])    # loss  - eto oshibka
#plt.grid(True)
#plt.show()

# визуализация процесса обучения нейро-сети ========== DRUGOY ROLOK
print(his.history.keys())     # - составление словаря обучения
print(his.history['accuracy'])    #  - печатаем значение аккуратности на обучающем наборе данных
print(his.history['val_accuracy']) # - печатаем значение аккуратности на тестовом проверочном наборе данных
# + VIZUALIZATION
plt.plot(his.history['accuracy'], label='Akkuratnost na obuchajuschem nabore')
plt.plot(his.history['val_accuracy'], label='Akkuratnost na proverochnom nabore')
plt.xlabel('Epoh obuchenia')
plt.ylabel('Akkuratnost')
plt.legend()
plt.show()

plt.plot(his.history['loss'], label='Oshibka na obuchajuschem nabore')
plt.plot(his.history['val_loss'], label='Oshibka na proverochnom nabore')
plt.xlabel('Epoh obuchenia')
plt.ylabel('Oshibka')
plt.legend()
plt.show()
#################

# проверка распознавания цифр ---- ЭТО МЫ ПРОВЕРЯЕМ ВРУЧНУЮ, а n - это картинка в массиве картинок
n = 727
x = np.expand_dims(x_test[n], axis=0) # создаем трехмерный тензор, т.к. у нас картинка и мы просто так ее на вход подать не можем
res = model.predict(x)
print( res )
print( f"Распознанная цифра: {np.argmax(res)}" )

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

#============
# распознавание всей тестовой выборки
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

print(pred.shape)

print(pred[:20])
print(y_test[:20])

# выделение всех неверных (нераспознанных) вариантов во всей выборке
mask = pred == y_test
print(mask[:10])

x_false = x_test[~mask]
p_false = pred[~mask]

# вывод всех нераспознаных изображений
print(x_false.shape)

# вывод первых 5-ти неверных результатов
for i in range(5):
    print("Значение сети: " + str(p_false[i]))
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()





