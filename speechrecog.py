import numpy as np
import matplotlib.pyplot as plt
import os

f_paths = []
labels = []
spoken = []
for f in os.listdir('audio'):
    for let in os.listdir('audio/' + f):
        f_paths.append('audio/' + f + '/' + let)
        labels.append(f)
        if f not in spoken:
            spoken.append(f)
print('Массив лексем:', spoken)
from scipy.io import wavfile

data = np.zeros((len(f_paths), 32000))
maxsize = -1
for n, file in enumerate(f_paths):
    # бежим по файлам и собираем частоты дискретизации
    # в выборках/сек
    # обычно *.wav хранит несжатое аудио, поэтому здесь мы получаем
    # LPCM-данные, т.е. линейной импульсно-кодовой модуляции
    _, d = wavfile.read(file)
    data[n, :d.shape[0]] = d
    if d.shape[0] > maxsize:
        maxsize = d.shape[0]
# матрица data представляет собой нечто схожее с матрицей DTW из 7-й ЛР,
# Мы ищем такую оптимальную последовательность пар сегментов, которой бы соответствовала
# минимальная суммарная оценка различия.
data = data[:, :maxsize]
print('Всего файлов с аудио:', data.shape[0])
# массив меток (отсчетов) заполняем нулями
all_labels = np.zeros(data.shape[0])
x = 0
y = 15
for j in range(5):

    for i in range(15):
        all_labels[x:y] = j

    x += 15
    y += 15

# Алгоритм К-средних пытается классифицировать данные
# без предварительного обучения на размеченных данных
# https://proglib.io/p/obyasnite-tak-kak-budto-mne-10-let-prostoe-opisanie-populyarnogo-algoritma-klasterizacii-k-srednih-2022-12-07
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data)
    # WCSS (within-cluster sum of squares) –
    # cумму квадратов внутрикластерных расстояний до центра кластера
    # для определения качества кластеров - отклонения от центров
    wcss.append(kmeans.inertia_)
# финальными кластерами становятся кластеры с наименьшим WCSS
plt.plot(range(1, 11), wcss)
plt.title('Правило локтя')
plt.xlabel('Число кластеров')
plt.ylabel('WCSS')
plt.show()
# K-means используется для инициализации параметров HMM - матрица
# переходов и матрица эмиссий. Мы используем K-means для кластеризации
# наблюдаемых значений и назначения каждому кластеру скрытого состояния.
kmeans = KMeans(init='k-means++', n_clusters=5, n_init=20, max_iter=1000)
kmeans.fit(data)
y_kmeans = kmeans.predict(data)
print(y_kmeans)

from numpy.lib.stride_tricks import as_strided
import scipy


# short-time Fourier transform - кратковременное преобразование Фурье
# С помощью STFT можно определить амплитуду различных частот,
# воспроизводимых в данный момент времени аудиосигнала
# В HMM для распознавания речи, мы можем использовать STFT для извлечения спектральных признаков из аудио-сигналов.
# Мы можем разбить аудио-сигнал на короткие фрагменты, называемые кадрами, и для каждого кадра вычислить STFT.
# Затем мы можем использовать полученные спектры как признаки для обучения HMM.
def stft(x, fftsize=64, overlap_pct=.5):
    hop = int(fftsize * (1 - overlap_pct))
    w = scipy.hanning(fftsize + 1)[:-1]
    # происходит "оцифровка" сигнала, т.е.
    # перневод LPCM-сигнала в последовательность цифр
    # извлекаем спектральные признаки из аудио-сигналов
    raw = np.array([np.fft.rfft(w * x[i:i + fftsize]) for i in range(0, len(x) - fftsize, hop)])
    return raw[:, :(fftsize // 2)]


def peak_find(x, n_peaks, l_size=3, r_size=3, c_size=3, f=np.mean):
    win_size = l_size + r_size + c_size
    shape = x.shape[:-1] + (x.shape[-1] - win_size + 1, win_size)
    strides = x.strides + (x.strides[-1],)
    xs = as_strided(x, shape=shape, strides=strides)

    # Идентификация пиков: Гауссовы пики могут использоваться для идентификации пиков в сигнале.
    def is_peak(x):
        centered = (np.argmax(x) == l_size + int(c_size / 2))
        l = x[:l_size]
        c = x[l_size:l_size + c_size]
        r = x[-r_size:]
        passes = np.max(c) > np.max([f(l), f(r)])
        if centered and passes:
            return np.max(c)
        else:
            return -1

    r = np.apply_along_axis(is_peak, 1, xs)
    top = np.argsort(r, None)[::-1]
    heights = r[top[:n_peaks]]
    # добираемся до фактического местоположения пика.
    top[top > -1] = top[top > -1] + l_size + int(c_size / 2.)
    return heights, top[:n_peaks]


all_obs = []
for i in range(data.shape[0]):
    # загружаем дату в Фурье
    d = np.abs(stft(data[i, :]))
    n_dim = 6
    obs = np.zeros((n_dim, d.shape[0]))
    for r in range(d.shape[0]):
        # производим расчет наблюдений - обучение
        _, t = peak_find(d[r, :], n_peaks=n_dim)
        obs[:, r] = t.copy()
    if i % 10 == 0:
        print("Обучение %s" % i)
    all_obs.append(obs)

all_obs = np.atleast_3d(all_obs)

from sklearn.model_selection import StratifiedShuffleSplit

# Этот объект перекрестной проверки представляет собой слияние StratifiedKFold и
# HMM может быть обучена на данных, которые разбиты на обучающую и тестовую выборки.
# Например, если у нас есть данные для распознавания речи, которые содержат записи разных людей,
# и мы хотим обучить HMM на этих данных, то мы можем использовать StratifiedShuffleSplit для разбиения данных
# на обучающую и тестовую выборки таким образом,
# чтобы пропорции записей каждого человека были сохранены в каждой выборке. У нас - директории с буквами и словами
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
# получим кол-во итераций разделения в кросс-валидаторе
sss.get_n_splits(all_obs, all_labels)
for n, i in enumerate(all_obs):
    # Складки производятся путем сохранения процентного соотношения образцов для каждого класса
    all_obs[n] /= all_obs[n].sum(axis=0)

for train_index, test_index in sss.split(all_obs, all_labels):
    # получаем данные для работы и разделяем эти данные на обучающее и тестовое подмножества
    X_train, X_test = all_obs[train_index, ...], all_obs[test_index, ...]
    y_train, y_test = all_labels[train_index], all_labels[test_index]
# В случае обработки сигналов, трехмерная матрица может быть интерпретирована как набор двумерных матриц,
# где каждая матрица представляет собой временной ряд или спектральные данные для одного сигнала.
# Например, если мы имеем трехмерную матрицу размером (100, 5000, 2), то это означает,
# что у нас есть 100 аудио-сигналов, каждый из которых представлен матрицей размером 5000x2,
# где первый столбец соответствует левому каналу, а второй столбец - правому каналу
print('Размеры training-матрицы:', X_train.shape)
print('Размеры testing-матрицы:', X_test.shape)

from hmm import HMM

ys = set(all_labels)
ms = [HMM(4) for y in ys]
# группируем поэлементно в кортежи
_ = [m.fit(X_train[y_train == y, :, :]) for m, y in zip(ms, ys)]
ps = [m.transform(X_test) for m in ms]
res = np.vstack(ps)
predicted_labels = np.argmax(res, axis=0)
missed = (predicted_labels != y_test)
print('Точность теста: %.2f процентов' % (100 * (1 - np.mean(missed))))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predicted_labels)
print(cm)

from sklearn.metrics import classification_report

# далее происходит вывод метрик, используемых для оценки качества классификации в машинном обучении
#  precision (точность) - это доля правильно классифицированных объектов положительного класса
#  относительно всех объектов, которые были отнесены к этому классу
#  recall (полнота) - это доля правильно классифицированных объектов положительного класса
#  относительно всех объектов этого класса в тестовой выборке
#  f1-score - среднее между recall и presision, что является мерой баланса
#  support - это количество объектов в каждом классе в тестовой выборке
print(classification_report(y_test, predicted_labels, labels=predicted_labels,
                            target_names=['a', 'apple', 'b', 'banana', 'k', 'zero']))
