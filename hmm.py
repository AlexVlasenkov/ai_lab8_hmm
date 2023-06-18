import scipy.stats as st
import numpy as np


class HMM:
    # https://russianblogs.com/article/9516996848/
    def __init__(self, n_states):
        # получаем HMM и последовательность наблюдений на вход,
        # на выходе - вероятность последовательности наблюдений
        self.n_states = n_states
        self.random_state = np.random.RandomState(0)

        # нормализуем случайное начальное состояние
        self.prior = self._normalize(self.random_state.rand(self.n_states, 1))
        # А - матрица вероятности перехода между состояниями
        self.A = self._stochasticize(self.random_state.rand(self.n_states, self.n_states))

        self.mu = None
        self.covs = None
        self.n_dims = None

    # Алгоритм прямого и обратного направления - нахождение прямой вероятности первого состояния
    # или обратной вероятности последнего состояния

    # вычисляем вероятность наблюдаемой последовательности при заданных параметрах модели
    def _forward(self, B):
        log_likelihood = 0.
        T = B.shape[1]
        alpha = np.zeros(B.shape)
        for t in range(T):
            # На каждом шаге алгоритма мы вычисляем вероятность наблюдения текущего символа
            # при заданном скрытом состоянии и вероятность перехода из
            # предыдущего скрытого состояния в текущее скрытое состояние
            if t == 0:
                alpha[:, t] = B[:, t] * self.prior.ravel()
            else:
                alpha[:, t] = B[:, t] * np.dot(self.A.T, alpha[:, t - 1])

            # умножаем эти вероятности и суммируем по всем возможным предыдущим скрытым состояниям.
            # Этот процесс продолжается до тех пор,
            # пока мы не вычислим вероятность наблюдаемой последовательности символов
            alpha_sum = np.sum(alpha[:, t])
            alpha[:, t] /= alpha_sum
            log_likelihood = log_likelihood + np.log(alpha_sum)
        return log_likelihood, alpha

    # переход из одного состояния обратно

    # The forward-backward procedure позволяет оценить вероятность того,
    # что в данной модели HMM появится выборка наблюдаемых значений

    # Алгоритм нахождения обратной вероятности backward в HMM используется для вычисления вероятности наблюдения
    # последовательности символов, начиная с определенного состояния модели.
    # Это позволяет оценить, насколько вероятно, что модель находится в определенном состоянии,
    # при заданной наблюдаемой последовательности символов.
    # Другими словами, алгоритм backward позволяет оценить вероятность того,
    # что модель находится в определенном состоянии, при заданной наблюдаемой последовательности символов,
    # начиная с последнего символа и двигаясь в обратном направлении.
    # Эта вероятность может быть использована для принятия решения о том, какое слово было произнесено.
    def _backward(self, B):
        T = B.shape[1]
        beta = np.zeros(B.shape)

        beta[:, -1] = np.ones(B.shape[0])

        for t in range(T - 1)[::-1]:
            # рассчитываем обратную вероятность
            beta[:, t] = np.dot(self.A, (B[:, t + 1] * beta[:, t + 1]))
            beta[:, t] /= np.sum(beta[:, t])
        return beta

    # вероятностные состояния
    def _state_likelihood(self, obs):
        obs = np.atleast_2d(obs)
        # B - матрица вероятности наблюдения
        B = np.zeros((self.n_states, obs.shape[1]))
        for s in range(self.n_states):
            # scipy 0.14
            np.random.seed(self.random_state.randint(1))
            B[s, :] = st.multivariate_normal.pdf(
                obs.T, mean=self.mu[:, s].T, cov=self.covs[:, :, s].T)
        return B

    # при вычислении вероятности наблюдаемой последовательности символов при заданных параметрах модели,
    # вероятности могут становиться очень маленькими или очень большими,
    # что может привести к ошибкам округления и потере точности (при обработке K)
    def _normalize(self, x):
        return (x + (x == 0)) / np.sum(x)

    # В HMM мы предполагаем, что наблюдаемые данные порождаются скрытыми состояниями,
    # которые мы не можем наблюдать напрямую.
    # Стохастизация позволяет нам моделировать неопределенность в процессе генерации данных
    # и учитывать ее при обучении модели.
    # Кроме того, стохастизация позволяет нам использовать вероятностные методы для оценки параметров модели
    # и принятия решений на основе вероятностных распределений.
    # Это особенно полезно в задачах, связанных с распознаванием речи, где мы хотим определить,
    # какое слово было произнесено на основе наблюдаемой последовательности звуков
    def _stochasticize(self, x):
        return (x + (x == 0)) / np.sum(x, axis=1)

    def _em_init(self, obs):
        # Использование функции _em_init позволяет использовать меньше аргументов конструктора
        if self.n_dims is None:
            self.n_dims = obs.shape[0]
        if self.mu is None:
            subset = self.random_state.choice(np.arange(self.n_dims), size=self.n_states, replace=False)
            self.mu = obs[:, subset]
        if self.covs is None:
            self.covs = np.zeros((self.n_dims, self.n_dims, self.n_states))
            self.covs += np.diag(np.diag(np.cov(obs)))[:, :, None]
        return self

    # без учителя - Баума-Велша
    # алгоритм «предположений и максимизаций» для поиска максимальной
    # вероятностной оценки параметров скрытой модели Маркова при заданном наборе наблюдений
    def _em_step(self, obs):
        obs = np.atleast_2d(obs)
        B = self._state_likelihood(obs)
        T = obs.shape[1]

        # сначала получим прямую и обратную вероятность
        log_likelihood, alpha = self._forward(B)
        beta = self._backward(B)

        xi_sum = np.zeros((self.n_states, self.n_states))
        gamma = np.zeros((self.n_states, T))

        # вычислим вероятности
        for t in range(T - 1):
            partial_sum = self.A * np.dot(alpha[:, t], (beta[:, t] * B[:, t + 1]).T)
            xi_sum += self._normalize(partial_sum)
            partial_g = alpha[:, t] * beta[:, t]
            gamma[:, t] = self._normalize(partial_g)

        # Вместо этого модель ищет скрытые закономерности в данных,
        # чтобы выделить группы похожих объектов или выделить признаки,
        # которые могут быть полезны для решения задачи.
        # именно поэтому в начале мы используем кластеризацию,
        # чтобы разбить произнесенные буквы на группы на основе их сходства
        partial_g = alpha[:, -1] * beta[:, -1]
        gamma[:, -1] = self._normalize(partial_g)

        expected_prior = gamma[:, 0]
        expected_A = self._stochasticize(xi_sum)

        expected_mu = np.zeros((self.n_dims, self.n_states))
        expected_covs = np.zeros((self.n_dims, self.n_dims, self.n_states))

        gamma_state_sum = np.sum(gamma, axis=1)
        # Установим нули в 1 перед делением
        gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0)

        for s in range(self.n_states):
            gamma_obs = obs * gamma[s, :]
            expected_mu[:, s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]
            partial_covs = np.dot(gamma_obs, obs.T) / gamma_state_sum[s] - np.dot(expected_mu[:, s],
                                                                                  expected_mu[:, s].T)
            # Симметризуем
            partial_covs = np.triu(partial_covs) + np.triu(partial_covs).T - np.diag(partial_covs)

        # Обеспечим положительную полуопределенность, добавив диагональную нагрузку
        expected_covs += .01 * np.eye(self.n_dims)[:, :, None]

        self.prior = expected_prior
        self.mu = expected_mu
        self.covs = expected_covs
        self.A = expected_A
        return log_likelihood

    def fit(self, obs, n_iter=15):
        if len(obs.shape) == 2:
            for i in range(n_iter):
                self._em_init(obs)
                log_likelihood = self._em_step(obs)
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            for n in range(count):
                for i in range(n_iter):
                    self._em_init(obs[n, :, :])
                    log_likelihood = self._em_step(obs[n, :, :])
        return self

    def transform(self, obs):
        # Поддержка 2D и 3D массивов
        # 2D должно быть n_features, n_dims
        # 3D должно быть n_examples, n_features, n_dims
        # Например, при 6 функциях на речевой сегмент 105 разных слов
        # этот массив должен быть размером
        # (105, 6, X), где X — количество кадров с извлеченными функциями
        # Для одного файла примера размер массива должен быть (6, X)
        if len(obs.shape) == 2:
            B = self._state_likelihood(obs)
            log_likelihood, _ = self._forward(B)
            return log_likelihood
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            out = np.zeros((count,))
            for n in range(count):
                B = self._state_likelihood(obs[n, :, :])
                log_likelihood, _ = self._forward(B)
                out[n] = log_likelihood
            return out


if __name__ == "__main__":
    rstate = np.random.RandomState(0)
    t1 = np.ones((4, 40)) + .001 * rstate.rand(4, 40)
    t1 /= t1.sum(axis=0)
    t2 = rstate.rand(*t1.shape)
    t2 /= t2.sum(axis=0)

    m1 = HMM(2)
    m1.fit(t1)
    m2 = HMM(2)
    m2.fit(t2)

    m1t1 = m1.transform(t1)
    m2t1 = m2.transform(t1)
    print("Likelihoods for test set 1")
    print("M1:", m1t1)
    print("M2:", m2t1)
    print("Prediction for test set 1")
    print("Model", np.argmax([m1t1, m2t1]) + 1)
    print()

    m1t2 = m1.transform(t2)
    m2t2 = m2.transform(t2)
    print("Likelihoods for test set 2")
    print("M1:", m1t2)
    print("M2:", m2t2)
    print("Prediction for test set 2")
    print("Model", np.argmax([m1t2, m2t2]) + 1)
