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

    # Алгоритм прямого и обратного направления - это не что иное, как нахождение прямой вероятности первого состояния
    # или обратной вероятности последнего состояния, а затем рекурсивно назад или вперед

    # Для данной модели найдите вероятность данной последовательности наблюдений длины T.
    # Идея метода прямого вычисления состоит в том, чтобы перечислить все последовательности
    # состояний длины T и вычислить совместную вероятность последовательности состояний и последовательности наблюдения
    # (от передачи скрытого состояния к наблюдению), суммируем все элементы перечисления.
    # Когда тип состояния равен N, всего существует N ^ T перестановок и комбинаций.
    # Объем вычислений для каждой комбинации для вычисления совместной вероятности равен T, а общая сложность - O(N^T)

    # В определении прямой вероятности определены два условия:
    # одно - это текущая последовательность наблюдений,
    # а другое - текущее состояние.
    # Следовательно, вычисление начального значения также имеет два элемента (наблюдение и состояние),
    # один - это вероятность начального состояния,
    # а другой - вероятность перехода к текущему наблюдению.
    def _forward(self, B):
        log_likelihood = 0.
        T = B.shape[1]
        alpha = np.zeros(B.shape)
        for t in range(T):
            if t == 0:
                alpha[:, t] = B[:, t] * self.prior.ravel()
            else:
                alpha[:, t] = B[:, t] * np.dot(self.A.T, alpha[:, t - 1])

            # Поскольку в момент времени T имеется в общей сложности N состояний,
            # которые запустили последнее наблюдение,
            # окончательный результат состоит в сложении этих вероятностей
            alpha_sum = np.sum(alpha[:, t])
            alpha[:, t] /= alpha_sum
            log_likelihood = log_likelihood + np.log(alpha_sum)
        return log_likelihood, alpha

    # переход из одного состояния обратно

    # The forward-backward procedure позволяет оценить вероятность того,
    # что в данной модели HMM появится выборка наблюдаемых значений
    def _backward(self, B):
        T = B.shape[1]
        beta = np.zeros(B.shape)

        beta[:, -1] = np.ones(B.shape[0])

        # Согласно определению, часть последовательности наблюдений от T + 1 до T на самом деле не существует,
        # поэтому жестко оговаривается, что это значение равно 1.
        # Идем от T-1, T-2, ... , 1
        for t in range(T - 1)[::-1]:
            # рассчитываем обратную вероятность
            beta[:, t] = np.dot(self.A, (B[:, t + 1] * beta[:, t + 1]))
            # Окончательное суммирование происходит потому, что существует N видов обратных вероятностей,
            # которые могут вывести последовательность наблюдений от 2 до T в первый момент времени,
            # поэтому сумма умножается на вероятность выхода O1 и получается окончательный результат.
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
            # This function can (and will!) return values >> 1
            # See the discussion here for the equivalent matlab function
            # https://groups.google.com/forum/#!topic/comp.soft-sys.matlab/YksWK0T74Ak
            # Key line: "Probabilities have to be less than 1,
            # Densities can be anything, even infinite (at individual points)."
            # This is evaluating the density at individual points...
        return B

    def _normalize(self, x):
        return (x + (x == 0)) / np.sum(x)

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
        # Поддержка 2D и 3D массивов
        # 2D должно быть n_features, n_dims
        # 3D должно быть n_examples, n_features, n_dims
        # Например, при 6 функциях на речевой сегмент 105 разных слов
        # этот массив должен быть размером
        # (105, 6, X), где X — количество кадров с извлеченными функциями
        # Для одного файла примера размер массива должен быть (6, X)
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
