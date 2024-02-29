import numpy as np
from scipy import spatial
import time
import matplotlib.pyplot as plt
import pandas as pd


wierzszcholki = 200  # liczba wierzchołków
wierz_сoord = np.random.rand(wierzszcholki, 2)  # losowe generowanie wierzchołków
print("Współrzędne wierzchołków:\n", wierz_сoord[:10], "\n")

# obliczanie macierzy odległości między wierzchołkami
distance_mat = spatial.distance.cdist(wierz_сoord, wierz_сoord, metric='euclidean')
# print("Macierz odległości:\n", distance_mat)

class ACO:  # klasa algorytmu kolonii mrówek do rozwiązywania problemu komiwojażera
    def __init__(self, func, ilosc, population=10, iteration=20, distance_mat=None, alpha=1, beta=2, rho=0.1):
        self.func = func
        self.ilosc = ilosc  # liczba miast
        self.population = population  # liczba mrówek
        self.iteration = iteration  # liczba iteracji
        self.alpha = alpha  # współczynnik ważności feromonów w wyborze ścieżki
        self.beta = beta  # współczynnik istotności odległości
        self.rho = rho  # szybkość parowania feromonów

        self.prob_mat_dist = 1 / (distance_mat + 1e-10 * np.eye(ilosc, ilosc))

        # Matryca feromonów aktualizowana w każdej iteracji
        self.Tau = np.ones((ilosc, ilosc))

        # Ścieżka każdej mrówki w określonym pokoleniu
        self.Tab = np.zeros((population, ilosc)).astype(int)
        self.y = None  # Całkowita odległość przebyta przez mrówkę w danym pokoleniu
        self.gen_best_x, self.gen_best_y = [], []  # naprawianie najlepszych generacji
        self.x_best_h, self.y_best_h = self.gen_best_x, self.gen_best_y
        self.best_x, self.best_y = None, None

    def algorytm(self, max_iter=None):
        self.iteration = max_iter or self.iteration
        for i in range(self.iteration):

            # prawdopodobieństwo przejścia bez normalizacji
            prob_mat = (self.Tau ** self.alpha) * (self.prob_mat_dist) ** self.beta
            for j in range(self.population):  # dla każdej mrówki

                # punkt początkowy ścieżki (może być dowolny, nie ma to znaczenia)
                self.Tab[j, 0] = 0
                for k in range(self.ilosc - 1):  # każdy wierzchołek, przez który przechodzą mrówki

                    # punkt, który został przekroczony i nie może być przekroczony ponownie
                    taboo = set(self.Tab[j, :k + 1])

                    # lista dozwolonych wierzchołków do wyboru
                    dost_lista = list(set(range(self.ilosc)) - taboo)
                    prob = prob_mat[self.Tab[j, k], dost_lista]
                    prob = prob / prob.sum()  # normalizacja prawdopodobieństwa
                    next_w = np.random.choice(dost_lista, size=1, p=prob)[0]
                    self.Tab[j, k + 1] = next_w

            # obliczanie odległości
            y = np.array([self.func(i) for i in self.Tab])

            # ustalić najlepsze rozwiązanie
            i_best = y.argmin()
            x_best, y_best = self.Tab[i_best, :].copy(), y[i_best].copy()
            self.gen_best_x.append(x_best)
            self.gen_best_y.append(y_best)

            # licząc feromon, który zostanie dodany do krawędzi
            delta_tau = np.zeros((self.ilosc, self.ilosc))
            for j in range(self.population):  # dla każdej mrówki
                for k in range(self.ilosc - 1):  # dla każdego wierzchołka

                    # mrówki przemieszczają się z wierzchołka n1 do wierzchołka n2
                    n1, n2 = self.Tab[j, k], self.Tab[j, k + 1]
                    delta_tau[n1, n2] += 1 / y[j]  # aplikacja feromonów

                # mrówki czołgają się z ostatniego szczytu z powrotem na pierwszy
                n1, n2 = self.Tab[j, self.ilosc - 1], self.Tab[j, 0]
                delta_tau[n1, n2] += 1 / y[j]  # aplikacja feromonów

            self.Tau = (1 - self.rho) * self.Tau + delta_tau

        best_gen = np.array(self.gen_best_y).argmin()
        self.best_x = self.gen_best_x[best_gen]
        self.best_y = self.gen_best_y[best_gen]
        return self.best_x, self.best_y

    fit = algorytm

    # obliczanie długości ścieżki
def all_dist(routine):
        wierzcholki, = routine.shape
        return sum([distance_mat[routine[i % wierzcholki], routine[(i + 1) % wierzcholki]] for i in range(wierzcholki)])

start_time = time.time()

# tworzenie obiektu algorytmu kolonii mrówek
aca = ACO(func=all_dist, ilosc=wierzszcholki,
        population=20,  # liczba mrówek
        iteration=10, distance_mat=distance_mat)
best_x, best_y = aca.algorytm()

# Wyniki na ekran
fig, ax = plt.subplots(1, 2)
best_p = np.concatenate([best_x, [best_x[0]]])
best_p_coord = wierz_сoord[best_p, :]
for i in range(0, len(best_p)):
    ax[0].annotate(best_p[i], (best_p_coord[i, 0], best_p_coord[i, 1]))
ax[0].plot(best_p_coord[:, 0],
            best_p_coord[:, 1], 'o-r')
pd.DataFrame(aca.y_best_h).cummin().plot(ax=ax[1])

plt.rcParams['figure.figsize'] = [20, 10]
print("time: %s seconds" % abs(time.time() - start_time))
plt.show()