# The traveling salesman problem using the ACO – Ant Colony Optimization algorithm

## Description of the problem

This optimization algorithm is based on the behavior of the ant colony, which allows the insects to find the most efficient route.

The essence of the algorithm is quite simple: initially, ants deliver food to the anthill randomly, each in its own way, thus covering various routes. When moving, ants leave a stinking trail of pheromones along their path, thus marking their routes. Therefore, the shorter the path, the more times the ants will have time to cross it at the same time and they will be able to leave more pheromones on it. Pheromones evaporate over time. For this reason, there are fewer pheromones on long routes than on short ones, even if the same number of ants are moving along them. In this way, ants are more likely to choose the "smelliest path" and forget about paths with fewer pheromones.

<p align="center">
  <img src="https://github.com/Irrisorr/Travelling-salesman-problem-with-ACO/blob/main/img/first.png">
</p>

## Conditions to be met
  - **The creation of ants.**

Even at this stage, there is an initial placement of a small amount of pheromone, so that at the first stage the probability of moving to an adjacent vertex is not zero.

  - **Decision Search.**

Formula to calculate the probability of an ant moving from vertex i to j:
<p align="center">
  <img src="https://github.com/Irrisorr/Travelling-salesman-problem-with-ACO/blob/main/img/second.jpg">
</p>
Where:

τ*ij*(t) − amount of pheromone between vertices i and j

η*ij* − distance between these vertices

α, β − are constant parameters

The closer the β parameter is to zero, the less ants will be guided by the distance between vertices when choosing a path and will focus only on the pheromone. As β increases, the value of closeness increases. The α parameter works in the same way, but for the pheromone level.

The upper part of the formula describes the ant's willingness to move from peak i to vertex j. It is proportional to the proximity of the peak and the level of pheromone on the way to it.

Therefore, the probability of going from vertex i to vertex j is equal to the desire to go there divided by the sum of the desire to go from vertex i to all available vertices that have not yet been visited. The sum of all probabilities is 1.

By expanding all the probabilities on the real line from 0 to 1, you can generate a random real number in that interval. The result will show which vertex the ant will go to.

  - Pheromones update

The formula for recalculating the pheromone level in each iteration of the algorithm:
<p align="center">
  <img src="https://github.com/Irrisorr/Travelling-salesman-problem-with-ACO/blob/main/img/third.png">
</p>
Where:

ρ - evaporation rate

t - iteration number

L*k*(t) - price of the current solution for the kth ant

Q - a parameter that has a value on the order of the price of the optimal solution

Q/L*k*(t) - pheromone deposited by the kth ant using edges (*i*, *j*)

The complexity of the ant colony algorithm depends on the number of vertices, the number of ants, and the lifetime of the colony.

## Code
### Libraries

```
import numpy as np
from scipy import spatial
import time
import matplotlib.pyplot as plt
import pandas as pd
```
### Graph generation

First, the coordinates of the points are generated, then the distance matrix (distance from each vertex to each vertex) is calculated using the spatial module from the scipy library.

```
wierzszcholki = 20 # liczba wierzchołków
wierz_сoord = np.random.rand(wierzszcholki, 2)  # losowe generowanie wierzchołków
print("Współrzędne wierzchołków:\n", wierz_сoord[:10], "\n")
 
# obliczanie macierzy odległości między wierzchołkami
distance_mat = spatial.distance.cdist(wierz_сoord, wierz_сoord, metric='euclidean')
#print("Macierz odległości:\n", distance_mat)
```

### Program output

<p align="left">
  <img src="https://github.com/Irrisorr/Travelling-salesman-problem-with-ACO/blob/main/img/four.png">
</p>

### Algorithm

```
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
        self.gen_best_x, self.gen_best_y = [], [] # naprawianie najlepszych generacji
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
```

### Function that calculates the distance

```
# obliczanie długości ścieżki
def all_dist(routine):
    wierzcholki, = routine.shape
    return sum([distance_mat[routine[i % wierzcholki], routine[(i + 1) % wierzcholki]] for i in range(wierzcholki)])
```

 ### Graph output

```
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
print("time: %s seconds" %abs (time.time() - start_time))
plt.show()
```

## Examples

  - With the number of ants and iterations 20 and 10, respectively, vertices = 20, we have:

<p align="right">
  <img src="https://github.com/Irrisorr/Travelling-salesman-problem-with-ACO/blob/main/img/five.png", 
align="left">
  <img src="https://github.com/Irrisorr/Travelling-salesman-problem-with-ACO/blob/main/img/six.png">
</p>

  - With the same number of ants and iterations, but already vertices = 1000, we have:

<p align="right">
  <img src="https://github.com/Irrisorr/Travelling-salesman-problem-with-ACO/blob/main/img/seven.png", 
align="left">
  <img src="https://github.com/Irrisorr/Travelling-salesman-problem-with-ACO/blob/main/img/eight.png">
</p>

## Conclusion

As you can see, the number of possible paths grows very quickly as the number of cities increases, and a simple enumeration of options (brute force method) significantly reduces the possibility of solving the problem.

The traveling salesman problem is one of the transcomputation problems. This means that in quite a small number of cities (66 or more), the best solution using a simple brute force search (the formula for the number of options for a symmetric problem - (n-1)!/2.) cannot be found by any of the most powerful computers of the time. shorter than billions of years. Therefore, it makes much more sense to find a solution to this problem using optimization algorithms (for example, the ant colony algorithm that I showed).
