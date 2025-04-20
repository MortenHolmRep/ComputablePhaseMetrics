from enum import Enum
from math import exp, inf, log, sqrt
from random import choice, random, randrange


class Ising2D:
    Tc: float = 2 / log(1 + sqrt(2))

    def __init__(self, size: int, T: float, initial_state: list[int] | None = None):
        self.L: int = size
        self.L2: int = size**2
        self.beta: float = 1 / T if T > 0 else inf
        self.prob: float = 1 - exp(-2 * self.beta)
        self.state: list[int] = (
            self.initial_state() if initial_state is None else initial_state
        )
        self.neighbours: dict[int, tuple[int, int, int, int]] = self.get_neighbours()

    def metropolis(self, sweeps: int = 1) -> None:
        for _ in range(sweeps * self.L2):
            idx: int = randrange(self.L2)
            dE: int = 2 * self.state[idx] * self.neighbour_sum(idx)
            if random() < exp(-dE * self.beta):
                self.state[idx] *= -1

    def wolff(self) -> None:
        rand_idx: int = randrange(self.L2)
        cluster: list[int] = [rand_idx]
        queue: list[int] = [rand_idx]
        while queue:
            idx: int = choice(queue)
            queue.remove(idx)
            # ** grow cluster **
            for nbr in self.neighbours[idx]:
                if (
                    self.state[idx] == self.state[nbr]  # <parallel-condition>
                    and nbr not in cluster
                    and random() < self.prob
                ):
                    cluster.append(nbr)
                    queue.append(nbr)
        # ** flip cluster **
        for s in cluster:
            self.state[s] *= -1

    def kawasaki(self, sweeps: int = 1) -> None:
        """method conserves average magnetization, spin flip random neighbour pairs"""
        for _ in range(sweeps * self.L2):
            idx: int = randrange(self.L2)
            opposite_spins: list[int] = []
            for nbr in self.neighbours[idx]:
                if self.state[idx] != self.state[nbr]:  # <antiparallel-condition>
                    opposite_spins.append(nbr)
            if len(opposite_spins) < 1:
                continue
            idx_nbr: int = choice(opposite_spins)
            E1: int = -self.state[idx] * self.neighbour_sum(idx)
            E2: int = -self.state[idx_nbr] * self.neighbour_sum(idx_nbr)
            dE: int = E1 - E2
            if random() < exp(-dE * self.beta):
                temp1, temp2 = self.state[idx], self.state[idx_nbr]
                self.state[idx], self.state[idx_nbr] = temp2, temp1

    def get_neighbours(self) -> dict[int, tuple[int, int, int, int]]:
        return {
            idx: (
                (idx // self.L) * self.L + (idx + 1) % self.L,
                (idx + self.L) % self.L2,
                (idx // self.L) * self.L + (idx - 1) % self.L,
                (idx - self.L) % self.L2,
            )
            for idx in range(self.L2)
        }

    def neighbour_sum(self, idx: int) -> int:
        return sum([self.state[nbr] for nbr in self.neighbours[idx]])

    def energy(self) -> float:
        return 0.5 * sum(
            [-self.state[idx] * self.neighbour_sum(idx) for idx in range(self.L2)]
        )

    def magnetization(self) -> int:
        return abs(sum(self.state))

    def update(self, T: float) -> None:
        self.beta, self.prob = 1 / T if T > 0 else inf, 1 - exp(-2 * self.beta)

    def initial_state(self) -> list[int]:
        return [choice([-1, 1]) for _ in range(self.L2)]


class Metropolis(Ising2D):
    def __init__(
        self, size: int, T: float = Ising2D.Tc, initial_state: list[int] | None = None
    ):
        super().__init__(size, T, initial_state)

    def __call__(self, ninfo: int = 1) -> None:
        self.metropolis(sweeps=ninfo)


class Kawasaki(Ising2D):
    def __init__(
        self, size: int, T: float = Ising2D.Tc, initial_state: list[int] | None = None
    ):
        super().__init__(size, T, initial_state)

    def __call__(self, ninfo: int = 1) -> None:
        for _ in range(ninfo):
            self.kawasaki()


class Wolff(Ising2D):
    def __init__(
        self, size: int, T: float = Ising2D.Tc, initial_state: list[int] | None = None
    ):
        super().__init__(size, T, initial_state)

    def __call__(self, ninfo: int = 1) -> None:
        for _ in range(ninfo):
            self.wolff()


class HybridWolfMetropolis(Ising2D):
    def __init__(
        self,
        size: int,
        T: float = Ising2D.Tc,
        ratio: int = 16,
        initial_state: list[int] | None = None,
    ):
        super().__init__(size, T, initial_state)
        self.ratio: int = ratio

    def __call__(self, ninfo: int = 1) -> None:
        for _ in range(ninfo):
            self.wolff()
            self.metropolis(self.ratio)


class Method(Enum):
    METROPOLIS = "metropolis"
    WOLFF = "wolff"
    KAWASAKI = "kawasaki"
    HYBRID = "hybrid"
