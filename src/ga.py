# -*- coding: utf-8 -*-
"""
遗传算法（GA）超参数寻优 —— 智能优化技术模块（从零实现，仅依赖 numpy）。

不引入 deap 等第三方库，手写标准遗传算法：
  - 参数编码：实数（float）/ 整数（int）/ 离散（choice，可为 None、字符串）三类混合；
  - 遗传算子：锦标赛选择 + 单点交叉 + 逐基因变异 + 精英保留（elitism）；
  - 适应度缓存：相同个体只评估一次，避免重复训练。

适配任意 sklearn 估计器：`fitness_fn(individual: dict) -> float`，返回值越大越好
（分类用 macro-F1，回归用 -MAE 取负号转化为最大化问题）。

用法：
    from src.ga import GeneticOptimizer

    space = {
        "n_estimators": {"type": "int", "low": 80, "high": 260},
        "max_depth":    {"type": "int", "low": 6, "high": 24},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
        "max_features": {"type": "choice", "values": ["sqrt", "log2", None]},
    }
    ga = GeneticOptimizer(space, fitness_fn, pop_size=12, n_generations=8, seed=42)
    best_params, best_fitness = ga.run()
"""
import copy

import numpy as np


class GeneticOptimizer:
    """遗传算法超参数寻优器。

    参数
    ----
    param_space : dict
        {参数名: {"type": "int"|"float"|"choice", ...}} 的搜索空间定义。
    fitness_fn : callable
        接收个体 dict、返回浮点适应度（越大越好）。
    pop_size : int
        种群规模。
    n_generations : int
        进化代数。
    crossover_rate : float
        交叉概率（单点交叉）。
    mutation_rate : float
        变异概率（逐基因随机重置）。
    elite_ratio : float
        精英保留比例（直接进入下一代）。
    maximize : bool
        True 为最大化适应度（默认）。
    seed : int
        随机种子，保证可复现。
    """

    def __init__(self, param_space, fitness_fn, pop_size=12, n_generations=8,
                 crossover_rate=0.8, mutation_rate=0.25, elite_ratio=0.2,
                 maximize=True, seed=42, verbose=True):
        self.space = param_space
        self.fitness_fn = fitness_fn
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.maximize = maximize
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self._cache = {}          # 适应度缓存：个体签名 -> 适应度
        self.history = []         # [(代数, 该代最佳适应度)]

    # ---------- 个体生成与遗传算子 ----------
    def _random_gene(self, spec):
        t = spec["type"]
        if t == "int":
            return int(self.rng.integers(spec["low"], spec["high"] + 1))
        if t == "float":
            return float(self.rng.uniform(spec["low"], spec["high"]))
        # choice
        return spec["values"][int(self.rng.integers(len(spec["values"])))]

    def _random_individual(self):
        return {k: self._random_gene(s) for k, s in self.space.items()}

    def _mutate(self, ind):
        out = copy.deepcopy(ind)
        k = self.rng.choice(list(self.space.keys()))
        out[k] = self._random_gene(self.space[k])
        return out

    def _crossover(self, a, b):
        """单点交叉：在参数维度上随机切一个点，右侧基因互换。"""
        keys = list(self.space.keys())
        if len(keys) < 2:
            return copy.deepcopy(a), copy.deepcopy(b)
        cut = int(self.rng.integers(1, len(keys)))
        c1, c2 = copy.deepcopy(a), copy.deepcopy(b)
        for k in keys[cut:]:
            c1[k], c2[k] = c2[k], c1[k]
        return c1, c2

    # ---------- 适应度评估（带缓存） ----------
    def _fit(self, ind):
        key = tuple(sorted(ind.items()))
        if key not in self._cache:
            self._cache[key] = self.fitness_fn(ind)
        return self._cache[key]

    def _tournament(self, scored, k=3):
        cand = self.rng.choice(len(scored), size=min(k, len(scored)), replace=False)
        return int(max(cand, key=lambda i: scored[i][1]))

    # ---------- 主循环 ----------
    def run(self):
        pop = [self._random_individual() for _ in range(self.pop_size)]

        for gen in range(self.n_generations):
            scored = [(ind, self._fit(ind)) for ind in pop]
            scored.sort(key=lambda x: x[1], reverse=self.maximize)
            best = scored[0]
            self.history.append((gen, best[1]))
            if self.verbose:
                print(f"  [GA] 第 {gen + 1}/{self.n_generations} 代  最佳适应度={best[1]:.4f}")

            # 精英保留
            n_elite = max(1, int(self.pop_size * self.elite_ratio))
            new_pop = [copy.deepcopy(ind) for ind, _ in scored[:n_elite]]

            # 选择 + 交叉 + 变异生成子代
            while len(new_pop) < self.pop_size:
                p1 = scored[self._tournament(scored)][0]
                p2 = scored[self._tournament(scored)][0]
                if self.rng.random() < self.crossover_rate:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                if self.rng.random() < self.mutation_rate:
                    c1 = self._mutate(c1)
                if self.rng.random() < self.mutation_rate:
                    c2 = self._mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            pop = new_pop[:self.pop_size]

        # 收敛后全体重算，取全局最优
        scored = [(ind, self._fit(ind)) for ind in pop]
        scored.sort(key=lambda x: x[1], reverse=self.maximize)
        self.best_individual_ = scored[0][0]
        self.best_fitness_ = scored[0][1]
        if self.verbose:
            print(f"  [GA] 收敛：最佳适应度={self.best_fitness_:.4f}  最佳参数={self.best_individual_}")
        return dict(self.best_individual_), self.best_fitness_
