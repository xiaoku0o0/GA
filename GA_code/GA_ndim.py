"""
遗传算法示例文件
多个决策变量，含约束条件的遗传算法
2025/11/18
"""

import numpy as np
from typing import Callable

class GA_2D:
    def __init__(
            self,
            # === 算法参数 ===
            population_size: int = 50,  # 种群规模，每代个体数
            gene_length: int = 8,  # 单个基因长度
            num_generations: int = 100,  # 代数
            crossing_rate: float = 0.4,  # 交叉率
            mutation_rate: float = 0.3,  # 突变率
            # === 待优化问题 ===
            dim: int = 2,  # 决策变量个数
            objective_func: Callable[[float, float], float] = lambda x: x,  # 目标函数
            objective_range: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),  # 决策变量范围
    ):
        self.population_size = population_size
        self.gene_length = gene_length
        self.num_generations = num_generations
        self.dim = dim
        self.objective_func = objective_func
        self.objective_range = objective_range
        self.cross_rate = crossing_rate
        self.mutation_rate = mutation_rate
        self.population_ls: list[tuple[int]] = []  # 当代所有种群
        self.calculate()

    def calculate(self):
        # 算法开始
        self.population_init()  # 初始化种群
        for idx_generation in range(self.num_generations):  # 迭代
            fitness = np.array([
                self.objective_func(*self.gray_decoding(population))
                for population in self.population_ls])  # 计算所有个体适应度
            elite: tuple[int] = self.population_ls[np.argmax(fitness)]  # 适应度最大的个体的基因
            fitness = (fitness - fitness.min()) / (fitness.max() - fitness.min())
            fitness = fitness / fitness.sum()  # 归一化
            parent_ls: list[tuple[int]] = self.choose(fitness)  # 基于轮盘赌的亲本选择
            next_ls: list[tuple[int]] = []
            for gene_idx in range(len(parent_ls) - 1):
                for _ in range(200):
                    gene_aft = self.mutation(self.cross(parent_ls[gene_idx:gene_idx + 2]))[0]
                    if self.judge_func(self.gray_decoding(gene_aft)):
                        break
                else:
                    gene_aft = parent_ls[gene_idx]
                next_ls.append(gene_aft)
            self.population_ls = next_ls.copy()
            self.population_ls.insert(0, elite)  # 保留精英
        max_point = self.gray_decoding(self.population_ls[0])  # 最大值点
        max_value = self.objective_func(*max_point)  # 最大值
        print(f"{max_point=}\n{max_value=}")

    def judge_func(self, x):
        return np.sum(x) < 5

    def population_init(self):
        """初始化种群"""
        self.population_ls = []
        for idx in range(self.population_size):
            while True:
                gene = tuple(np.random.randint(0, 1 + 1) for _ in range(self.gene_length * self.dim))
                if self.judge_func(self.gray_decoding(gene)):
                    self.population_ls.append(gene)
                    break

    def gray_decoding(self,
                      gene: tuple[int]  # 基因
                      ) -> list[float]:
        """格雷解码"""
        gene = np.array(gene)  # 便于计算
        res: list[float] = []
        for var_idx in range(self.dim):
            mask = np.triu(np.ones(self.gene_length, dtype=int))
            gene_aft = np.mod(gene[var_idx * self.gene_length:(var_idx + 1) * self.gene_length] @ mask, 2)  # 模二加法
            res.append(np.interp(
                int(''.join(map(str, gene_aft)), 2),  # 十进制的值
                [0, 2 ** self.gene_length - 1],  # 基因可编码的范围
                self.objective_range[var_idx]  # 定义域
            ))
        return res

    def choose(self,
               fitness: np.ndarray  # 归一化的适应度
               ) -> list[tuple[int]]:
        """轮盘赌选择"""
        prefix_sum = np.cumsum(fitness)  # 前缀和
        random_ls: np.ndarray = np.random.uniform(low=0, high=1, size=len(fitness))  # 产生随机数
        res_ls: list[tuple[int]] = []
        for random_num in random_ls:
            idx = np.argmax(prefix_sum >= random_num)  # prefix_sum第一个大于random_num的索引
            res_ls.append(self.population_ls[idx])
        return res_ls

    def cross(self,
              parent_ls: list[tuple[int]]  # 亲本
              ) -> list[tuple[int]]:
        """亲本交叉操作"""
        parents = np.array(parent_ls)
        random_ls = np.random.uniform(low=0, high=1, size=parents.shape) < self.cross_rate
        res = parents[:-1, :] * (1 - random_ls[:-1, :]) + parents[1:, :] * random_ls[:-1, :]
        return list(map(tuple, res))

    def mutation(self,
                 parent_ls: list[tuple[int]]  # 亲本
                 ) -> list[tuple[int]]:
        """变异操作"""
        parents = np.array(parent_ls)
        random_ls = np.random.uniform(low=0, high=1, size=parents.shape) < self.mutation_rate
        res = np.bitwise_xor(parents, random_ls)
        return list(map(tuple, res))

if __name__ == "__main__":
    x_range = ((-20, 20), (-20, 20))
    aim_func = lambda x1, x2: 20 * np.exp(-0.2 * np.sqrt(x1 ** 2 / 2 + x2 ** 2 / 2)) + \
                              np.exp(np.cos(0.5 * np.pi * x1) / 2 + np.cos(0.5 * np.pi * x2) / 2)
    GA_2D(crossing_rate=0.2, mutation_rate=0.08, population_size=20, num_generations=100, objective_func=aim_func,
          objective_range=x_range)
