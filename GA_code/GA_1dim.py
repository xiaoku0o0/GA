"""
遗传算法示例文件
一个决策变量，不含约束条件的遗传算法
2025/10/9
"""

from typing import Callable
import numpy as np

class GA:
    def __init__(self,
                 # === 算法参数 ===
                 population_size:int = 50,  # 种群规模，每代个体数
                 gene_length:int = 8,       # 基因长度
                 num_generations:int = 100, # 代数
                 crossing_rate:float = 0.4, # 交叉率
                 mutation_rate:float = 0.3, # 突变率
                 # === 待优化问题 ===
                 objective_func:Callable[[float], float] = lambda x: x, # 目标函数
                 objective_range:tuple[float, float] = (0, 1),          # 决策变量最值
                 ):
        self.population_size = population_size
        self.gene_length = gene_length
        self.num_generations = num_generations
        self.objective_func = objective_func
        self.objective_range = objective_range
        self.cross_rate = crossing_rate
        self.mutation_rate = mutation_rate
        self.population_ls:list[tuple[int]] = []    # 当代所有种群

    def calculate(self):
        # 算法开始
        self.population_init()  # 初始化种群
        for idx_generation in range(self.num_generations):   # 迭代
            fitness = np.array([
                self.objective_func(self.gray_decoding(population))
                for population in self.population_ls])   # 计算所有个体适应度
            elite:tuple[int] = self.population_ls[np.argmax(fitness)]  # 适应度最大的个体的基因
            fitness = (fitness - fitness.min()) / (fitness.max() - fitness.min())
            fitness = fitness / fitness.sum()   # 归一化
            parent_ls: list[tuple[int]] = self.choose(fitness)  # 基于轮盘赌的亲本选择
            self.population_ls = self.mutation(self.cross(parent_ls))   # 交叉与变异，产生下一代
            self.population_ls.insert(0, elite)     # 保留精英
        max_point = self.gray_decoding(self.population_ls[0])   # 最大值点
        max_value = self.objective_func(max_point)   # 最大值
        return max_value, max_point

    def population_init(self):
        """初始化种群"""
        self.population_ls = [
            tuple(np.random.randint(0, 1+1) for _ in range(self.gene_length))
        for idx in range(self.population_size)]

    def gray_decoding(self,
                      gene:tuple[int]  # 基因
                      ) -> float:
        """格雷解码"""
        gene = np.array(gene)  # 便于计算
        mask = np.triu(np.ones(len(gene), dtype=int))
        gene_aft = np.mod(gene @ mask, 2)  # 模二加法

        return np.interp(
            int(''.join(map(str, gene_aft)), 2),  # 十进制的值
            [0, 2 ** self.gene_length - 1],  # 基因可编码的范围
            self.objective_range  # 定义域
        )

    def choose(self,
               fitness:np.ndarray   # 归一化的适应度
               ) -> list[tuple[int]]:
        """轮盘赌选择"""
        prefix_sum = np.cumsum(fitness)    # 前缀和
        random_ls: np.ndarray = np.random.uniform(low=0, high=1, size=len(fitness)) # 产生随机数
        res_ls: list[tuple[int]] = []
        for random_num in random_ls:
            idx = np.argmax(prefix_sum >= random_num)   # prefix_sum第一个大于random_num的索引
            res_ls.append(self.population_ls[idx])
        return res_ls

    def cross(self,
              parent_ls: list[tuple[int]]    # 亲本
              ) -> list[tuple[int]]:
        """亲本交叉操作"""
        parents = np.array(parent_ls)
        random_ls = np.random.uniform(low=0, high=1, size=parents.shape) < self.cross_rate
        res = parents[:-1,:] * (1 - random_ls[:-1,:]) + parents[1:,:] * random_ls[:-1,:]
        return list(map(tuple, res))

    def mutation(self,
                 parent_ls:list[tuple[int]] # 亲本
                 ) -> list[tuple[int]]:
        """变异操作"""
        parents = np.array(parent_ls)
        random_ls = np.random.uniform(low=0, high=1, size=parents.shape) < self.mutation_rate
        res = np.bitwise_xor(parents, random_ls)
        return list(map(tuple, res))

if __name__ == "__main__":
    ga =  GA(objective_func=lambda x: -x**3+x**2, objective_range=(-1, 1))
    print(ga.calculate())
    print(1)
