from manim import *
import numpy as np
from config import *
from typing import Callable
from custom_class import Histogram

frame_time = 0.2

class BM10(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        np.random.seed(6)   # 固定随机数种子，防止每次渲染动画时结果不同
        self.manim_ga()

    def manim_ga(manim_self):
        # 遗传算法动画
        class GA:
            # 遗传算法类
            def __init__(self,
                         # === 算法参数 ===
                         population_size: int = 50,  # 种群规模，每代个体数
                         gene_length: int = 8,  # 基因长度
                         num_generations: int = 100,  # 代数
                         crossing_rate: float = 0.4,  # 交叉率
                         mutation_rate: float = 0.3,  # 突变率
                         # === 待优化问题 ===
                         objective_func: Callable[[float], float] = lambda x: x,  # 目标函数
                         objective_range: tuple[float, float] = (0, 1),  # 决策变量最值
                         ):
                self.population_size = population_size
                self.gene_length = gene_length
                self.num_generations = num_generations
                self.objective_func = objective_func
                self.objective_range = objective_range
                self.cross_rate = crossing_rate
                self.mutation_rate = mutation_rate
                self.population_ls: list[tuple[int]] = []  # 当代所有种群
                self.max_fit_rec: list[float] = []  # 历代最大适应度记录
                self.calculate()

            def calculate(self):
                # 算法开始
                self.population_init()  # 初始化种群
                for idx_generation in range(self.num_generations):  # 迭代
                    fitness = np.array([
                        self.objective_func(self.gray_decoding(population))
                        for population in self.population_ls])  # 计算所有个体适应度
                    self.max_fit_rec.append(np.max(fitness))
                    elite: tuple[int] = self.population_ls[np.argmax(fitness)]  # 适应度最大的个体的基因
                    fitness = (fitness - fitness.min()) / (fitness.max() - fitness.min())
                    fitness = fitness / fitness.sum()  # 归一化
                    parent_ls: list[tuple[int]] = self.choose(fitness)  # 基于轮盘赌的亲本选择
                    self.population_ls = self.mutation(self.cross(parent_ls))  # 交叉与变异，产生下一代
                    self.population_ls.insert(0, elite)  # 保留精英
                    self.manim_animate()  # 动画
                max_point = self.gray_decoding(self.population_ls[0])  # 最大值点
                max_value = self.objective_func(max_point)  # 最大值
                return max_value, max_point

            def population_init(self):
                """初始化种群"""
                self.population_ls = [
                    tuple(np.random.randint(0, 1 + 1) for _ in range(self.gene_length))
                    for idx in range(self.population_size)]

            def gray_decoding(self,
                              gene: tuple[int]  # 基因
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

            def manim_animate(self):
                fitness = np.array([
                    self.objective_func(self.gray_decoding(population))
                    for population in self.population_ls])  # 计算所有个体适应度
                font_size=18    # 坐标轴字号
                fit_range=[-10,10]
                fit_step=2  # 适应度刻度线步长
                # 直方图
                histogram_obj = Histogram(
                    data=fitness, axis_range=fit_range, axis_step=fit_step, bin_num=50, size=(4, 2.5), font_size=font_size,
                    bar_stroke_color=BLACK, bar_fill_color=YELLOW_COLOR, boxline_color=YELLOW_COLOR,
                    bar_stroke_opacity=1, boxline_width=0.2, box_buff=0.6, box_width=0.3,
                )
                # 历代最大适应度
                max_fit_rec_axes = Axes(
                    x_range=[0, self.num_generations, 10],
                    y_range=histogram_obj.obj_axes.y_range,
                    x_length=8,
                    y_length=histogram_obj.obj_axes.y_length,
                )
                max_fit_rec_axes.shift(histogram_obj.obj_axes.c2p(0,0,0)-max_fit_rec_axes.c2p(0,0,0)).shift(1.5*RIGHT)
                max_fit_rec_x_axes = NumberLine(
                    x_range=max_fit_rec_axes.x_range,
                    length=max_fit_rec_axes.x_length,
                    include_tip=False,
                    include_numbers=True,
                    font_size=font_size,
                )
                max_fit_rec_x_axes.shift(
                    max_fit_rec_axes.c2p(0, max_fit_rec_axes.y_range[0], 0) - max_fit_rec_x_axes.n2p(0))
                max_fit_rec_x_label = Text("代数", font_size=font_size, font="Dream Han Serif CN"
                                           ).move_to(max_fit_rec_x_axes.n2p(self.num_generations/2)).shift(DOWN * 0.6)

                max_fit_rec_y_axes = NumberLine(
                    x_range=max_fit_rec_axes.y_range,
                    length=max_fit_rec_axes.y_length,
                    include_tip=False,
                    include_numbers=True,
                    font_size=font_size,
                    label_direction=LEFT,
                    rotation=90*DEGREES,
                )
                max_fit_rec_y_axes.shift(
                    max_fit_rec_axes.c2p(0, 0, 0) - max_fit_rec_y_axes.n2p(0))
                max_fit_rec_y_label = MathTex("y", font_size=font_size).move_to(max_fit_rec_y_axes.n2p(max_fit_rec_y_axes.x_max)).shift(RIGHT * 0.3)
                max_fit_rec_obj = VGroup(max_fit_rec_x_axes, max_fit_rec_y_axes, max_fit_rec_x_label, max_fit_rec_y_label)
                zero_point = Dot(point=max_fit_rec_x_axes.n2p(0), radius=0, fill_opacity=0, stroke_opacity=0) # 定位点
                for idx, fit in enumerate(self.max_fit_rec):
                    max_fit_rec_obj.add(
                        Dot(
                            point=max_fit_rec_axes.c2p(idx, fit, 0),
                            color=YELLOW_COLOR,
                            radius=0.01,
                        )
                    )
                    if idx != 0:
                        max_fit_rec_obj.add(
                            Line(
                                start=max_fit_rec_axes.c2p(idx-1, self.max_fit_rec[idx-1], 0),
                                end=max_fit_rec_axes.c2p(idx, fit, 0),
                                color=YELLOW_COLOR,
                                stroke_width=0.4,
                            )
                        )
                histogram_all_obj = VGroup(histogram_obj, max_fit_rec_obj, zero_point).center().to_edge(DOWN, buff=0.5)  # 包含直方图和历代数据

                # 当代数据
                func_axes = Axes(
                    x_range=self.objective_range,
                    y_range=histogram_obj.obj_axes.y_range,
                    x_length=8,
                    y_length=3,
                )
                func_axes.shift(zero_point.get_center()-func_axes.c2p(func_axes.x_range[0],0,0))
                x_axes = NumberLine(x_range=func_axes.x_range,
                                    length=func_axes.x_length,
                                    include_numbers=True,
                                    include_tip=False,
                                    font_size=font_size)
                x_axes.shift(func_axes.c2p(0, func_axes.y_range[0], 0) - x_axes.n2p(0))  # 下移
                x_label = MathTex("x", font_size=font_size
                                  ).move_to(x_axes.n2p(0)).shift(DOWN * 0.6)

                y_axes = NumberLine(x_range=func_axes.y_range,
                                    length=func_axes.y_length,
                                    include_numbers=True,
                                    label_direction=LEFT,
                                    include_tip=False,
                                    font_size=font_size,
                                    rotation=90 * DEGREES)
                y_axes.shift(func_axes.c2p(func_axes.x_range[0], 0, 0) - y_axes.n2p(0))  # 左移
                y_label = MathTex("y", font_size=font_size
                                  ).move_to(y_axes.n2p(y_axes.x_max)).shift(RIGHT * 0.3)
                aim_func_obj = func_axes.plot(self.objective_func, x_range=[*self.objective_range, np.diff(self.objective_range)[0]/3000])
                aim_func_obj.set_stroke(width=1)

                func_obj = VGroup()
                x_ls = [self.gray_decoding(population) for population in self.population_ls]
                for x, fit in zip(x_ls, fitness):
                    func_obj.add(
                        Line(
                            start=func_axes.c2p(x, func_axes.y_range[0], 0),
                            end=func_axes.c2p(x, fit, 0),
                            color=YELLOW_COLOR,
                            stroke_width=0.5,
                        )
                    )

                func_obj.add(x_axes, y_axes, x_label, y_label, aim_func_obj)
                func_obj.add(Star(   # 最优个体标记
                    outer_radius=0.05,color=YELLOW_COLOR,fill_color=YELLOW_COLOR, fill_opacity=0.6
                ).move_to(func_axes.c2p(x_ls[np.argmax(fitness)], np.max(fitness), 0)))
                func_obj.to_edge(UP, buff=0.5)
                # 标签信息
                # 目标函数
                aim_func_tex_obj = MathTex(r"\text{max }y=f(x)=(x^2-1.5) \sum\limits_{n=1}^{100}0.5^n "
                                           r"\cos(7^n \pi x)+\frac{x}{2}", font_size=font_size)
                aim_func_tex_obj.set_color(WHITE)

                boundary_tex_obj = MathTex(r"x \in [-3, 3]", font_size=font_size)
                boundary_tex_obj.set_color(WHITE)

                rate_info = Text(f"种群大小:{self.population_size}, 交叉率:{self.cross_rate}, 突变率:{self.mutation_rate}",
                                 font_size=font_size, color=WHITE, font="Dream Han Serif CN")
                now_info = VGroup(
                    Text("当前第", font_size=font_size, color=WHITE, font="Dream Han Serif CN"),
                    Text(f"{len(self.max_fit_rec)}", font_size=font_size, color=YELLOW_COLOR, font="DINPro"),
                    Text("代，最大值:", font_size=font_size, color=WHITE, font="Dream Han Serif CN"),
                    Text(f"{np.max(fitness):.3f}", font_size=font_size, color=YELLOW_COLOR, font="DINPro"),
                ).arrange(RIGHT, buff=0.1)
                info = VGroup(aim_func_tex_obj, boundary_tex_obj, rate_info, now_info).arrange(DOWN, buff=0.3)
                info.to_edge(UL, buff=0.5)


                manim_self.add(histogram_all_obj, func_obj, info)
                manim_self.wait(frame_time)
                manim_self.remove(histogram_all_obj, func_obj, info)

        aim_func = lambda x: (x ** 2 - 1.5) * np.sum(
            [0.5 ** n * np.cos(7 ** n * np.pi * x) for n in np.arange(1, 100 + 1, 1)]) + x / 2
        GA(population_size=20, objective_func=aim_func, objective_range=[-3, 3], num_generations=200, crossing_rate=0.2, mutation_rate=0.08)
