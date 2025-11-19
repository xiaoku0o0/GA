from manim import *
import numpy as np
from config import *
from typing import Callable
from custom_class import Histogram

frame_time = 1

class CM2(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        np.random.seed(0)   # 固定随机数种子，防止每次渲染动画时结果不同
        self.manim_bpnn_ga()

    def manim_bpnn_ga(manim_self):
        # BPNN-GA动画
        rotate_value_tracker = ValueTracker(0)  # 坐标轴旋转
        rotate_step = 5*DEGREES     # 旋转步进

        class BPNN_GA:
            def __init__(
                    self,
                    node_num_ls: list[int],  # 节点配置
                    activation_func_ls: list[Callable],  # 激活函数配置(激活函数需要支持ndarray列向量操作)
            ):
                self.node_num_ls = node_num_ls
                self.layer_num = len(node_num_ls)
                self.activation_func_ls = activation_func_ls
                if len(self.node_num_ls) != len(self.activation_func_ls) + 1:
                    raise ValueError(
                        f"node_num_ls:{len(self.node_num_ls)} activation_func_ls:{len(self.activation_func_ls)}\n"
                        f"节点数长度应为激活函数长度-1（输入层无激活函数）")

            def train(
                    self,
                    train_data: np.ndarray,  # 训练集
                    except_data: np.ndarray,
                    gene_length: int = 16,  # 单变量基因长度
                    population_size: int = 50,  # 种群规模
                    num_generations: int = 100,  # 代数
                    cross_rate: float = 0.4,  # 交叉率
                    mutation_rate: float = 0.3,  # 突变率
                    weight_range: tuple[float, float] = (-10, 10),  # 权重范围
                    bias_range: tuple[float, float] = (-10, 10),  # 偏置范围
            ):
                if except_data.shape[0] != train_data.shape[0]:
                    raise ValueError(f"输入数据量{train_data.shape[0]}与预期输出数据量不匹配{except_data.shape[0]}")
                # BPNN训练主函数
                self.train_data = train_data
                self.except_data = except_data

                self.gene_length = gene_length
                self.population_size = population_size
                self.num_generations = num_generations
                self.cross_rate = cross_rate
                self.mutation_rate = mutation_rate
                self.weight_range = weight_range
                self.bias_range = bias_range

                self.gene_all_length = gene_length * (
                        np.array(self.node_num_ls[:-1]) @ np.array(self.node_num_ls[1:]) + sum(self.node_num_ls[1:])
                )  # 基因总长度

                self.init_gene()
                self.max_fit_rec: list[float] = []  # 历代最大适应度记录
                for idx_generation in range(self.num_generations):  # 迭代
                    fitness = np.array([
                        -self.get_loss(train_data, except_data, *self.gene2parameter(self.population_ls[idx]))
                        for idx in range(self.population_size)])
                    self.max_fit_rec.append(np.max(fitness))
                    elite: tuple[int] = self.population_ls[np.argmax(fitness)]  # 适应度最大的个体的基因
                    fitness = (fitness - fitness.min()) / (fitness.max() - fitness.min())
                    fitness = fitness / fitness.sum()  # 归一化
                    parent_ls: list[tuple[int]] = self.choose(fitness)  # 基于轮盘赌的亲本选择
                    self.population_ls = self.mutation(self.cross(parent_ls))  # 交叉与变异，产生下一代
                    self.population_ls.insert(0, elite)  # 保留精英
                    self.manim_animate()  # 动画
                max_point = self.gene2parameter(self.population_ls[0])  # 最大值点
                max_value = -self.get_loss(train_data, except_data, *max_point)  # 最大值

            def init_gene(self):
                # 产生初始种群
                self.population_ls = [
                    tuple(np.random.randint(0, 1 + 1) for _ in range(self.gene_all_length))
                    for idx in range(self.population_size)]

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

            def gene2parameter(self, gene: tuple[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
                # 基因转参数
                weight_ls = []
                bias_ls = []

                def decoding(
                        gene: tuple[int],
                        gene_range: tuple[float, float],
                        res_size: tuple[int, int] = (1, 1),
                ) -> np.ndarray:
                    # 基因转ndarray
                    if len(gene) != res_size[0] * res_size[1] * self.gene_length:
                        raise ValueError(f"基因长度不符，预期为{res_size}*{self.gene_length}，实际为{len(gene)}")
                    data = np.array(gene).reshape(res_size[0] * res_size[1], self.gene_length)
                    res = []
                    mask = np.triu(np.ones(self.gene_length, dtype=int))
                    for line in data:
                        gray_decode_res = np.mod(line @ mask, 2)  # 格雷解码
                        res.append(
                            int(''.join(map(str, gray_decode_res)), 2)
                        )
                    res = np.interp(
                        np.array(res),
                        (0, (1 << self.gene_length) - 1),
                        gene_range
                    ).reshape(res_size[0], res_size[1])
                    return res

                # 填充weight_ls
                gene_idx = 0
                for front_idx in range(self.layer_num - 1):
                    shape = (self.node_num_ls[front_idx], self.node_num_ls[front_idx + 1])
                    weight_ls.append(decoding(
                        gene=gene[gene_idx:gene_idx + shape[0] * shape[1] * self.gene_length],
                        gene_range=self.weight_range,
                        res_size=shape,
                    ))
                    gene_idx += shape[0] * shape[1] * self.gene_length

                # 填充bias_ls
                for front_idx in range(1, self.layer_num):
                    shape = (self.node_num_ls[front_idx], 1)  # bias为列向量
                    bias_ls.append(decoding(
                        gene=gene[gene_idx:gene_idx + shape[0] * shape[1] * self.gene_length],
                        gene_range=self.bias_range,
                        res_size=shape,
                    ))
                    gene_idx += shape[0] * shape[1] * self.gene_length

                if gene_idx < len(gene):
                    raise RuntimeError(f"存在基因冗余, 现基因长度为:{len(gene)}")

                return weight_ls, bias_ls

            def data_input(
                    self,
                    data: np.ndarray,
                    weight_ls: list[np.ndarray],
                    bias_ls: list[np.ndarray],
                    until: int=-1, # 在第几层终止计算
            ) -> np.ndarray[float]:
                # 数据正向传递
                if data.shape[1] != self.node_num_ls[0]:
                    raise ValueError(
                        f"数据输入大小{data.shape}与网络输入层节点数{self.node_num_ls[0]}（输入预期列数）不符")
                res_now = data
                for layer_step in range(1, (self.layer_num + 1 + until) if until < 0 else until):
                    res_now = self.activation_func_ls[layer_step - 1](res_now @ weight_ls[layer_step - 1] -
                                                                      np.repeat(bias_ls[layer_step - 1].T,
                                                                                data.shape[0], axis=0))
                return res_now

            def get_loss(
                    self,
                    input_data: np.ndarray,
                    except_data: np.ndarray,
                    weight_ls: list[np.ndarray],
                    bias_ls: list[np.ndarray],
            ) -> float:
                # 计算损失
                return np.sum((self.data_input(input_data, weight_ls, bias_ls) - except_data) ** 2)

            def manim_animate(self):
                fitness = np.array([
                    -self.get_loss(train_data, except_data, *self.gene2parameter(self.population_ls[idx]))
                    for idx in range(self.population_size)])  # 计算所有个体适应度
                elite_idx = np.argmax(fitness)
                elite_gene = self.population_ls[elite_idx]
                font_size=18    # 坐标轴字号
                fit_range=[-3,0]
                fit_step=1  # 适应度刻度线步长
                # 直方图=================================
                histogram_obj = Histogram(
                    data=fitness, axis_range=fit_range, axis_step=fit_step, bin_num=50, size=(4, 2), font_size=font_size,
                    bar_stroke_color=BLACK, bar_fill_color=YELLOW_COLOR, boxline_color=YELLOW_COLOR,
                    bar_stroke_opacity=1, boxline_width=0.2, box_buff=0.6, box_width=0.3,
                )
                # 历代最大适应度
                max_fit_rec_axes = Axes(
                    x_range=[0, self.num_generations, 5],
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
                max_fit_rec_y_label = MathTex("-Loss", font_size=font_size).move_to(max_fit_rec_y_axes.n2p(max_fit_rec_y_axes.x_max)).shift(RIGHT * 0.5)
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

                # 添加激活函数上各点================
                act_fig = VGroup()
                latex_fontsize = 27
                weight_ls, bias_ls = self.gene2parameter(elite_gene)
                data_in = self.data_input(
                    self.train_data, weight_ls, bias_ls, until=-2
                )
                data_out = self.except_data
                b = bias_ls[-1]
                w = weight_ls[-1]
                axes_act = Axes(
                    x_range=[-10, 10, 5],
                    y_range=[-0.1, 1.1, 1],
                    x_length=6.5,
                    y_length=1.2,
                    y_axis_config={
                        "numbers_to_include": np.array([0, 1]),
                        "font_size": latex_fontsize - 3,
                    },
                    tips=False
                ).to_edge(UR, buff=0.5)
                axes_lab = MathTex("z", font_size=latex_fontsize).move_to(axes_act.c2p(1, 1, 0))
                act_func_obj = VGroup(
                    axes_act.plot(lambda x: 1 / (1 + np.exp(-x)), x_range=[axes_act.x_range[0], 0], color=BLUE_D, stroke_width=1.7),
                    axes_act.plot(lambda x: 1 / (1 + np.exp(-x)), x_range=[0, axes_act.x_range[1]], color=YELLOW_D, stroke_width=1.7)
                )

                def update_dot(idx):
                    x = data_in[idx][0] * w[0, 0] + data_in[idx][1] * w[1, 0] + \
                        data_in[idx][2] * w[2, 0] - b[0, 0]
                    dot = Square(
                        side_length=0.1, color=BLUE_B if data_out[idx][0] == 0 else YELLOW_B,
                        fill_opacity=True, fill_color=BLACK,
                        stroke_width=2
                    ).move_to(
                        axes_act.coords_to_point(x, 1 / (1 + np.exp(-x)))
                    ).rotate(45 * DEGREES)
                    return dot

                act_dot_queue = [
                    update_dot(idx).add_updater(lambda x, i=idx: x.become(update_dot(i))) for idx in range(len(data_in))
                ]
                act_fig.add(axes_act, axes_lab, act_func_obj, *act_dot_queue)

                # 当代最优============================
                axes_anchor = VGroup(
                    *[Dot(radius=0) for _ in range(2)]
                ).arrange(RIGHT, buff=4).next_to(axes_act, DOWN, buff=1.5)    # 坐标轴定位点
                axes_fig = VGroup()
                def load_svg(svg_type: int, xyz: list, fill: bool, axes:Axes|ThreeDAxes, fill_opacity: float=1):
                    """加载svg图形"""
                    """
                    svg_type: 0-圆形; 1-三角形
                    xyz: 坐标轴上的xyz坐标
                    fill: 是否填充
                    axes: 载入坐标轴
                    """
                    if fill:
                        path = [r".\svg_files\圆形_填充.svg", r".\svg_files\三角形_填充.svg"]
                    else:
                        path = [r".\svg_files\圆形.svg", r".\svg_files\三角形.svg"]
                    color = [YELLOW_COLOR, YELLOW_COLOR]
                    return SVGMobject(
                        path[svg_type],
                        fill_color=color[svg_type],
                        fill_opacity=fill_opacity,
                        height=0.2
                    ).move_to(axes.coords_to_point(*xyz))

                # 创建坐标轴:2->3----------------------------
                axes_length = 2.3
                def flash_axes_23()->ThreeDAxes:
                    tmp = ThreeDAxes(
                        x_range=[-0.2, 1.2, 1],
                        y_range=[-0.2, 1.2, 1],
                        z_range=[-0.2, 1.2, 1],
                        x_length=axes_length,
                        y_length=axes_length,
                        z_length=axes_length,
                        axis_config={"include_tip": False}
                    )
                    return tmp.rotate(
                        45 * DEGREES + rotate_value_tracker.get_value(), axis=UP,
                        about_point=tmp.c2p((tmp.x_range[1]+tmp.x_range[0]+0.2)/2, (tmp.y_range[1]+tmp.y_range[0]+0.2)/2, (tmp.z_range[1]+tmp.z_range[0])/2)
                    ).rotate(
                        40 * DEGREES, axis=RIGHT
                    ).shift(
                        axes_anchor[1].get_center()-tmp.c2p((tmp.x_range[1]+tmp.x_range[0]+0.2)/2, (tmp.y_range[1]+tmp.y_range[0]+0.2)/2, (tmp.z_range[1]+tmp.z_range[0])/2)
                    ).shift(0.4*LEFT+0.1*UP)
                axes_23 = flash_axes_23().add_updater(lambda m: m.become(flash_axes_23()))
                def flash_axis23_label(idx)->Tex:
                    return [
                        Tex("$y_1$", font_size=latex_fontsize).move_to(axes_23.coords_to_point(1.3, 0, 0)),
                        Tex("$y_2$", font_size=latex_fontsize).move_to(axes_23.coords_to_point(0.1, 1.2, 0)),
                        Tex("$y_3$", font_size=latex_fontsize).move_to(axes_23.coords_to_point(0, 0, 1.3))
                    ][idx]
                axis23_label = VGroup(*[
                    flash_axis23_label(idx).add_updater(lambda m, i=idx: m.become(flash_axis23_label(i)))
                for idx in range(3)])
                for obj in axis23_label:
                    axes_fig.add(obj)
                axes_fig.add(axes_23)

                # 载入图形
                svg23_queue = [
                    load_svg(svg_type=data_out[i, 0], xyz=data_in[i], fill=True, axes=axes_23, fill_opacity=0.8).add_updater(
                        lambda m, idx=i: m.move_to(axes_23.c2p(*data_in[idx]))
                    )
                for i in range(len(data_in))]
                axes_fig.add(*svg23_queue)

                # 标注坐标
                svg23_label = [
                    Text(
                        f"({', '.join(list(map(str, np.round(data_in[idx],2))))})", font="Times New Roman",
                        font_size=15, color=YELLOW_COLOR,
                        fill_opacity=0.7
                    ).move_to(axes_23.coords_to_point(*data_in[idx])).shift(0.2 * UP).add_updater(
                        lambda m, i=idx: m.move_to(axes_23.coords_to_point(*data_in[i])).shift(0.2 * UP)) for idx in range(len(data_in))
                ]
                axes_fig.add(*svg23_label)

                def cut_surface_func(x: float, y: float) -> np.ndarray:
                    return axes_23.coords_to_point(x, y, (b[0, 0] - w[0, 0] * x - w[1, 0] * y) / w[2, 0])

                def space_surface_func(x: float, y: float) -> np.ndarray:
                    return axes_23.coords_to_point(*self.data_input(np.array([[x, y]]), weight_ls, bias_ls, -2)[0])

                x_range = axes_23.x_range[:-1]
                y_range = axes_23.y_range[:-1]
                z_range = axes_23.z_range[:-1]
                def flash_cut_surface()->Surface:
                    return Surface(
                        cut_surface_func,
                        u_range=x_range, v_range=y_range,
                        resolution=[1], checkerboard_colors=[ORANGE, ORANGE],
                        fill_opacity=0.2
                    )
                cut_surface = flash_cut_surface().add_updater(lambda m: m.become(flash_cut_surface()))
                def flash_area_surface()->Surface:
                    return Surface(
                        space_surface_func,
                        u_range=x_range, v_range=y_range,
                        resolution=[32], checkerboard_colors=[BLUE_D, BLUE_D],
                        fill_opacity=0.1
                    )
                area_surface = flash_area_surface().add_updater(lambda m: m.become(flash_area_surface()))
                axes_fig.add(area_surface, cut_surface)

                # 创建坐标轴:1->2----------------------------
                axes_12 = Axes(
                    x_range=[-0.2, 1.2, 1],
                    y_range=[-0.2, 1.2, 1],
                    x_length=axes_length,
                    y_length=axes_length,
                    axis_config={"include_tip": False}
                ).move_to(
                    axes_anchor[0].get_center()
                )
                axis12_label = VGroup(
                    Tex("$x_1$", font_size=latex_fontsize).move_to(axes_12.coords_to_point(1.3, 0, 0)),
                    Tex("$x_2$", font_size=latex_fontsize).move_to(axes_12.coords_to_point(0, 1.3, 0))
                )
                area12_surface = Surface(
                    lambda u, v: axes_12.c2p(u, v, 0),
                    u_range=x_range, v_range=y_range,
                    resolution=[32], checkerboard_colors=[BLUE_D, BLUE_D],
                    stroke_opacity=0.2,
                    fill_opacity=0.1
                )
                axes_fig.add(area12_surface)
                for obj in axis12_label:
                    axes_fig.add(obj)
                axes_fig.add(axes_12)

                # 载入图形
                input = self.data_input(
                    self.train_data, weight_ls, bias_ls, until=0
                )
                svg12_queue = [load_svg(svg_type=data_out[i, 0], xyz=input[i], fill=True, axes=axes_12, fill_opacity=0.8) for i in range(len(data_in))]
                axes_fig.add(*svg12_queue)

                # 标注坐标
                svg12_label = [
                    Text(
                        f"({', '.join(list(map(str, np.round(input[idx], 0))))})", font="Times New Roman",
                        font_size=15, color=YELLOW_COLOR,
                        fill_opacity=0.7
                    ).move_to(axes_12.coords_to_point(*input[idx])).shift(0.2 * UP) for idx in range(len(input))
                ]
                axes_fig.add(*svg12_label)


                # 网络========================
                nn_fig = VGroup()
                nn_node_obj = VGroup(*[
                    VGroup(*[
                        Circle(radius=0.2, stroke_color=WHITE, stroke_width=0.5, stroke_opacity=1, fill_opacity=0)
                    for _ in range(self.node_num_ls[layer_idx])]).arrange(DOWN, buff=0.5)
                for layer_idx in range(self.layer_num)]).arrange(RIGHT, buff=1.8)
                nn_node_lab_obj = VGroup(*[
                    MathTex(
                        f"x_{idx+1}",
                        color=WHITE,
                        opacity=0.8,
                        font_size=18,
                    ).move_to(
                        nn_node_obj[0][idx].get_center()
                    )
                for idx in range(self.node_num_ls[0])])
                nn_node_lab_obj.add(*[
                    MathTex(
                        f"y_{idx + 1}",
                        color=WHITE,
                        opacity=0.8,
                        font_size=18,
                    ).move_to(
                        nn_node_obj[1][idx].get_center()
                    )
                    for idx in range(self.node_num_ls[1])])
                nn_node_lab_obj.add(
                    MathTex(
                        "z",
                        color=WHITE,
                        opacity=0.8,
                        font_size=18,
                    ).move_to(
                        nn_node_obj[2][0].get_center()
                    )
                )
                nn_line_obj = VGroup(*[
                    VGroup(*[
                        VGroup(*[
                            Line(
                                start=nn_node_obj[front_layer_idx][front_node_idx].get_right(),
                                end=nn_node_obj[front_layer_idx+1][next_node_idx].get_left(),
                                color=WHITE,
                                stroke_width=1,
                                stroke_opacity=0.8,
                            )
                        for next_node_idx in range(self.node_num_ls[front_layer_idx+1])])
                    for front_node_idx in range(self.node_num_ls[front_layer_idx])])
                for front_layer_idx in range(self.layer_num-1)])
                nn_weight_obj = VGroup(*[
                    VGroup(*[
                        VGroup(*[
                            Text(
                                text=f"{weight_ls[front_layer_idx][front_node_idx, next_node_idx]:.3f}",
                                color=YELLOW_COLOR,
                                font="Times new Roman",
                                font_size=10,
                            ).rotate(
                                (line:=nn_line_obj[front_layer_idx][front_node_idx][next_node_idx]).get_angle()
                            ).move_to(
                                line.get_end() - line.get_unit_vector()*0.21
                            ).shift(
                                UP*0.1*(1 if front_node_idx==0 else (-1 if front_layer_idx==0 or front_node_idx==2 else 0))
                            )
                        for next_node_idx in range(self.node_num_ls[front_layer_idx+1])])
                    for front_node_idx in range(self.node_num_ls[front_layer_idx])])
                for front_layer_idx in range(self.layer_num-1)])
                nn_bias_obj = VGroup(*[
                    VGroup(*[
                        Text(
                            text=f"{bias_ls[layer_idx-1][node_idx, 0]:.3f}\nsigmoid",
                            color=YELLOW_COLOR,
                            font="Times new Roman",
                            font_size=10,
                        ).next_to(
                            nn_node_obj[layer_idx][node_idx], direction=RIGHT, buff=0.1
                        )
                        for node_idx in range(self.node_num_ls[layer_idx])])
                    for layer_idx in range(1, self.layer_num)])
                nn_fig.add(nn_node_obj, nn_node_lab_obj, nn_line_obj, nn_weight_obj, nn_bias_obj)

                # 标签信息
                rate_info = Text(
                    f"种群大小:{self.population_size}, 交叉率:{self.cross_rate}, 突变率:{self.mutation_rate}",
                    font_size=font_size, color=WHITE, font="Dream Han Serif CN")
                now_info = VGroup(
                    Text("当前第", font_size=font_size, color=WHITE, font="Dream Han Serif CN"),
                    Text(f"{len(self.max_fit_rec)}", font_size=font_size, color=YELLOW_COLOR, font="DINPro"),
                    Text("代，最小损失:", font_size=font_size, color=WHITE, font="Dream Han Serif CN"),
                    Text(f"{-np.max(fitness):.3f}", font_size=font_size, color=YELLOW_COLOR, font="DINPro"),
                ).arrange(RIGHT, buff=0.1)
                info = VGroup(rate_info, now_info).arrange(DOWN, buff=0.3)

                nn_fig.center().next_to(info, direction=DOWN, buff=0.3)
                VGroup(info, nn_fig).to_edge(UL, buff=0.5)

                manim_self.add(histogram_all_obj, axes_fig, act_fig, nn_fig, info)
                rotate_start_value = rotate_value_tracker.get_value()
                def update_func(mob, alpha):
                    current_value = rotate_start_value + rotate_step * alpha
                    rotate_value_tracker.set_value(current_value)
                manim_self.play(
                    UpdateFromAlphaFunc(
                        rotate_value_tracker,
                        update_func,
                        run_time=frame_time,
                        rate_func=linear
                    )
                )
                # 测试用
                # rotate_value_tracker.set_value(rotate_start_value+rotate_step)
                # manim_self.wait(0.1)

                manim_self.remove(histogram_all_obj, axes_fig, act_fig, nn_fig, info)

        def sigmoid(x):
            # sigmoid激活函数
            return 1 / (1 + np.exp(-x))

        np.random.seed(0)
        train_data = np.array([[1, 1], [0, 0], [0, 1], [1, 0]])
        except_data = np.array([[0], [0], [1], [1]])
        nn = BPNN_GA(node_num_ls=[2, 3, 1], activation_func_ls=[sigmoid, sigmoid])
        nn.train(
            train_data=train_data,
            except_data=except_data,
            num_generations=70,
        )
