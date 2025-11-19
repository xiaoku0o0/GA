from manim import *
import numpy as np
from config import *

class BM5(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        lab_buff = 0.2  # 左侧标签间距
        move_right = -2.5  # 整体向右偏移量
        box_height = 0.4  # 标签高度
        box_width = 1.15  # 标签宽度
        box_buff = 0.1  # 标签间距
        label_width = 1.1  # 左侧标签宽度
        font_size = 18
        code_down = 0.75  # 代码中心下移距离
        """
        0-保留基因、适应度、亲本
        """
        # 目标函数
        aim_func_tex_obj = MathTex(r"\text{max }y=f(x)=(x^2-1.5) \sum\limits_{n=1}^{100}0.5^n "
                                   r"\cos(7^n \pi x)+\frac{x}{2}", font_size=30)
        aim_func_tex_obj.set_color(BLUE_COLOR)
        aim_func_tex_obj.to_edge(UP, buff=0.5)

        boundary_tex_obj = MathTex(r"\text{s.t. } -3 \le x \le 3", font_size=30)
        boundary_tex_obj.set_color(BLUE_COLOR)
        boundary_tex_obj.next_to(aim_func_tex_obj, DOWN)
        self.add(aim_func_tex_obj, boundary_tex_obj)

        # 基因
        gene = ["00001011", "01011101", "01101001", "10000110", "10110011"]
        gene_obj_ls = [VGroup(
            Rectangle(width=box_width, height=box_height, fill_opacity=0, stroke_color=BLUE_COLOR, stroke_width=1),
            Text(gene[idx], font_size=font_size, color=BLUE_COLOR, font="Times New Roman")
        ) for idx in range(len(gene))
        ]
        gene_obj = VGroup(*gene_obj_ls)
        gene_obj.arrange(RIGHT, buff=box_buff)
        gene_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1,
                      stroke_color=BLUE_COLOR, stroke_width=1, stroke_opacity=1),
            Text("基因", font_size=font_size, color=BLUE_COLOR, font="Dream Han Serif CN")
        ).next_to(gene_obj, LEFT, buff=lab_buff)
        gene_all_obj = VGroup(gene_obj, gene_lab).shift(RIGHT * move_right + UP * 1.6)
        self.add(gene_all_obj)

        # 目标函数
        dec_val = [int(gene[idx], 2) for idx in range(len(gene))]
        res_val = [
            np.interp(dec_val[idx],
                      [0, 2 ** 8 - 1],
                      [-3, 3])
            for idx in range(len(gene_obj_ls))]
        aim_func = lambda x: (x ** 2 - 1.5) * np.sum(
            [0.5 ** n * np.cos(7 ** n * np.pi * x) for n in np.arange(1, 100 + 1, 1)]) + x / 2
        func_val = [
            aim_func(res_val[idx])
            for idx in range(len(gene_obj_ls))]

        func_obj_ls = [VGroup(
            Rectangle(width=box_width, height=box_height, fill_opacity=0, stroke_color=BLUE_COLOR, stroke_width=1),
            Text(f"{func_val[idx]:.2f}", font_size=font_size, color=BLUE_COLOR, font="Times New Roman")
        ) for idx in range(len(gene))]
        func_obj = VGroup(*func_obj_ls)
        func_obj.arrange(RIGHT, buff=box_buff)
        func_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1,
                      stroke_color=BLUE_COLOR, stroke_width=1, stroke_opacity=1),
            Text("适应度", font_size=font_size, color=BLUE_COLOR, font="Dream Han Serif CN")
        ).next_to(func_obj, LEFT, buff=lab_buff)
        func_all_obj = VGroup(func_obj, func_lab).shift(RIGHT * move_right + 0.8 * UP)
        self.add(func_all_obj)
        line_15_obj_ls = [
            Line(start=gene_obj_ls[idx].get_center() - np.array([0, box_height / 2, 0]),
                 end=func_obj_ls[idx].get_center() + np.array([0, box_height / 2, 0]),
                 color=RED_COLOR, stroke_width=1
                 ).add_updater(lambda m, j=idx: m.become(
                    Line(start=gene_obj_ls[j].get_center() - np.array([0, box_height / 2, 0]),
                         end=func_obj_ls[j].get_center() + np.array([0, box_height / 2, 0]),
                         color=RED_COLOR, stroke_width=1
                         )
                    if gene_obj_ls[j].get_y() - func_obj_ls[j].get_y() > box_height else
                    Line(stroke_opacity=0, fill_opacity=0)  # 不显示线条
            ))  # 基因与目标函数之间的连线
            for idx in range(len(gene_obj_ls))]
        self.add(*line_15_obj_ls)

        # 亲本
        func_val_ndarray = np.array(func_val)
        uniform_step1 = (func_val_ndarray - func_val_ndarray.min()) / (func_val_ndarray.max() - func_val_ndarray.min())
        uniform_step2 = uniform_step1 / uniform_step1.sum()
        angles = uniform_step2 * TAU
        angles_offset = np.cumsum((0, *angles[:-1]))
        choose_angles_ls = [2740, 5760, 7420, 9390, 12170]
        def judge_gene(degree: float):
            # 判断弧度对应的个体，返回个体索引
            radian = degree*DEGREES
            radian %= 2*np.pi
            return np.argmax(np.cumsum(angles)>radian)
        parent_ls = [gene[judge_gene(choose_angle)] for choose_angle in choose_angles_ls]
        parent_obj_ls = [VGroup(
            Rectangle(width=box_width, height=box_height, fill_opacity=0, stroke_color=BLUE_COLOR, stroke_width=1),
            Text(f"{parent_ls[idx]}", font_size=font_size, color=BLUE_COLOR,
                 font="Times New Roman")
        ) for idx in range(len(gene))]
        parent_obj = VGroup(*parent_obj_ls).arrange(RIGHT, buff=box_buff)
        parent_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1,
                      stroke_color=BLUE_COLOR, stroke_width=1, stroke_opacity=1),
            Text("亲本", font_size=font_size, color=BLUE_COLOR, font="Dream Han Serif CN")
        ).next_to(parent_obj, LEFT, buff=lab_buff)
        parent_all_obj = VGroup(parent_obj, parent_lab).shift(RIGHT * move_right)
        self.add(parent_all_obj)
        self.wait()

        """
        1-现在  我们选择了5个亲本用于产生子一代
        """
        # 亲本快闪
        parent_fill_obj_ls = [VGroup(
            Rectangle(width=box_width, height=box_height,
                      fill_color=YELLOW_COLOR, fill_opacity=0.8, stroke_color=BLUE_COLOR, stroke_width=1),
            Text(f"{parent_ls[idx]}", font_size=font_size, color=BLUE_COLOR,
                 font="Times New Roman")
        ) for idx in range(len(gene))]
        parent_fill_obj = VGroup(*parent_fill_obj_ls).arrange(RIGHT, buff=box_buff).move_to(parent_obj.get_center())

        for _ in range(2):
            self.add(parent_fill_obj)
            self.wait(flash_time)
            self.remove(parent_fill_obj)
            self.wait(flash_time)
        self.wait()

        """
        2-杂交
        """
        parent_color_ls = [["#aaffaa", "#005f00"], ["#fff1aa", "#5f4f00"]]  #[0-绿色，1-金色][0-浅色，1-深色]
        np.random.seed(1)   # 设置随机数种子，使每次测试产生的随机数相同
        rate_cross = 0.4
        rate_mutation = 0.3
        down_dist = 3
        filial_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1,
                      stroke_color=BLUE_COLOR, stroke_width=1, stroke_opacity=1),
            MathTex(r"\text{F}_{1}", font_size=font_size+2, color=BLUE_COLOR)
        ).move_to(down_dist * DOWN + np.array([gene_lab.get_x(), 0, 0]))    # 子代左侧标签
        filial_obj_ls = []  # 子代对象存储
        filial_val = []     # 子代基因
        # 代码：交叉
        code_cross = VGroup(
            Code(code_string=r'''
# python =====================================================
import numpy as np
cross(parent_ls)
def cross(parent_ls: list[tuple[int]],      # 亲本
          rate: float=0.4                   # 交叉率
          ) -> list[tuple[int]]:            # 交叉操作
    parents = np.array(parent_ls)   # 转为ndarray对象便于操作
    random_ls = np.random.uniform(
        low=0, high=1, size=parents.shape) < rate   # 随机数
    res = parents[:-1,:] * (1 - random_ls[:-1,:]) +
          parents[1:,:] * random_ls[:-1,:]
    return list(map(tuple, res))''', language="python", **code_kargs),
            Code(code_string=r'''
% matlab =====================================================
%% 交叉
random_ls = randi([0,1],size(parent_ls)) < rate;   % 随机数
cross_res = parent_ls(1:end-1,:).*(~random_ls(1:end-1,:)) +...
            parent_ls(2:end,:).*random_ls(1:end-1,:);''', language='matlab', **code_kargs)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).to_edge(RIGHT, buff=0.1).shift(DOWN * code_down)
        # 代码：突变
        code_mutation = VGroup(
            Code(code_string=r'''
# python =====================================================
import numpy as np
filial_ls = mutation(cross(parent_ls))
def cross(parent_ls: list[tuple[int]],      # 亲本
          rate: float=0.4                   # 交叉率
          ) -> list[tuple[int]]:            # 交叉操作
    parents = np.array(parent_ls)   # 转为ndarray对象便于操作
    random_ls = np.random.uniform(
        low=0, high=1, size=parents.shape) < rate   # 随机数
    res = parents[:-1,:] * (1 - random_ls[:-1,:]) +
          parents[1:,:] * random_ls[:-1,:]
    return list(map(tuple, res))
def mutation(parent_ls: list[tuple[int]],   # 亲本
             rate: float=0.4                # 突变率
             ) -> list[tuple[int]]:         # 突变操作
    parents = np.array(parent_ls)
    random_ls = np.random.uniform(
        low=0, high=1, size=parents.shape) < rate   # 随机数
    res = np.bitwise_xor(parents, random_ls)
    return list(map(tuple, res))''', language="python", **code_kargs),
            Code(code_string=r'''
% matlab =====================================================
%% 交叉
random_ls = randi([0,1],size(parent_ls)) < rate;   % 随机数
cross_res = parent_ls(1:end-1,:).*(~random_ls(1:end-1,:)) +...
            parent_ls(2:end,:).*random_ls(1:end-1,:);
%% 突变
random_ls = randi([0,1],size(parent_ls)) < rate;   % 随机数
filial_ls = double(xor(parent_ls, random_ls));''', language='matlab', **code_kargs)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).to_edge(RIGHT, buff=0.1).shift(DOWN * code_down)
        # 代码：精英保留
        code_elite = VGroup(
            Code(code_string=r'''
# python =====================================================
import numpy as np
filial_ls = mutation(cross(parent_ls))
filial_ls.append(population_ls[np.argmax(fit_ls)])  # 精英保留
def cross(parent_ls: list[tuple[int]],      # 亲本
          rate: float=0.4                   # 交叉率
          ) -> list[tuple[int]]:            # 交叉操作
    parents = np.array(parent_ls)   # 转为ndarray对象便于操作
    random_ls = np.random.uniform(
        low=0, high=1, size=parents.shape) < rate   # 随机数
    res = parents[:-1,:] * (1 - random_ls[:-1,:]) +
          parents[1:,:] * random_ls[:-1,:]
    return list(map(tuple, res))
def mutation(parent_ls: list[tuple[int]],   # 亲本
             rate: float=0.4                # 突变率
             ) -> list[tuple[int]]:         # 突变操作
    parents = np.array(parent_ls)
    random_ls = np.random.uniform(
        low=0, high=1, size=parents.shape) < rate   # 随机数
    res = np.bitwise_xor(parents, random_ls)
    return list(map(tuple, res))''', language="python", **code_kargs),
            Code(code_string=r'''
% matlab =====================================================
%% 交叉
random_ls = randi([0,1],size(parent_ls)) < rate;   % 随机数
cross_res = parent_ls(1:end-1,:).*(~random_ls(1:end-1,:)) +...
            parent_ls(2:end,:).*random_ls(1:end-1,:);
%% 突变
random_ls = randi([0,1],size(parent_ls)) < rate;   % 随机数
filial_ls = double(xor(parent_ls, random_ls));
%% 精英保留
[~, elite_idx] = max(fit_ls);
filial_ls(end+1,:) = parent_ls(elite_idx, :);''', language='matlab', **code_kargs)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).to_edge(RIGHT, buff=0.1).shift(DOWN * code_down)
        for first_idx in range(len(gene)-1):
            # 亲本放大
            parent_now = VGroup(
                *[VGroup(Rectangle(width=box_width, height=box_height,
                                   fill_color=parent_color_ls[idx][0], fill_opacity=1,
                                   stroke_color=parent_color_ls[idx][1], stroke_width=1, stroke_opacity=1),
                         Text(f"{parent_ls[first_idx+idx]}", color=parent_color_ls[idx][1],
                              font_size=font_size, font="Times New Roman")
                         )
                  for idx in (0, 1)]
            ).arrange(RIGHT, buff=box_width+2*box_buff).shift(RIGHT*move_right+0.5*DOWN)
            # 杂交符号
            if first_idx == 0:
                x_symbol = Text("×", color=RED_COLOR, font_size=font_size).move_to(parent_now.get_center())
            # 交叉
            cross_random = np.random.uniform(low=0, high=1, size=8) < rate_cross
            cross_res = ''.join([parent_ls[first_idx][idx] if cross_random[idx] else parent_ls[first_idx+1][idx]
                                 for idx in range(8)])
            cross_res_obj = VGroup(
                *[VGroup(
                    Rectangle(width=box_width/8, height=box_height,
                              fill_color=parent_color_ls[0][0] if cross_random[idx] else parent_color_ls[1][0],
                              fill_opacity=1, stroke_opacity=0),
                    Text(f"{cross_res[idx]}",
                         color=parent_color_ls[0][1] if cross_random[idx] else parent_color_ls[1][1],
                         font_size=font_size,
                         font="Times New Roman")
                ) for idx in range(8)]
            ).arrange(RIGHT, buff=0).shift(RIGHT*move_right+1.4*DOWN)
            if first_idx == 0:
                cross_arrow = Arrow(start=parent_now.get_center()+DOWN*box_height/2*0.8,
                                    end=cross_res_obj.get_center()+UP*box_height/2+0.1*UP,
                                    buff=0,
                                    stroke_color=RED_COLOR,
                                    stroke_width=1,
                                    tip_shape=StealthTip)
                cross_lab = VGroup(
                    Text("交叉率", color=BLUE_COLOR, font_size=font_size-2, font="Dream Han Serif CN"),
                    MathTex(f"P={rate_cross:.1f}", font_size=font_size, color=BLUE_COLOR)
                ).arrange(DOWN, buff=0.05).next_to(cross_arrow, LEFT, buff=0.1)
            # 突变
            mutation_random = np.random.uniform(low=0, high=1, size=8) < rate_mutation
            mutation_res = ''.join([str(1-int(cross_res[idx]) if mutation_random[idx] else cross_res[idx])
                                 for idx in range(8)])
            mutation_res_obj = VGroup(
                *[VGroup(
                    Rectangle(width=box_width / 8, height=box_height,
                              fill_color=RED_COLOR if mutation_random[idx] else YELLOW_COLOR,
                              fill_opacity=0.2, stroke_opacity=0),
                    Text(f"{mutation_res[idx]}",
                         color=RED_COLOR,
                         font_size=font_size,
                         font="Times New Roman")
                ) for idx in range(8)]
            ).arrange(RIGHT, buff=0).shift(RIGHT * move_right + 2.3 * DOWN)
            filial_val.append(mutation_res)
            if first_idx == 0:
                mutation_arrow = Arrow(start=cross_res_obj.get_center() + DOWN * box_height / 2,
                                    end=mutation_res_obj.get_center()+UP*box_height/2+0.1*UP,
                                    buff=0,
                                    stroke_color=RED_COLOR,
                                    stroke_width=1,
                                    tip_shape=StealthTip)
                mutation_lab = VGroup(
                    Text("突变率", color=BLUE_COLOR, font_size=font_size - 2, font="Dream Han Serif CN"),
                    MathTex(f"P={rate_mutation:.1f}", font_size=font_size, color=BLUE_COLOR)
                ).arrange(DOWN, buff=0.05).next_to(mutation_arrow, LEFT, buff=0.1)
            filial_obj_ls.append(
                VGroup(
                    Rectangle(width=box_width, height=box_height, fill_color=WHITE, fill_opacity=1, stroke_color=BLUE_COLOR,
                              stroke_width=1),
                    Text(f"{mutation_res}", font_size=font_size, color=BLUE_COLOR,
                         font="Times New Roman")).move_to(down_dist*DOWN + np.array([gene_obj_ls[first_idx].get_x(), 0, 0]))
            )
            # manim动画
            if first_idx==0:
                self.play(
                    FadeIn(parent_fill_obj_ls[first_idx]),
                    FadeIn(parent_fill_obj_ls[first_idx + 1])
                )  # 亲本填充
                self.wait(0.2)
                self.play(
                    TransformFromCopy(parent_fill_obj_ls[first_idx], parent_now[0]),
                    TransformFromCopy(parent_fill_obj_ls[first_idx + 1], parent_now[1]),
                    Write(x_symbol)
                )   # 杂交符号
                self.wait(0.2)
                self.play(Write(cross_arrow), Write(cross_lab)) # 交叉
                self.play(Create(cross_res_obj), Write(code_cross))    # 交叉结果
                self.wait()
                self.play(Write(mutation_arrow), Write(mutation_lab))   # 突变
                self.play(Create(mutation_res_obj), ReplacementTransform(code_cross, code_mutation)) # 突变结果
                self.play(Write(filial_lab))    # 书写子代标签
                self.play(TransformFromCopy(mutation_res_obj, filial_obj_ls[-1]))   # 子代结果转移
            else:
                aft_all_obj = VGroup(parent_now, cross_res_obj, mutation_res_obj)
                for _ in range(1):
                    self.remove(bef_all_obj)
                    self.add(aft_all_obj)
                    self.wait(flash_time)
                    self.remove(aft_all_obj)
                    self.add(bef_all_obj)
                    self.wait(flash_time)
                self.remove(bef_all_obj)
                self.add(aft_all_obj)
                self.wait(0.2)
                self.play(TransformFromCopy(mutation_res_obj, filial_obj_ls[-1]))  # 子代结果转移
            bef_all_obj = VGroup(parent_now, cross_res_obj, mutation_res_obj)
            self.remove(parent_now, cross_res_obj, mutation_res_obj)
            self.add(bef_all_obj)   # 偷梁换柱，防止引用出错
            self.wait(0.5)
            if first_idx < len(gene)-2:
                self.play(FadeOut(parent_fill_obj_ls[first_idx]),
                          FadeIn(parent_fill_obj_ls[first_idx+2]),
                          run_time=0.5)
            else:
                self.play(FadeOut(parent_fill_obj_ls[first_idx]),
                          FadeOut(parent_fill_obj_ls[first_idx+1]),
                          FadeOut(VGroup(x_symbol, bef_all_obj, cross_arrow, cross_lab, mutation_lab, mutation_arrow)),
                          runtime=0.5)
        self.wait()
        # 精英保留
        elite_gene_fill_obj = VGroup(
                Rectangle(width=box_width, height=box_height,
                          fill_color=YELLOW_COLOR, fill_opacity=1,
                          stroke_color=BLUE_COLOR, stroke_width=1),
                Text(f"{gene[np.argmax(func_val)]}", font_size=font_size, color=BLUE_COLOR,
                     font="Times New Roman")).move_to(
                gene_obj_ls[np.argmax(func_val)].get_center()
        )
        filial_val.append(gene[np.argmax(func_val)])
        for _ in range(2):
            self.add(elite_gene_fill_obj)
            self.wait(flash_time)
            self.remove(elite_gene_fill_obj)
            self.wait(flash_time)
        self.add(elite_gene_fill_obj)
        filial_obj_ls.append(
            VGroup(
                Rectangle(width=box_width, height=box_height, fill_color=WHITE, fill_opacity=1, stroke_color=BLUE_COLOR,
                          stroke_width=1),
                Text(f"{gene[np.argmax(func_val)]}", font_size=font_size, color=BLUE_COLOR,
                     font="Times New Roman")).move_to(
                down_dist * DOWN + np.array([gene_obj_ls[-1].get_x(), 0, 0]))
        )
        filial_obj = VGroup(*filial_obj_ls)
        filial_all_obj = VGroup(filial_lab, filial_obj)
        self.play(TransformFromCopy(elite_gene_fill_obj, filial_obj_ls[-1]),
                  ReplacementTransform(code_mutation, code_elite))
        self.wait()

        """
        3-子二代、子三代
        """
        self.play(filial_all_obj.animate.move_to(gene_all_obj.get_center()),
                  FadeOut(elite_gene_fill_obj))
        for generation in [2, 3, 4, 5]:
            ## 轮换
            self.remove(gene_all_obj)
            gene_all_obj = filial_all_obj
            self.remove(filial_all_obj)
            self.add(gene_all_obj)  # 依旧是偷梁换柱
            gene = filial_val.copy()
            ## 重新计算
            dec_val = [int(gene[idx], 2) for idx in range(len(gene))]
            res_val = [
                np.interp(dec_val[idx],
                          [0, 2 ** 8 - 1],
                          [-3, 3])
                for idx in range(len(gene))]
            func_val = [
                aim_func(res_val[idx])
                for idx in range(len(gene))]
            func_val_ndarray = np.array(func_val)
            uniform_step1 = (func_val_ndarray - func_val_ndarray.min()) / (func_val_ndarray.max() - func_val_ndarray.min())
            uniform_step2 = uniform_step1 / uniform_step1.sum()
            # 轮盘赌
            prefix_sum = np.cumsum(uniform_step2)  # 前缀和
            random_ls: np.ndarray = np.random.uniform(low=0, high=1, size=len(uniform_step2))  # 产生随机数
            parent_ls = []
            for random_num in random_ls:
                idx = np.argmax(prefix_sum >= random_num)  # prefix_sum第一个大于random_num的索引
                parent_ls.append(gene[idx])
            # 适应度与亲本修改
            bef_info_obj = VGroup(func_obj, parent_obj)
            func_obj_ls = [VGroup(
                Rectangle(width=box_width, height=box_height, fill_color=WHITE, fill_opacity=1, stroke_color=BLUE_COLOR, stroke_width=1),
                Text(f"{func_val[idx]:.2f}", font_size=font_size, color=BLUE_COLOR, font="Times New Roman")
            ) for idx in range(len(gene))]
            func_obj = VGroup(*func_obj_ls).arrange(RIGHT, buff=box_buff).move_to(bef_info_obj[0].get_center())
            parent_fill_obj_ls = [VGroup(
                Rectangle(width=box_width, height=box_height,
                          fill_color=WHITE, fill_opacity=1, stroke_color=BLUE_COLOR, stroke_width=1),
                Text(f"{parent_ls[idx]}", font_size=font_size, color=BLUE_COLOR,
                     font="Times New Roman")
            ) for idx in range(len(gene))]
            parent_obj = VGroup(*parent_fill_obj_ls).arrange(RIGHT, buff=box_buff).move_to(bef_info_obj[1].get_center())
            aft_info_obj = VGroup(func_obj, parent_obj)

            filial_lab = VGroup(
                Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1,
                          stroke_color=BLUE_COLOR, stroke_width=1, stroke_opacity=1),
                MathTex(r"\text{F}_"+f"{generation}", font_size=font_size + 2, color=BLUE_COLOR)
            ).move_to(down_dist * DOWN + np.array([gene_lab.get_x(), 0, 0]))  # 子代左侧标签
            mid_all_obj = VGroup()
            filial_obj_ls = []  # 子代对象
            filial_val = []     # 子代基因
            all_obj_now = VGroup()
            for first_idx in range(len(gene)-1):
                # 交叉
                cross_random = np.random.uniform(low=0, high=1, size=8) < rate_cross
                cross_res = ''.join([parent_ls[first_idx][idx] if cross_random[idx] else parent_ls[first_idx + 1][idx]
                                     for idx in range(8)])
                cross_res_obj = VGroup(
                    *[VGroup(
                        Rectangle(width=box_width / 8, height=box_height,
                                  fill_color=parent_color_ls[0][0] if cross_random[idx] else parent_color_ls[1][0],
                                  fill_opacity=1, stroke_opacity=0),
                        Text(f"{cross_res[idx]}",
                             color=parent_color_ls[0][1] if cross_random[idx] else parent_color_ls[1][1],
                             font_size=font_size,
                             font="Times New Roman")
                    ) for idx in range(8)]
                ).arrange(RIGHT, buff=0).move_to(np.array([
                    (gene_obj_ls[first_idx].get_x()+gene_obj_ls[first_idx+1].get_x())/2,
                    0, 0])).shift(1.1 * DOWN)
                # 突变
                mutation_random = np.random.uniform(low=0, high=1, size=8) < rate_mutation
                mutation_res = ''.join([str(1 - int(cross_res[idx]) if mutation_random[idx] else cross_res[idx])
                                        for idx in range(8)])
                mutation_res_obj = VGroup(
                    *[VGroup(
                        Rectangle(width=box_width / 8, height=box_height,
                                  fill_color=RED_COLOR if mutation_random[idx] else YELLOW_COLOR,
                                  fill_opacity=0.2, stroke_opacity=0),
                        Text(f"{mutation_res[idx]}",
                             color=RED_COLOR,
                             font_size=font_size,
                             font="Times New Roman")
                    ) for idx in range(8)]
                ).arrange(RIGHT, buff=0).move_to(np.array([
                    (gene_obj_ls[first_idx].get_x()+gene_obj_ls[first_idx+1].get_x())/2,
                    0, 0])).shift(2 * DOWN)
                # 下代
                filial_obj_ls.append(
                    VGroup(
                        Rectangle(width=box_width, height=box_height, fill_color=WHITE, fill_opacity=1, stroke_color=BLUE_COLOR,
                                  stroke_width=1),
                        Text(f"{mutation_res}", font_size=font_size, color=BLUE_COLOR,
                             font="Times New Roman")).move_to(
                        down_dist * DOWN + np.array([gene_obj_ls[first_idx].get_x(), 0, 0]))
                )
                filial_val.append(mutation_res)
                # 箭头
                arrow_obj = VGroup(
                    Line(
                        start=(parent_obj_ls[first_idx].get_center() + parent_obj_ls[
                            first_idx + 1].get_center()) / 2 + DOWN * box_height / 2,
                        end=cross_res_obj.get_center() + UP * box_height / 2,
                        color=RED_COLOR,
                        stroke_width=1,
                    ),
                    Line(
                        start=cross_res_obj.get_center() + DOWN * box_height / 2,
                        end=mutation_res_obj.get_center() + UP * box_height / 2,
                        color=RED_COLOR,
                        stroke_width=1,
                    ),
                    Line(
                        start=mutation_res_obj.get_center() + DOWN * box_height / 2,
                        end=filial_obj_ls[-1].get_center() + UP * box_height / 2,
                        color=RED_COLOR,
                        stroke_width=1,
                    ),
                )
                mid_all_obj += VGroup(cross_res_obj, mutation_res_obj, arrow_obj)
            # 精英
            elite_gene_fill_obj = VGroup(
                Rectangle(width=box_width, height=box_height,
                          fill_color=YELLOW_COLOR, fill_opacity=1,
                          stroke_color=BLUE_COLOR, stroke_width=1),
                Text(f"{gene[np.argmax(func_val)]}", font_size=font_size, color=BLUE_COLOR,
                     font="Times New Roman")).move_to(
                gene_obj_ls[np.argmax(func_val)].get_center()
            )
            filial_val.append(gene[np.argmax(func_val)])
            filial_obj_ls.append(
                VGroup(
                    Rectangle(width=box_width, height=box_height, fill_color=WHITE, fill_opacity=1, stroke_color=BLUE_COLOR,
                              stroke_width=1),
                    Text(f"{gene[np.argmax(func_val)]}", font_size=font_size, color=BLUE_COLOR,
                         font="Times New Roman")).move_to(
                    down_dist * DOWN + np.array([gene_obj_ls[-1].get_x(), 0, 0]))
            )

            filial_obj = VGroup(*filial_obj_ls)
            filial_all_obj = VGroup(filial_lab, filial_obj)
            all_obj_now += VGroup(mid_all_obj, elite_gene_fill_obj)
            ## manim动画
            # 适应度、亲本快闪
            self.wait(0.5)
            self.add(aft_info_obj)
            self.remove(bef_info_obj)
            self.wait(flash_time)
            self.remove(aft_info_obj)
            self.add(bef_info_obj)
            self.wait(flash_time)
            self.add(aft_info_obj)
            self.remove(bef_info_obj)
            # 下方信息快闪
            self.add(all_obj_now, filial_all_obj)
            self.wait(flash_time)
            self.remove(all_obj_now, filial_all_obj)
            self.wait(flash_time)
            self.add(all_obj_now, filial_all_obj)

            self.wait()
            self.play(filial_all_obj.animate.move_to(gene_all_obj.get_center()),
                      FadeOut(all_obj_now))
