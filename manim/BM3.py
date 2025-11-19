from manim import *
import numpy as np
from config import *


class BM3(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        lab_buff = 0.2    # 左侧标签间距
        move_right = -2.5    # 整体向右偏移量
        box_height = 0.4    # 标签高度
        box_width = 1.15       # 标签宽度
        box_buff = 0.1      # 标签间距
        label_width = 1.1     # 左侧标签宽度
        font_size = 18
        code_down = 0.73       # 代码中心下移距离

        """
        0-目标函数与边界
        """
        aim_func_tex_obj = MathTex(r"\text{max }y=f(x)=(x^2-1.5) \sum\limits_{n=1}^{100}0.5^n "
                                   r"\cos(7^n \pi x)+\frac{x}{2}", font_size=30)
        aim_func_tex_obj.set_color(BLUE_COLOR)
        aim_func_tex_obj.to_edge(UP, buff=0.5)

        boundary_tex_obj = MathTex(r"\text{s.t. } -3 \le x \le 3", font_size=30)
        boundary_tex_obj.set_color(BLUE_COLOR)
        boundary_tex_obj.next_to(aim_func_tex_obj, DOWN)
        self.add(aim_func_tex_obj, boundary_tex_obj)

        """
        1-展示5个个体
        """
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
        gene_all_obj = VGroup(gene_obj, gene_lab)
        self.play(*[Write(gene_obj_ls[idx]) for idx in range(len(gene_obj_ls))], Write(gene_lab))
        self.wait()
        code_bef = VGroup(
            Code(code_string=r'''# python =====================================================
import numpy as np
population_ls = [
    tuple(np.random.randint(0, 1+1) for _ in range(8))
for idx in range(5)]''', language="python", **code_kargs),
            Code(code_string=r'''% matlab =====================================================
f=randi([0, 1],5 ,8);    % 随机获得初始种群''', language='matlab', **code_kargs)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).to_edge(RIGHT, buff=0.1).shift(DOWN*code_down)
        self.play(gene_all_obj.animate.shift(RIGHT*move_right),
                  Write(code_bef))
        self.wait()

        """
        2-展示转十进制
        """
        dec_val = [int(gene[idx], 2) for idx in range(len(gene))]
        dec_obj_ls = [VGroup(
                Rectangle(width=box_width, height=box_height, fill_opacity=0, stroke_color=YELLOW_COLOR, stroke_width=1),
                Text(f"{dec_val[idx]:03d}", font_size=font_size, color=BLUE_COLOR, font="Times New Roman")
            ) for idx in range(len(dec_val))
        ]
        dec_obj = VGroup(*dec_obj_ls)
        dec_obj.arrange(RIGHT, buff=box_buff)
        dec_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1),
            Text("十进制", font_size=font_size, color=BLUE_COLOR, font="Dream Han Serif CN")
        ).next_to(dec_obj, LEFT, buff=lab_buff)
        dec_all_obj = VGroup(dec_obj, dec_lab).shift(RIGHT*move_right)
        line12_obj_ls = [
            Line().add_updater(lambda m, j=idx: m.become(
                Line(start=gene_obj_ls[j].get_center()-np.array([0, box_height/2, 0]),
                     end=dec_obj_ls[j].get_center()+np.array([0, box_height/2, 0]),
                     color=RED_COLOR, stroke_width=1
                     )
                if gene_obj_ls[j].get_y() - dec_obj_ls[j].get_y() > box_height else
                Line(stroke_opacity=0, fill_opacity=0)    # 不显示线条
            ))  # 连接基因与十进制之间的红线
        for idx in range(len(gene_obj_ls))]
        code_aft = VGroup(
            Code(code_string=r'''# python =====================================================
import numpy as np
population_ls = [
    tuple(np.random.randint(0, 1+1) for _ in range(8))
for idx in range(5)]
decimalism_ls = [int(''.join(map(str, population_ls[idx])), 2)
                 for idx in range(5)] # 转十进制''', language="python", **code_kargs),
            Code(code_string=r'''% matlab =====================================================
population_ls = randi([0, 1], 5, 8);    % 随机获得初始种群
%% 转十进制
for idx = 1:5
    decimalism_ls(idx) = 0;
    for j = 1:8
        decimalism_ls(idx) = decimalism_ls(idx) +...
        population_ls(idx, 9-j)*(2^(j-1));    % 加权和
    end
end''', language='matlab', **code_kargs)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).to_edge(RIGHT, buff=0.1).shift(DOWN*code_down)
        self.play(
            gene_all_obj.animate.shift(UP*0.8),
            *[Write(dec_obj_ls[idx]) for idx in range(len(dec_obj_ls))],
            *[Write(line12_obj_ls[idx]) for idx in range(len(line12_obj_ls))],
            Write(dec_lab),
            ReplacementTransform(code_bef, code_aft)
        )
        code_bef = code_aft
        self.wait()

        """
        3-映射至定义域
        """
        res_val = [
            np.interp(dec_val[idx],
                      [0,2**8-1],
                      [-3,3])
        for idx in range(len(gene_obj_ls))]
        bef_number_axes = NumberLine(
            x_range=[0, 2**8-1, 2**8-1],
            length=4,
            include_numbers=True,
            font_size=font_size-2
        ).set_color(BLUE_COLOR)
        aft_number_axes = NumberLine(
            x_range=[-3, 3, 3],
            length=4,
            include_numbers=True,
            font_size=font_size-2
        ).set_color(BLUE_COLOR)
        number_axes = VGroup(bef_number_axes, aft_number_axes)
        number_axes.arrange(DOWN, buff=0.5)
        aft_number_axes.shift(np.array([
            bef_number_axes.n2p(2**7)[0]-aft_number_axes.n2p(0)[0],0,0
        ]))
        number_axes.move_to(DOWN*2+RIGHT*move_right)
        number_axes_lab = VGroup(
            Text("基因编码范围", font_size=font_size-2, color=BLUE_COLOR, font="Dream Han Serif CN"
                 ).next_to(bef_number_axes, LEFT, buff=1),
            Text("定义域范围", font_size=font_size-2, color=BLUE_COLOR, font="Dream Han Serif CN"
                 ).next_to(aft_number_axes, LEFT, buff=0.88)
        ).shift(0.1*UP)
        number_all_obj = VGroup(number_axes, number_axes_lab)
        self.play(Write(bef_number_axes), Write(aft_number_axes), Write(number_axes_lab))
        self.wait(0.1)

        short_line_length = 0.1
        line23_obj_ls = [
            VGroup(
                Line().add_updater(lambda m, j=idx: m.become(
                    Line(start=dec_obj_ls[j].get_center() - np.array([0, box_height / 2, 0]),
                         end=dec_obj_ls[j].get_center() - np.array([0, box_height / 2+short_line_length, 0]),
                         color=RED_COLOR, stroke_width=1
                         )
                )),  # 上侧短线
                Line().add_updater(lambda m, j=idx: m.become(
                    Line(start=dec_obj_ls[j].get_center() - np.array([0, box_height / 2 + short_line_length, 0]),
                         end=bef_number_axes.n2p(dec_val[j]),
                         color=RED_COLOR, stroke_width=1
                         )
                )),  # 中间斜线
                Line().add_updater(lambda m, j=idx: m.become(
                    Line(start=bef_number_axes.n2p(dec_val[j]),
                         end=aft_number_axes.n2p(res_val[j]),
                         color=RED_COLOR, stroke_width=1
                         )
                )),  # 竖轴跨线
            )   # 十进制与映射之间的连线
        for idx in range(len(gene_obj_ls))]

        res_obj_ls = [VGroup(
            Rectangle(width=box_width, height=box_height, fill_opacity=0, stroke_color=YELLOW_COLOR, stroke_width=1),
            Text(f"{res_val[idx]:.2f}", font_size=font_size, color=BLUE_COLOR, font="Times New Roman")
        ) for idx in range(len(gene))]
        res_obj = VGroup(*res_obj_ls)
        res_obj.arrange(RIGHT, buff=box_buff)
        res_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1),
            Text("决策变量", font_size=font_size, color=BLUE_COLOR, font="Dream Han Serif CN")
        ).next_to(res_obj, LEFT, buff=lab_buff)
        res_all_obj = VGroup(res_obj, res_lab).shift(RIGHT * move_right + 3*DOWN)

        line34_obj_ls = [
            VGroup(
                Line().add_updater(lambda m, j=idx: m.become(
                    Line(start=aft_number_axes.n2p(res_val[j]),
                         end=res_obj_ls[j].get_center() + np.array([0, box_height / 2+short_line_length, 0]),
                         color=RED_COLOR, stroke_width=1
                         )
                )),  # 中间斜线
                Line().add_updater(lambda m, j=idx: m.become(
                    Line(start=res_obj_ls[j].get_center() + np.array([0, box_height / 2+short_line_length, 0]),
                         end=res_obj_ls[j].get_center() + np.array([0, box_height / 2, 0]),
                         color=RED_COLOR, stroke_width=1
                         )
                )),  # 末尾短线
            ) for idx in range(len(gene_obj_ls))]
        code_aft = VGroup(
            Code(code_string=r'''# python =====================================================
import numpy as np
population_ls = [
    tuple(np.random.randint(0, 1+1) for _ in range(8))
for idx in range(5)]
decimalism_ls = [int(''.join(map(str, population_ls[idx])), 2)
                 for idx in range(5)] # 转十进制
decision_ls = [np.interp(   # 映射至定义域
    decimalism_ls[idx], # 十进制的值
    [0, 2**8-1],        # 基因可编码的范围
    [-3,3]              # 定义域
) for idx in range(5)]''', language="python", **code_kargs),
            Code(code_string=r'''% matlab =====================================================
population_ls = randi([0, 1], 5, 8);    % 随机获得初始种群
%% 转十进制
for idx = 1:5
    decimalism_ls(idx) = 0;
    for j = 1:8
        decimalism_ls(idx) = decimalism_ls(idx) +...
        population_ls(idx, 9-j)*(2^(j-1));    % 加权和
    end
end
%% 映射至定义域
[Min, Max] = deal(-3, 3);
decision_ls = Min+decimalism_ls.*(Max-Min)./(2^8-1);''', language='matlab', **code_kargs)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).to_edge(RIGHT, buff=0.1).shift(DOWN * code_down)
        self.play(*[Write(line23_obj_ls[idx]) for idx in range(len(gene_obj_ls))])
        self.play(number_all_obj.animate.shift(UP*0.4),
                  *[Write(res_obj_ls[idx]) for idx in range(len(res_obj_ls))], Write(res_lab),
                  *[Write(line34_obj_ls[idx]) for idx in range(len(gene_obj_ls))],
                  ReplacementTransform(code_bef, code_aft))
        code_bef = code_aft
        self.wait()

        """
        4-代入目标函数
        """
        bef_all_obj = VGroup(gene_all_obj, dec_all_obj, number_all_obj, res_all_obj)
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
        func_all_obj = VGroup(func_obj, func_lab).shift(RIGHT * move_right + 3*DOWN)

        line45_obj_ls = [
            Line().add_updater(lambda m, j=idx: m.become(
                Line(start=res_obj_ls[j].get_center() - np.array([0, box_height / 2, 0]),
                     end=func_obj_ls[j].get_center() + np.array([0, box_height / 2, 0]),
                     color=RED_COLOR, stroke_width=1
                     )
                if res_obj_ls[j].get_y() - func_obj_ls[j].get_y() > box_height else
                Line(stroke_opacity=0, fill_opacity=0)  # 不显示线条
            ))  # 决策变量与目标函数之间的连线
        for idx in range(len(gene_obj_ls))]
        code_aft = VGroup(
            Code(code_string=r'''# python =====================================================
import numpy as np
population_ls = [
    tuple(np.random.randint(0, 1+1) for _ in range(8))
for idx in range(5)]
decimalism_ls = [int(''.join(map(str, population_ls[idx])), 2)
                 for idx in range(5)] # 转十进制
decision_ls = [np.interp(   # 映射至定义域
    decimalism_ls[idx], # 十进制的值
    [0, 2**8-1],        # 基因可编码的范围
    [-3,3]              # 定义域
) for idx in range(5)]
aim_func = lambda x: (x ** 2 - 1.5) * np.sum(
    [0.5 ** n * np.cos(7 ** n * np.pi * x)
     for n in np.arange(0, 100 + 1, 1)]) + x / 2 # 目标函数
fit_ls = [aim_func(decision_ls[idx])
          for idx in range(5)] # 适应度''', language="python", **code_kargs),
            Code(code_string=r'''% matlab =====================================================
population_ls = randi([0, 1], 5, 8);    % 随机获得初始种群
%% 转十进制
for idx = 1:5
    decimalism_ls(idx) = 0;
    for j = 1:8
        decimalism_ls(idx) = decimalism_ls(idx) +...
        population_ls(idx, 9-j)*(2^(j-1));    % 加权和
    end
end
%% 映射至定义域
[Min, Max] = deal(-3, 3);
decision_ls = Min+decimalism_ls.*(Max-Min)./(2^8-1);
n = 1:100;  % 迭代变量
aim_func = @(x) (x.^2-1.5).*...
            sum(0.5.^n'.*cos(7.^n'*pi.*x))+x/2; % 目标函数
fit_ls = aim_func(decision_ls);                 % 适应度''', language='matlab', **code_kargs)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).to_edge(RIGHT, buff=0.1).shift(DOWN * code_down)
        self.play(bef_all_obj.animate.shift(UP*0.8),
                  *[Write(func_obj_ls[idx]) for idx in range(len(func_obj_ls))], Write(func_lab),
                  *[Write(line45_obj_ls[idx]) for idx in range(len(line45_obj_ls))],
                  ReplacementTransform(code_bef, code_aft))
        code_bef = code_aft
        self.wait()
