from manim import *
import numpy as np
from config import *
from custom_class import Pie



class BM4(Scene):
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
        0-只保留BM3镜头的基因和适应度
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
        gene_all_obj = VGroup(gene_obj, gene_lab).shift(RIGHT*move_right)
        self.add(gene_all_obj)

        # 十进制
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
        dec_all_obj = VGroup(dec_obj, dec_lab).shift(RIGHT * move_right)
        line12_obj_ls = [
            Line().add_updater(lambda m, j=idx: m.become(
                Line(start=gene_obj_ls[j].get_center() - np.array([0, box_height / 2, 0]),
                     end=dec_obj_ls[j].get_center() + np.array([0, box_height / 2, 0]),
                     color=RED_COLOR, stroke_width=1
                     )
                if gene_obj_ls[j].get_y() - dec_obj_ls[j].get_y() > box_height else
                Line(stroke_opacity=0, fill_opacity=0)  # 不显示线条
            ))  # 连接基因与十进制之间的红线
            for idx in range(len(gene_obj_ls))]
        self.add(*line12_obj_ls, dec_all_obj)
        gene_all_obj.shift(UP*0.8)

        # 映射
        res_val = [
            np.interp(dec_val[idx],
                      [0, 2 ** 8 - 1],
                      [-3, 3])
            for idx in range(len(gene_obj_ls))]
        bef_number_axes = NumberLine(
            x_range=[0, 2 ** 8 - 1, 2 ** 8 - 1],
            length=4,
            include_numbers=True,
            font_size=font_size - 2
        ).set_color(BLUE_COLOR)
        aft_number_axes = NumberLine(
            x_range=[-3, 3, 3],
            length=4,
            include_numbers=True,
            font_size=font_size - 2
        ).set_color(BLUE_COLOR)
        number_axes = VGroup(bef_number_axes, aft_number_axes)
        number_axes.arrange(DOWN, buff=0.5)
        aft_number_axes.shift(np.array([
            bef_number_axes.n2p(2 ** 7)[0] - aft_number_axes.n2p(0)[0], 0, 0
        ]))
        number_axes.move_to(DOWN * 2 + RIGHT * move_right)
        number_axes_lab = VGroup(
            Text("基因编码范围", font_size=font_size - 2, color=BLUE_COLOR, font="Dream Han Serif CN"
                 ).next_to(bef_number_axes, LEFT, buff=1),
            Text("定义域范围", font_size=font_size - 2, color=BLUE_COLOR, font="Dream Han Serif CN"
                 ).next_to(aft_number_axes, LEFT, buff=0.88)
        ).shift(0.1 * UP)
        number_all_obj = VGroup(number_axes, number_axes_lab)
        self.add(number_all_obj)

        short_line_length = 0.1
        line23_obj_ls = [
            VGroup(
                Line().add_updater(lambda m, j=idx: m.become(
                    Line(start=dec_obj_ls[j].get_center() - np.array([0, box_height / 2, 0]),
                         end=dec_obj_ls[j].get_center() - np.array([0, box_height / 2 + short_line_length, 0]),
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
            )  # 十进制与映射之间的连线
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
        res_all_obj = VGroup(res_obj, res_lab).shift(RIGHT * move_right + 3 * DOWN)

        line34_obj_ls = [
            VGroup(
                Line().add_updater(lambda m, j=idx: m.become(
                    Line(start=aft_number_axes.n2p(res_val[j]),
                         end=res_obj_ls[j].get_center() + np.array([0, box_height / 2 + short_line_length, 0]),
                         color=RED_COLOR, stroke_width=1
                         )
                )),  # 中间斜线
                Line().add_updater(lambda m, j=idx: m.become(
                    Line(start=res_obj_ls[j].get_center() + np.array([0, box_height / 2 + short_line_length, 0]),
                         end=res_obj_ls[j].get_center() + np.array([0, box_height / 2, 0]),
                         color=RED_COLOR, stroke_width=1
                         )
                )),  # 末尾短线
            ) for idx in range(len(gene_obj_ls))]

        self.add(*line23_obj_ls, res_all_obj, *line34_obj_ls)
        number_all_obj.shift(UP*0.4)

        # 目标函数
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
        func_all_obj = VGroup(func_obj, func_lab).shift(RIGHT * move_right + 3 * DOWN)

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
        self.add(func_all_obj, *line45_obj_ls)
        bef_all_obj.shift(UP*0.8) # 此时线条未被加载，需要等待下一个play动画刷新

        # 动画
        line_15_obj_ls = [
            Line().add_updater(lambda m, j=idx: m.become(
                Line(start=gene_obj_ls[j].get_center() - np.array([0, box_height / 2, 0]),
                     end=func_obj_ls[j].get_center() + np.array([0, box_height / 2, 0]),
                     color=RED_COLOR, stroke_width=1
                     )
                if gene_obj_ls[j].get_y() - func_obj_ls[j].get_y() > box_height else
                Line(stroke_opacity=0, fill_opacity=0)  # 不显示线条
            ))  # 基因与目标函数之间的连线
        for idx in range(len(gene_obj_ls))]
        self.play(Write(Line(start=ORIGIN, end=ORIGIN), run_time=0.1))  # play一个空动画，把Line给刷新出来
        self.wait()
        runtime = 0.7
        self.play(
            *[Write(line_15_obj_ls[idx]) for idx in range(len(line_15_obj_ls))],
            *[Unwrite(dec_obj_ls[idx], run_time=runtime) for idx in range(len(dec_obj_ls))],
            *[Unwrite(line12_obj_ls[idx], run_time=runtime) for idx in range(len(line12_obj_ls))],
            Unwrite(dec_lab[0], run_time=runtime), Unwrite(dec_lab[1], run_time=runtime),
            *[Unwrite(line23_obj_ls[idx], run_time=runtime) for idx in range(len(line23_obj_ls))],
            Unwrite(bef_number_axes, run_time=runtime),
            Unwrite(aft_number_axes, run_time=runtime),
            Unwrite(number_axes_lab, run_time=runtime),
            *[Unwrite(res_obj_ls[idx], run_time=runtime) for idx in range(len(res_obj_ls))],
            Unwrite(res_lab[0], run_time=runtime), Unwrite(res_lab[1], run_time=runtime),
            *[Unwrite(line34_obj_ls[idx], run_time=runtime) for idx in range(len(gene_obj_ls))],
            *[Unwrite(line45_obj_ls[idx], run_time=runtime) for idx in range(len(line45_obj_ls))],
            func_all_obj.animate.shift(UP*4)
        )
        self.wait()

        """
        1-适应度归一化（先归一化再除以总和）
        """
        func_val_ndarray = np.array(func_val)
        uniform_step1 = (func_val_ndarray - func_val_ndarray.min()) / (func_val_ndarray.max() - func_val_ndarray.min())
        uniform_step2 = uniform_step1 / uniform_step1.sum()
        bef_number_axes = NumberLine(
            x_range=[func_val_ndarray.min().round(2), func_val_ndarray.max().round(2), func_val_ndarray.max().round(2)-func_val_ndarray.min().round(2)],
            numbers_to_include=[func_val_ndarray.min().round(2), func_val_ndarray.max().round(2)],
            exclude_origin_tick=True,
            length=4,
            include_numbers=True,
            font_size=font_size - 2
        ).set_color(BLUE_COLOR)
        bef_number_axes.add(VGroup(
            bef_number_axes.get_tick(func_val_ndarray.min().round(2)),
            bef_number_axes.get_tick(func_val_ndarray.max().round(2))))
        step1_number_axes = NumberLine(
            x_range=[0, 1, 1],
            length=4,
            include_numbers=True,
            font_size=font_size - 2
        ).set_color(BLUE_COLOR)
        step1_func_obj = MathTex(r"""b_{i}=\frac{a_{i}-a_{\min}}{a_{\max}-a_{\min}}""",
                                 color=BLUE_COLOR, font_size=font_size - 1)
        step2_number_axes = NumberLine(
            x_range=[0, uniform_step2.max().round(3), uniform_step2.max().round(3)],
            length=4,
            include_numbers=True,
            font_size=font_size - 2
        ).set_color(BLUE_COLOR)
        step2_func_obj = MathTex(r"""c_{i}=\frac{b_{i}}{\sum\limits_{j}{b_{j}}}""",
                                 color=BLUE_COLOR, font_size=font_size - 1)
        number_axes = VGroup(bef_number_axes, step1_func_obj, step1_number_axes, step2_func_obj, step2_number_axes)
        number_axes.arrange(DOWN, buff=0.1)
        step1_number_axes.shift(np.array([
            bef_number_axes.n2p(func_val_ndarray.min().round(2))[0] - step1_number_axes.n2p(0)[0], 0, 0
        ]))
        step2_number_axes.shift(np.array([
            bef_number_axes.n2p(func_val_ndarray.min().round(2))[0] - step2_number_axes.n2p(0)[0], 0, 0
        ]))
        number_axes.move_to(DOWN * 1 + RIGHT * move_right)
        number_axes_lab = VGroup(
            Text("适应度", font_size=font_size - 2, color=BLUE_COLOR, font="Dream Han Serif CN"
                 ).next_to(bef_number_axes, LEFT, buff=0.3),
            Text("归一化", font_size=font_size - 2, color=BLUE_COLOR, font="Dream Han Serif CN"
                 ).next_to(step2_number_axes, LEFT, buff=0.4)
        ).shift(0.1 * UP)
        number_all_obj = VGroup(number_axes, number_axes_lab)
        line56_obj_ls = [
            VGroup(
                Line().add_updater(lambda m, j=idx: m.become(
                    Line(start=func_obj_ls[j].get_center() - np.array([0, box_height / 2, 0]),
                         end=func_obj_ls[j].get_center() - np.array([0, box_height / 2 + short_line_length, 0]),
                         color=RED_COLOR, stroke_width=1
                         )
                )),  # 上侧短线
                Line().add_updater(lambda m, j=idx: m.become(
                    Line(start=func_obj_ls[j].get_center() - np.array([0, box_height / 2 + short_line_length, 0]),
                         end=bef_number_axes.n2p(func_val[j]),
                         color=RED_COLOR, stroke_width=1
                         )
                )),  # 中间斜线
            )  # 适应度与映射之间的连线
            for idx in range(len(gene_obj_ls))]
        line_step1_obj_ls = [
            Line().add_updater(lambda m, j=idx: m.become(
                Line(start=bef_number_axes.n2p(func_val[j]),
                     end=step1_number_axes.n2p(uniform_step1[j]),
                     color=RED_COLOR, stroke_width=1)
            ))
         for idx in range(len(gene_obj_ls))]
        line_step2_obj_ls = [
            Line().add_updater(lambda m, j=idx: m.become(
                Line(start=step1_number_axes.n2p(uniform_step1[j]),
                     end=step2_number_axes.n2p(uniform_step2[j]),
                     color=RED_COLOR, stroke_width=1)
            ))
            for idx in range(len(gene_obj_ls))]
        # 归一化
        one_obj_ls = [VGroup(
            Rectangle(width=box_width, height=box_height, fill_opacity=0, stroke_color=BLUE_COLOR, stroke_width=1),
            Text(f"{uniform_step2[idx]:.3f}", font_size=font_size, color=BLUE_COLOR, font="Times New Roman")
        ) for idx in range(len(gene))]
        one_obj = VGroup(*one_obj_ls)
        one_obj.arrange(RIGHT, buff=box_buff)
        one_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1,
                      stroke_color=BLUE_COLOR, stroke_width=1, stroke_opacity=1),
            Text("概率", font_size=font_size, color=BLUE_COLOR, font="Dream Han Serif CN")
        ).next_to(one_obj, LEFT, buff=lab_buff)
        one_all_obj = VGroup(one_obj, one_lab).shift(RIGHT * move_right + 3 * DOWN)
        line67_obj_ls = [
            VGroup(
                Line().add_updater(lambda m, j=idx: m.become(
                    Line(start=step2_number_axes.n2p(uniform_step2[j]),
                         end=one_obj_ls[j].get_center() + np.array([0, box_height / 2 + short_line_length, 0]),
                         color=RED_COLOR, stroke_width=1
                         )
                )),  # 中间斜线
                Line().add_updater(lambda m, j=idx: m.become(
                    Line(start=one_obj_ls[j].get_center() + np.array([0, box_height / 2 + short_line_length, 0]),
                         end=one_obj_ls[j].get_center() + np.array([0, box_height / 2, 0]),
                         color=RED_COLOR, stroke_width=1
                         )
                )),  # 下侧短线
            )  # 映射与概率之间的连线
            for idx in range(len(gene_obj_ls))]
        code_bef = VGroup(
            Code(code_string=r'''# python =====================================================
import numpy as np
fit_ndarray = np.array(fit_ls)
temp = (fit_ndarray - fit_ndarray.min()) /
       (fit_ndarray.max() - fit_ndarray.min())
probability = temp / temp.sum() # 概率''', language="python", **code_kargs),
            Code(code_string=r'''% matlab =====================================================
%% 归一化
temp = (fit_ls - min(fit_ls)) / (max(fit_ls) - min(fit_ls));
probability = temp / sum(temp);''', language='matlab', **code_kargs)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).to_edge(RIGHT, buff=0.1).shift(DOWN * code_down)
        self.play(Write(bef_number_axes),Write(number_axes_lab[0]),
                  *[Write(obj) for obj in line56_obj_ls])
        self.play(Write(step1_number_axes), Write(step1_func_obj), run_time=0.5)
        self.play(*[Write(obj) for obj in line_step1_obj_ls], run_time=0.5)
        self.play(Write(step2_number_axes), Write(number_axes_lab[1]), Write(step2_func_obj), run_time=0.5)
        self.play(*[Write(obj) for obj in line_step2_obj_ls], run_time=0.5)
        self.play(*[Write(one_obj_ls[idx]) for idx in range(len(one_obj_ls))], Write(one_lab),
                  *[Write(obj) for obj in line67_obj_ls],
                  Write(code_bef))
        self.wait()

        """
        2-轮盘
        """
        # 隐去多余内容
        line57_obj_ls = [
            Line().add_updater(lambda m, j=idx: m.become(
                Line(start=func_obj_ls[j].get_center() - np.array([0, box_height / 2, 0]),
                     end=one_obj_ls[j].get_center() + np.array([0, box_height / 2, 0]),
                     color=RED_COLOR, stroke_width=1
                     )
                if func_obj_ls[j].get_y() - one_obj_ls[j].get_y() > box_height else
                Line(stroke_opacity=0, fill_opacity=0)  # 不显示线条
            ))  # 适应度与概率之间的连线
        for idx in range(len(gene_obj_ls))]
        self.play(
            Unwrite(number_all_obj, run_time=runtime),
            Unwrite(VGroup(*line56_obj_ls, *line67_obj_ls, *line_step1_obj_ls, *line_step2_obj_ls), run_time=runtime),
            *[Write(obj, run_time=runtime) for obj in line57_obj_ls],
            one_all_obj.animate.next_to(func_all_obj, DOWN, buff=0.2)
        )
        self.wait()

        # 绘制饼图
        angles = uniform_step2 * TAU
        angles_offset = np.cumsum((0, *angles[:-1]))
        radius = 1.1
        pie_origin = DOWN*1.3+RIGHT*move_right
        pie0 = [Pie(
                start_angle=ao, angle=a,
                stroke_width=1,
                stroke_color=BLUE_COLOR,
                radius=radius,
                fill_opacity=0)
            for ao, a in zip(angles_offset, angles)]
        pie1 = [Pie(
                start_angle=ao, angle=a,
                stroke_width=1,
                stroke_color=BLUE_COLOR,
                radius=radius,
                fill_color=YELLOW_COLOR, fill_opacity=0.5)
            for i, (ao, a) in enumerate(zip(angles_offset, angles))]
        VGroup(*pie0, *pie1).move_to(pie_origin)

        # 标签
        mid_angles = angles_offset + angles / 2
        end_points_ls = [pie_origin +
                         (np.array([[np.cos(mid_angle), -np.sin(mid_angle), 0], [np.sin(mid_angle), np.cos(mid_angle), 0], [0, 0, 1]]) @ np.array([[radius], [0], [0]])).T
        for mid_angle in mid_angles]
        pie_lab_obj = VGroup(*[
            Text(
                text=f"{uniform_step2[idx]:.3f}",
                color=BLUE_COLOR, font_size=font_size-2, font="Times New Roman", opacity=0.7
            ).move_to(0.7*end_points_ls[idx]+0.3*pie_origin)
        for idx in range(len(gene_obj_ls))])
        self.play(*[Create(obj, run_time=0.5) for obj in pie0])
        self.play(
            *[Transform(obj0, obj1, run_time=0.5) for (obj0, obj1) in zip(pie0, pie1)],
            Write(pie_lab_obj, run_time=0.5)
        )
        self.wait()

        # 指针
        choose_angles_ls = [2740, 5760, 7420, 9390, 12170]
        arrow_angle = ValueTracker(30*DEGREES)
        pie_center_dot = Dot(pie_origin, color=RED_COLOR, radius=DEFAULT_SMALL_DOT_RADIUS)
        arrow_obj = Arrow().add_updater(lambda m: m.become(
            Arrow(
                start=pie_origin,
                end=(pie_origin + (np.array([[np.cos(arrow_angle.get_value()), -np.sin(arrow_angle.get_value()), 0],
                                             [np.sin(arrow_angle.get_value()), np.cos(arrow_angle.get_value()), 0],
                                             [0, 0, 1]]) @ np.array([[radius*0.8], [0], [0]])).T)[0],
                buff=radius*0.8,
                stroke_color=RED_COLOR, stroke_width=2, stroke_opacity=0.8,tip_shape=StealthTip
            )
        ))

        # 亲本
        def judge_gene(degree: float):
            # 判断弧度对应的个体，返回个体索引
            radian = degree*DEGREES
            radian %= 2*np.pi
            return np.argmax(np.cumsum(angles)>radian)
        parent_obj_ls = [VGroup(
            Rectangle(width=box_width, height=box_height, fill_opacity=0, stroke_color=BLUE_COLOR, stroke_width=1),
            Text(f"{gene[judge_gene(choose_angles_ls[idx])]}", font_size=font_size, color=BLUE_COLOR, font="Times New Roman")
        ) for idx in range(len(gene))]
        parent_obj = VGroup(*parent_obj_ls)
        parent_obj.arrange(RIGHT, buff=box_buff)
        parent_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1,
                      stroke_color=BLUE_COLOR, stroke_width=1, stroke_opacity=1),
            Text("亲本", font_size=font_size, color=BLUE_COLOR, font="Dream Han Serif CN")
        ).next_to(parent_obj, LEFT, buff=lab_buff)
        parent_all_obj = VGroup(parent_obj, parent_lab).shift(RIGHT * move_right + 3 * DOWN)

        # 代码
        code_aft = VGroup(
            Code(code_string=r'''# python =====================================================
import numpy as np
fit_ndarray = np.array(fit_ls)
temp = (fit_ndarray - fit_ndarray.min()) /
       (fit_ndarray.max() - fit_ndarray.min())
probability = temp / temp.sum() # 概率
prefix_sum = np.cumsum(probability) # 前缀和
random_ls = np.random.uniform(
    low=0, high=1, size=len(5)) # 生成随机数
parent_ls = []  # 亲本
for random_num in random_ls:
    idx = np.argmax(prefix_sum >= random_num)
    parent_ls.append(population_ls[idx])''', language="python", **code_kargs),
            Code(code_string=r'''% matlab =====================================================
%% 归一化
temp = (fit_ls - min(fit_ls)) / (max(fit_ls) - min(fit_ls));
probability = temp / sum(temp);
%% 轮盘赌选择
prefix_sum = cumsum(probability);
random_ls = sort(rand(1,5));
parent_ls = []; % 亲本
for random_num = random_ls
    idx = find(prefix_sum>random_num, 1);
    parent_ls(end+1,:) = population_ls(idx,:);
end''', language='matlab', **code_kargs)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).to_edge(RIGHT, buff=0.1).shift(DOWN * code_down)

        self.play(Write(VGroup(pie_center_dot, arrow_obj), run_time=1))
        self.play(arrow_angle.animate.set_value(choose_angles_ls[0]*DEGREES), run_time=4)
        self.wait()
        self.play(Write(parent_lab), Write(parent_obj_ls[0]), ReplacementTransform(code_bef, code_aft))
        code_bef = code_aft
        self.wait()
        for idx in range(1,len(gene)):
            self.play(arrow_angle.animate.set_value(choose_angles_ls[idx]*DEGREES), run_time=1)
            self.play(Write(parent_obj_ls[idx]))
        self.wait()
