from manim import *
import numpy as np
from config import *


class BM2(Scene):
    def construct(self):

        self.camera.background_color = WHITE

        """
        1-展示目标函数与图像
        """
        aim_func_tex_obj = MathTex(r"\text{max }y=f(x)=(x^2-1.5) \sum\limits_{n=1}^{100}0.5^n "
                               r"\cos(7^n \pi x)+\frac{x}{2}", font_size=30)
        aim_func_tex_obj.set_color(BLUE_COLOR)
        aim_func_tex_obj.to_edge(UP,buff=0.5)

        boundary_tex_obj = MathTex(r"\text{s.t. } -3 \le x \le 3", font_size=30)
        boundary_tex_obj.set_color(BLUE_COLOR)
        boundary_tex_obj.next_to(aim_func_tex_obj, DOWN)

        x_range = (-3, 3, 1)
        y_range = (-20, 20+4, 5)
        tick_size = 30
        label_size = 35
        axes = Axes(
            x_range=x_range,
            y_range=y_range,
            x_length=10,
            y_length=5,
        )
        # axes.shift(DOWN*0.2)
        axes.set_color(BLUE_COLOR)

        aim_func = lambda x: (x**2-1.5)*np.sum([0.5**n * np.cos(7**n * np.pi * x) for n in np.arange(1,100+1,1)])+x/2
        aim_func_obj = axes.plot(aim_func, x_range=[-3,3,0.001])
        aim_func_obj.set_stroke(width=1)
        aim_func_obj.set_color(BLUE_COLOR)

        x_axes = NumberLine(x_range=x_range,
                            length=axes.x_length,
                            include_numbers=True,
                            include_tip=False,
                            font_size=tick_size)
        x_axes.set_color(BLUE_COLOR)
        x_axes.shift(axes.c2p(0,y_range[0],0)-x_axes.n2p(0))   # 下移
        x_label = MathTex("x",font_size=label_size
                          ).move_to(x_axes.n2p(0)).shift(DOWN*0.7)
        x_label.set_color(BLUE_COLOR)

        y1_axes = NumberLine(x_range=y_range,
                             length=axes.y_length,
                             include_numbers=True,
                             label_direction=LEFT,
                             include_tip=True,
                             tip_shape=StealthTip,
                             font_size=tick_size,
                             rotation=90*DEGREES)
        y1_axes.set_color(BLUE_COLOR)
        y1_axes.shift(axes.c2p(x_range[0],0,0)-y1_axes.n2p(0))  # 左移
        y1_label = MathTex("y", font_size=label_size
                           ).move_to(y1_axes.n2p(y1_axes.x_max)).shift(RIGHT*0.3)
        y1_label.set_color(BLUE_COLOR)

        y2_axes = NumberLine(x_range=[0,1.1,1],
                             length=axes.y_length,
                             include_numbers=True,
                             label_direction=RIGHT,
                             include_tip=True,
                             tip_shape=StealthTip,
                             font_size=tick_size,
                             rotation=90 * DEGREES)
        y2_axes.set_color(BLUE_COLOR)
        y2_axes.shift(axes.c2p(x_range[1], y_range[0], 0) - y2_axes.n2p(0))  # 左移
        y2_label = Text("存活力\n繁殖力", font_size=20
                           ).move_to(y2_axes.n2p(y2_axes.x_max)).shift(LEFT*0.6)
        y2_label.set_color(BLUE_COLOR)

        self.play(Write(aim_func_tex_obj), Write(boundary_tex_obj), Write(x_axes),
                  Write(x_label), Write(y1_axes), Write(y1_label), Write(aim_func_obj))
        self.wait()
        self.play(Write(y2_axes),Write(y2_label))
        self.wait()

        """
        2-展示决策变量的改变
        """
        x_value = ValueTracker(-3)
        dot = Dot(radius=0.05).set_color(RED_COLOR)
        dot.add_updater(lambda m: m.move_to(axes.c2p(x_value.get_value(), aim_func(x_value.get_value()), 0)))
        vertical_line = Line()
        vertical_line.add_updater(lambda m: m.become(
            Line(start=axes.c2p(x_value.get_value(), aim_func(x_value.get_value()), 0),
                 end=axes.c2p(x_value.get_value(), y_range[0], 0),
                 stroke_width=0.7,
                 stroke_color=RED_COLOR)
        ))
        horizontal_line = Line()
        horizontal_line.add_updater(lambda m: m.become(
            Line(start=axes.c2p(x_value.get_value(), aim_func(x_value.get_value()), 0),
                 end=axes.c2p(x_range[1], aim_func(x_value.get_value()), 0),
                 stroke_width=0.7,
                 stroke_color=RED_COLOR)
        ))
        fit_label = Text("")
        fit_label.add_updater(lambda m: m.become(
            Text(f"适应度(a.u.)\n{y2_axes.p2n(dot.get_center()):.3f}",
                 color=RED_COLOR,font_size=20,font="Dream Han Serif CN",
                 ).move_to(axes.c2p(x_range[1], aim_func(x_value.get_value()), 0)).shift(RIGHT*0.87)
        ))
        gene_label = Text("")
        gene_label.add_updater(lambda m: m.become(
            Text(f"  基因\n{x_value.get_value():.3f}",
                 color=RED_COLOR,font_size=20,font="Dream Han Serif CN",
                 ).move_to(axes.c2p(x_value.get_value(), y_range[0], 0)).shift(LEFT*0.4+UP*0.4)
        ))
        if True:    # MB1镜头修改为False
            self.play(Write(dot),Write(vertical_line),Write(horizontal_line),Write(fit_label),Write(gene_label),
                      run_time=0.3)
        else:
            self.play(Write(dot), Write(vertical_line), Write(horizontal_line), Write(fit_label),
                      run_time=0.3)
        self.play(x_value.animate.set_value(3), run_time=15)
        self.wait()
