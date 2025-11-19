from manim import *
import numpy as np
from config import *

class BM6(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        lab_buff = 0.8  # 左侧标签间距
        box_height = 0.4  # 标签高度
        box_width = 1.15  # 标签宽度
        box_buff = 0.3  # 标签间距
        label_width = 1.1  # 左侧标签宽度
        font_size = 18

        """
        1-展示十进制与二进制
        """
        # 基因
        gene = ["0110", "0111", "1000", "1001"]
        gene_obj_ls = [VGroup(
            Rectangle(width=box_width, height=box_height, fill_opacity=0, stroke_color=BLUE_COLOR, stroke_width=1),
            Text(gene[idx], font_size=font_size, color=BLUE_COLOR, font="Times New Roman")
        ) for idx in range(len(gene))]
        gene_obj = VGroup(*gene_obj_ls)
        gene_obj.arrange(RIGHT, buff=box_buff)
        gene_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1,
                      stroke_color=BLUE_COLOR, stroke_width=1, stroke_opacity=1),
            Text("基因", font_size=font_size, color=BLUE_COLOR, font="Dream Han Serif CN")
        ).next_to(gene_obj, LEFT, buff=lab_buff)
        gene_all_obj = VGroup(gene_obj, gene_lab).shift(UP * 1.6)

        # 十进制
        dec_val = [int(gene[idx], 2) for idx in range(len(gene))]
        dec_obj_ls = [VGroup(
            Rectangle(width=box_width, height=box_height, fill_opacity=0, stroke_color=YELLOW_COLOR, stroke_width=1),
            Text(f"{dec_val[idx]}", font_size=font_size, color=BLUE_COLOR, font="Times New Roman")
        ) for idx in range(len(dec_val))
        ]
        dec_obj = VGroup(*dec_obj_ls)
        dec_obj.arrange(RIGHT, buff=box_buff)
        dec_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1),
            Text("十进制", font_size=font_size, color=BLUE_COLOR, font="Dream Han Serif CN")
        ).next_to(dec_obj, LEFT, buff=lab_buff)
        dec_all_obj = VGroup(dec_obj, dec_lab)
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

        # manim动画
        self.play(*[Write(obj) for obj in dec_all_obj])
        self.wait()
        self.add(*line12_obj_ls)
        self.play(TransformFromCopy(dec_all_obj, gene_all_obj))

        """
        2-强调7和8的二进制
        """
        gene_fill_obj = VGroup(*[
            VGroup(
                Rectangle(width=box_width, height=box_height,
                          fill_color=RED_COLOR, fill_opacity=0.6, stroke_color=BLUE_COLOR, stroke_width=1),
                Text(gene[idx], font_size=font_size, color=WHITE, font="Times New Roman")
            ).move_to(gene_obj_ls[idx].get_center())
        for idx in (1, 2)])
        self.add(gene_fill_obj)
        self.wait(flash_time)
        self.remove(gene_fill_obj)
        self.wait(flash_time)
        self.add(gene_fill_obj)
        self.wait()

        """
        3-7和8的十进制擦除
        """
        fill_clock = ValueTracker(0)
        dec_fill_obj = VGroup(*[
            Rectangle().add_updater(lambda m, j=idx: m.become(
                Rectangle(
                    width=box_width, height=box_height*fill_clock.get_value(),
                    fill_color=RED_COLOR, fill_opacity=0.6, stroke_color=YELLOW_COLOR, stroke_width=1
                ).move_to(dec_obj_ls[j].get_center()+UP*(1-fill_clock.get_value())*box_height/2)
            ))
        for idx in (1, 2)])
        self.add(dec_fill_obj)
        self.play(fill_clock.animate.set_value(1))
        self.wait()
