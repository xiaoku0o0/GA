from manim import *
import numpy as np
from config import *

class CM1(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        lab_buff = 0.2  # 左侧标签间距
        move_right = 0.7  # 整体向右偏移量
        box_height = 0.4  # 标签高度
        box_width = 2.3  # 标签宽度
        box_buff = 0.3  # 标签间距
        label_width = 1.1  # 左侧标签宽度
        font_size = 18

        """
        1-基因拆分
        """
        gene_ls = ["00001011", "01101001", "10000110", "10110011"]
        gene_obj = VGroup(*[
            VGroup(*[
                VGroup(
                    Rectangle(width=box_width / 8, height=box_height,
                              fill_opacity=0,
                              stroke_color=YELLOW_COLOR, stroke_opacity=1, stroke_width=1),
                    Text(f"{gene_ls[var_idx][gene_idx]}",
                         color=YELLOW_COLOR,
                         font_size=font_size,
                         font="Times New Roman")
                )
            for gene_idx in range(len(gene_ls[var_idx]))]).arrange(RIGHT, buff=0)
        for var_idx in range(len(gene_ls))]).arrange(RIGHT, buff=0).shift(RIGHT*move_right)
        gene_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1,
                      stroke_color=BLUE_COLOR, stroke_width=1, stroke_opacity=1),
            Text("基因", font_size=font_size, color=BLUE_COLOR, font="Dream Han Serif CN")
        ).next_to(gene_obj, LEFT, buff=lab_buff).add_updater(lambda m: m.next_to(gene_obj, LEFT, buff=lab_buff))
        gene_all_obj = VGroup(gene_obj, gene_lab)
        self.add(gene_all_obj)
        self.play(gene_obj.animate.arrange(RIGHT, buff=box_buff).shift(UP*3+RIGHT*move_right))
        self.wait()

        """
        2-格雷解码
        """
        def gray_decoding(
                gene: tuple[int|str]  # 基因
        ) -> tuple[int]:
            """格雷解码"""
            gene = np.array(gene).astype(int)  # 便于计算
            mask = np.triu(np.ones(len(gene), dtype=int))
            gene_aft = np.mod(gene @ mask, 2)  # 模二加法
            return tuple(gene_aft)
        gray_ls = [
            gray_decoding(tuple(gene))
        for gene in gene_ls]

        gray_obj = VGroup(*[
            VGroup(*[
                VGroup(
                    Rectangle(width=box_width / 8, height=box_height,
                              fill_opacity=0,
                              stroke_color=YELLOW_COLOR, stroke_opacity=1, stroke_width=1),
                    Text(f"{gray_ls[var_idx][gene_idx]}",
                         color=YELLOW_COLOR,
                         font_size=font_size,
                         font="Times New Roman")
                )
                for gene_idx in range(len(gene_ls[var_idx]))]).arrange(RIGHT, buff=0)
            for var_idx in range(len(gene_ls))]).arrange(RIGHT, buff=box_buff).shift(UP * 1.5 + RIGHT * move_right)
        gray_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1,
                      stroke_color=BLUE_COLOR, stroke_width=1, stroke_opacity=1),
            Text("格雷解码", font_size=font_size, color=BLUE_COLOR, font="Dream Han Serif CN")
        ).next_to(gray_obj, LEFT, buff=lab_buff)
        # 箭头
        array_buff = 0.05
        decode_arrow_obj = VGroup(*[
            VGroup(
                Arrow(
                    start=gene_obj[var_idx][0].get_bottom(),
                    end=gray_obj[var_idx][0].get_top(),
                    buff=array_buff,
                    color=CYAN_COLOR,
                    stroke_width=1,
                    stroke_opacity=0.9,
                    max_tip_length_to_length_ratio=0.1,
                    tip_shape=StealthTip
                ),
                *[
                    VGroup(
                        Line(
                            start=gray_obj[var_idx][gene_bit-1].get_top() + (UP + RIGHT) * array_buff,
                            end=gene_obj[var_idx][gene_bit].get_bottom() + DOWN * array_buff,
                            color=CYAN_COLOR,
                            stroke_width=1,
                            stroke_opacity=0.9
                        ),
                        Arrow(
                            start=gene_obj[var_idx][gene_bit].get_bottom(),
                            end=gray_obj[var_idx][gene_bit].get_top(),
                            buff=array_buff,
                            color=CYAN_COLOR,
                            stroke_width=1,
                            stroke_opacity=0.9,
                            max_tip_length_to_length_ratio=0.1,
                            tip_shape=StealthTip
                        )
                    )
                for gene_bit in range(1, len(gene_ls[var_idx]))]
            )
        for var_idx in range(len(gene_ls))])
        gray_all_obj = VGroup(decode_arrow_obj, gray_obj, gray_lab)
        self.play(Write(decode_arrow_obj), Write(gray_lab), Write(gray_obj), run_time=0.8)

        """
        3-转十进制
        """
        int_ls = [
            int(''.join(map(str, gray_ls[var_idx])), 2)
        for var_idx in range(len(gray_ls))]
        int_obj = VGroup(*[
            VGroup(
                Rectangle(width=box_width, height=box_height,
                          fill_opacity=0,
                          stroke_color=YELLOW_COLOR, stroke_opacity=1, stroke_width=1),
                Text(f"{int_ls[var_idx]}",
                     color=YELLOW_COLOR,
                     font_size=font_size,
                     font="Times New Roman")
            )
            for var_idx in range(len(gene_ls))]).arrange(RIGHT, buff=box_buff).shift(UP * 0 + RIGHT * move_right)
        int_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1,
                      stroke_color=BLUE_COLOR, stroke_width=1, stroke_opacity=1),
            Text("十进制", font_size=font_size, color=BLUE_COLOR, font="Dream Han Serif CN")
        ).next_to(int_obj, LEFT, buff=lab_buff)
        int_arrow_obj = VGroup(*[
            Arrow(
                start=gray_obj[var_idx].get_bottom(),
                end=int_obj[var_idx].get_top(),
                buff=array_buff,
                color=CYAN_COLOR,
                stroke_width=1,
                stroke_opacity=0.9,
                max_tip_length_to_length_ratio=0.1,
                tip_shape=StealthTip
            )
        for var_idx in range(len(gene_ls))])
        self.play(Write(int_arrow_obj), Write(int_lab), Write(int_obj), run_time=0.8)

        """
        4-映射至定义域
        """
        def_domain_ls = [(-5, 5), (-5, 5), (0, 1), (0, 1)]

        res_ls = [
            np.interp(
                int_ls[var_idx],
                (0, 1<<len(gene_ls[var_idx])-1),
                def_domain_ls[var_idx]
            )
        for var_idx in range(len(gene_ls))]
        res_obj = VGroup(*[
            VGroup(
                Rectangle(width=box_width, height=box_height,
                          fill_opacity=0,
                          stroke_color=YELLOW_COLOR, stroke_opacity=1, stroke_width=1),
                Text(f"{res_ls[var_idx]}",
                     color=YELLOW_COLOR,
                     font_size=font_size,
                     font="Times New Roman")
            )
            for var_idx in range(len(gene_ls))]).arrange(RIGHT, buff=box_buff).shift(DOWN*1.5 + RIGHT * move_right)
        res_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1,
                      stroke_color=BLUE_COLOR, stroke_width=1, stroke_opacity=1),
            Text("决策变量", font_size=font_size, color=BLUE_COLOR, font="Dream Han Serif CN")
        ).next_to(res_obj, LEFT, buff=lab_buff)
        def_arrow_obj = VGroup(*[
            Arrow(
                start=int_obj[var_idx].get_bottom(),
                end=res_obj[var_idx].get_top(),
                buff=array_buff,
                color=CYAN_COLOR,
                stroke_width=1,
                stroke_opacity=0.9,
                max_tip_length_to_length_ratio=0.1,
                tip_shape=StealthTip
            )
        for var_idx in range(len(gene_ls))])
        def_obj = VGroup(*[
            VGroup(
                Rectangle(width=box_width / 2.3, height=box_height,
                          fill_opacity=0,
                          stroke_color=CYAN_COLOR, stroke_opacity=1, stroke_width=1),
                Text(f"[{def_domain[0]},{def_domain[1]}]",
                     color=CYAN_COLOR,
                     font_size=font_size,
                     font="Times New Roman")
            ).next_to(def_arrow_obj[var_idx], LEFT, 0.1)
        for var_idx, def_domain in enumerate(def_domain_ls)])
        def_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=BLUE_COLOR, fill_opacity=0.4, stroke_opacity=0),
            Text("定义域", font_size=font_size, color=YELLOW_COLOR, font="Dream Han Serif CN")
        ).move_to(np.array([gene_lab.get_x(), def_obj.get_y(), 0]))

        self.play(Write(def_obj), Write(def_lab), Write(def_arrow_obj), Write(res_lab), Write(res_obj), run_time=0.8)
        self.wait()
