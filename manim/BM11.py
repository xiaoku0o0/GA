from manim import *
import numpy as np
from config import *
from custom_class import FlowBox, FlowPoint, FlowArrow

class BM11(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        font_size = 18

        unit_height=0.4     # 单位高度
        color_type=YELLOW

        """
        0-无约束条件的流程图
        """
        flow1_obj_ls = [
            FlowBox(text="开始", shape=1, box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="产生初始种群", box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="格雷解码", box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="转十进制", box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="映射至定义域", box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="计算适应度", box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="达到最大迭代次数", shape=3, box_height=0.5, box_width=3.4, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="输出最大值点", shape=2, box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="结束", shape=1, box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type)
        ]
        VGroup(*flow1_obj_ls).arrange(DOWN, buff=0.4)
        flow2_obj_ls = [
            FlowBox(text="选择", box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="交叉", box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="变异", box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type)
        ]
        VGroup(*flow2_obj_ls).arrange(UP, buff=0.4).shift(RIGHT * 3.5)
        input_obj_ls = [
            FlowBox(text="定义域", shape=2, box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type
                    ).next_to(flow1_obj_ls[4], LEFT, buff=1.5),
            FlowBox(text="目标函数", shape=2, box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type
                    ).next_to(flow1_obj_ls[5], LEFT, buff=1.5),
        ]

        arrow_obj_ls1 = [
            FlowArrow(flow1_obj_ls[idx], flow1_obj_ls[idx+1], line_color=color_type)
        for idx in range(len(flow1_obj_ls)-1)]
        arrow_obj_ls2 = [
            FlowArrow(flow2_obj_ls[idx], flow2_obj_ls[idx+1], direct1=UP, direct2=DOWN, line_color=color_type)
        for idx in range(len(flow2_obj_ls)-1)]
        arrow_obj_ls2.insert(0, FlowArrow(
            flow1_obj_ls[6], flow2_obj_ls[0], direct1=RIGHT, direct2=DOWN, line_color=color_type
        ))
        arrow_obj_ls2.append(FlowArrow(
            flow2_obj_ls[2], FlowPoint().move_to(arrow_obj_ls1[1].get_center()), direct1=UP, direct2=RIGHT, line_color=color_type
        ))
        input_arrow_obj_ls = [
            FlowArrow(input_obj_ls[idx], flow1_obj_ls[4+idx], direct1=RIGHT, direct2=LEFT, line_color=color_type)
        for idx in range(len(input_obj_ls))]

        judge_lab_ls = [
            Text(
                text="是", font_size=font_size, font="Dream Han Serif CN", color=color_type
            ).next_to(arrow_obj_ls1[-2], LEFT, buff=0.1),
            Text(
                text="否", font_size=font_size, font="Dream Han Serif CN", color=color_type
            ).next_to(arrow_obj_ls2[0], LEFT, buff=-0.1).shift(DOWN*0.1),
        ]

        bef_flow = VGroup(
            *flow1_obj_ls, *flow2_obj_ls, *input_obj_ls,
            *arrow_obj_ls1, *arrow_obj_ls2, *input_arrow_obj_ls, *judge_lab_ls
        )
        # manim
        self.add(bef_flow)
        self.wait()

        """
        1-有约束条件的流程图
        """
        runtime = 0.8
        breaktime = 0.2

        flow1_obj_ls = [
            FlowBox(text="开始", shape=1, box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="产生初始种群", box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="符合约束条件", shape=3, box_height=0.5, box_width=3.4, fill_color=YELLOW_COLOR, fill_opacity=0.17, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="格雷解码", box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="转十进制", box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="映射至定义域", box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="计算适应度", box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="达到最大迭代次数", shape=3, box_height=0.5, box_width=3.4, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="输出最大值点", shape=2, box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="结束", shape=1, box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type)
        ]
        VGroup(*flow1_obj_ls).arrange(DOWN, buff=0.4)
        flow1_obj_ls[2].shift(UP*0.2)
        flow2_obj_ls = [
            FlowBox(text="选择", box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="交叉", box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="变异", box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type),
            FlowBox(text="符合约束条件", shape=3, box_height=0.5, box_width=3.4, fill_color=YELLOW_COLOR, fill_opacity=0.17, font_size=font_size, stroke_color=color_type, font_color=color_type),
        ]
        VGroup(*flow2_obj_ls).arrange(UP, buff=0.4).shift(RIGHT * 3.5)
        input_obj_ls = [
            FlowBox(text="定义域", shape=2, box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type
                    ).next_to(flow1_obj_ls[5], LEFT, buff=1.5),
            FlowBox(text="目标函数", shape=2, box_height=unit_height, font_size=font_size, stroke_color=color_type, font_color=color_type
                    ).next_to(flow1_obj_ls[6], LEFT, buff=1.5),
        ]

        arrow_obj_ls1 = [
            FlowArrow(flow1_obj_ls[idx], flow1_obj_ls[idx + 1], line_color=color_type)
            for idx in range(len(flow1_obj_ls) - 1)]
        arrow_obj_ls2 = [
            FlowArrow(flow2_obj_ls[idx], flow2_obj_ls[idx + 1], direct1=UP, direct2=DOWN, line_color=color_type)
            for idx in range(len(flow2_obj_ls) - 1)]
        arrow_obj_ls2.insert(0, FlowArrow(
            flow1_obj_ls[7], flow2_obj_ls[0], direct1=RIGHT, direct2=DOWN, line_color=color_type
        ))
        arrow_obj_ls2.append(FlowArrow(
            flow2_obj_ls[-1], FlowPoint().move_to(arrow_obj_ls1[2].get_center()), direct1=UP, direct2=RIGHT, line_color=color_type
        ))
        input_arrow_obj_ls = [
            FlowArrow(input_obj_ls[idx], flow1_obj_ls[5 + idx], direct1=RIGHT, direct2=LEFT, line_color=color_type)
            for idx in range(len(input_obj_ls))]

        judge_lab_ls = [
            Text(
                text="是", font_size=font_size, font="Dream Han Serif CN", color=color_type
            ).next_to(arrow_obj_ls1[-2], LEFT, buff=0.1),
            Text(
                text="否", font_size=font_size, font="Dream Han Serif CN", color=color_type
            ).next_to(arrow_obj_ls2[0], LEFT, buff=-0.1).shift(DOWN * 0.1),
        ]
        add_arrow = [
            VGroup(
                tmp:=Line(start=flow1_obj_ls[2].get_left(), end=flow1_obj_ls[2].get_left()+LEFT, stroke_width=1, color=color_type),
                FlowArrow(obj1=FlowPoint().move_to(tmp.get_end()), obj2=FlowPoint().move_to(arrow_obj_ls1[0].get_center()),
                          direct1=UP, direct2=LEFT, line_color=color_type)
            ),
            VGroup(
                tmp:=Line(start=flow2_obj_ls[-1].get_right(), end=flow2_obj_ls[-1].get_right()+RIGHT, stroke_width=1, color=color_type),
                FlowArrow(obj1=FlowPoint().move_to(tmp.get_end()), obj2=FlowPoint().move_to(np.array([arrow_obj_ls2[1].get_center()[0], arrow_obj_ls2[0].get_right()[1], 0])),
                          direct1=DOWN, direct2=RIGHT, line_color=color_type)
            )
        ]
        add_label = [
            Text(
                text="是", font_size=font_size, font="Dream Han Serif CN", color=color_type
            ).next_to(arrow_obj_ls1[2], LEFT, buff=0.1),
            Text(
                text="否", font_size=font_size, font="Dream Han Serif CN", color=color_type
            ).next_to(add_arrow[0][0], RIGHT, buff=-0.1).shift(UP * 0.2),
            Text(
                text="是", font_size=font_size, font="Dream Han Serif CN", color=color_type
            ).next_to(arrow_obj_ls2[-1], RIGHT, buff=0.1).shift(DOWN * 0),
            Text(
                text="否", font_size=font_size, font="Dream Han Serif CN", color=color_type
            ).next_to(add_arrow[1][0], LEFT, buff=-0.1).shift(UP * 0.2),
        ]

        aft_flow = VGroup(
            *flow1_obj_ls[:2], *flow1_obj_ls[3:], *flow2_obj_ls[:-1], *input_obj_ls,
            *arrow_obj_ls1[:2], *arrow_obj_ls1[3:], *arrow_obj_ls2[:-2], arrow_obj_ls2[-1], *input_arrow_obj_ls, *judge_lab_ls
        )
        addition_obj = VGroup(
            flow1_obj_ls[2], arrow_obj_ls1[2], flow2_obj_ls[-1], arrow_obj_ls2[-2],
            *add_arrow, *add_label
        )
        # manim
        self.play(Transform(bef_flow, aft_flow, run_time=2), Write(addition_obj, run_time=2))
        self.wait()
