from manim import *
import numpy as np
from config import *
from custom_class import FlowBox, FlowPoint, FlowArrow

class BM9(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        font_size = 18

        unit_height=0.4     # 单位高度

        flow1_obj_ls = [
            FlowBox(text="开始", shape=1),
            FlowBox(text="产生初始种群"),
            FlowBox(text="格雷解码"),
            FlowBox(text="转十进制"),
            FlowBox(text="映射至定义域"),
            FlowBox(text="计算适应度"),
            FlowBox(text="达到最大迭代次数", shape=3, box_height=0.6, box_width=3.4),
            FlowBox(text="输出最大值点", shape=2),
            FlowBox(text="结束", shape=1)
        ]
        VGroup(*flow1_obj_ls).arrange(DOWN, buff=0.4)
        flow2_obj_ls = [
            FlowBox(text="选择"),
            FlowBox(text="交叉"),
            FlowBox(text="变异")
        ]
        VGroup(*flow2_obj_ls).arrange(UP, buff=0.4).shift(RIGHT * 3.5)
        input_obj_ls = [
            FlowBox(text="定义域", shape=2).next_to(flow1_obj_ls[4], LEFT, buff=1.5),
            FlowBox(text="目标函数", shape=2).next_to(flow1_obj_ls[5], LEFT, buff=1.5),
        ]

        arrow_obj_ls1 = [
            FlowArrow(flow1_obj_ls[idx], flow1_obj_ls[idx+1])
        for idx in range(len(flow1_obj_ls)-1)]
        arrow_obj_ls2 = [
            FlowArrow(flow2_obj_ls[idx], flow2_obj_ls[idx+1], direct1=UP, direct2=DOWN)
        for idx in range(len(flow2_obj_ls)-1)]
        arrow_obj_ls2.insert(0, FlowArrow(
            flow1_obj_ls[6], flow2_obj_ls[0], direct1=RIGHT, direct2=DOWN
        ))
        arrow_obj_ls2.append(FlowArrow(
            flow2_obj_ls[2], FlowPoint().move_to(arrow_obj_ls1[1].get_center()), direct1=UP, direct2=RIGHT
        ))
        input_arrow_obj_ls = [
            FlowArrow(input_obj_ls[idx], flow1_obj_ls[4+idx], direct1=RIGHT, direct2=LEFT)
        for idx in range(len(input_obj_ls))]

        judge_lab_ls = [
            Text(
                text="是", font_size=font_size, font="Dream Han Serif CN", color=BLUE_COLOR
            ).next_to(arrow_obj_ls1[-2], LEFT, buff=0.1),
            Text(
                text="否", font_size=font_size, font="Dream Han Serif CN", color=BLUE_COLOR
            ).next_to(arrow_obj_ls2[0], LEFT, buff=-0.1).shift(DOWN*0.1),
        ]

        temp_arrow = VGroup(
            Line(
                start=flow1_obj_ls[-4].get_center()+DOWN*unit_height/2,
                end=flow1_obj_ls[-3].get_center(),
                color=BLUE_COLOR,
                stroke_width=1
            ),
            FlowArrow(FlowPoint().move_to(flow1_obj_ls[-3].get_center()), flow2_obj_ls[0], direct1=RIGHT, direct2=DOWN)
        )

        # manim
        runtime = 0.8
        breaktime = 0.2
        # 产生初始种群
        self.play(Write(flow1_obj_ls[0]), run_time=runtime)
        self.play(Write(arrow_obj_ls1[0]), Write(flow1_obj_ls[1]), run_time=runtime)
        self.wait()
        # 格雷解码转十进制映射回定义域
        self.play(Write(flow1_obj_ls[2]), Write(arrow_obj_ls1[1]), run_time=runtime)
        self.wait(breaktime)
        self.play(Write(flow1_obj_ls[3]), Write(arrow_obj_ls1[2]), run_time=runtime)
        self.wait(breaktime)
        self.play(Write(flow1_obj_ls[4]), Write(arrow_obj_ls1[3]), Write(input_obj_ls[0]), Write(input_arrow_obj_ls[0]), run_time=runtime)
        self.wait(breaktime)
        self.play(Write(flow1_obj_ls[5]), Write(arrow_obj_ls1[4]), Write(input_obj_ls[1]), Write(input_arrow_obj_ls[1]), run_time=runtime)
        self.wait()
        # 选择亲本
        self.play(Write(flow2_obj_ls[0]), Write(temp_arrow), run_time=runtime)
        self.wait()
        # 交叉与变异
        self.play(Write(flow2_obj_ls[1]), Write(arrow_obj_ls2[1]), run_time=runtime)
        self.wait(breaktime)
        self.play(Write(flow2_obj_ls[2]), Write(arrow_obj_ls2[2]), run_time=runtime)
        self.wait(breaktime)
        self.play(Write(arrow_obj_ls2[3]), run_time=runtime)
        # 不断往复
        self.play(Write(flow1_obj_ls[6]),
                  Write(VGroup(arrow_obj_ls1[5], arrow_obj_ls2[0], judge_lab_ls[1])),
                  Unwrite(temp_arrow))
        self.wait()
        # 最后一代
        self.play(Write(flow1_obj_ls[7]), Write(arrow_obj_ls1[6]), Write(judge_lab_ls[0]), run_time=runtime)
        self.play(Write(flow1_obj_ls[8]), Write(arrow_obj_ls1[7]), run_time=runtime)
        self.wait()
