from manim import *
import numpy as np
from config import *
from custom_class import ThreeLineTable
from copy import deepcopy

class BM7(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        lab_buff = 0.8  # 左侧标签间距
        box_height = 0.4  # 标签高度
        box_width = 1.5  # 标签宽度
        box_buff = 0.8  # 标签间距
        label_width = 1.1  # 左侧标签宽度
        font_size = 18

        move_right = -2.5  # 整体向右偏移量
        code_down = 0.75  # 代码中心下移距离

        """
        1-展示7和8的二进制
        """
        binary_val = ["0111", "1000"]
        binary_obj_ls = [
            VGroup(*[VGroup(
                Rectangle(width=box_width / 4, height=box_height,
                          fill_opacity=0,
                          stroke_color=BLUE_COLOR, stroke_opacity=1, stroke_width=1),
                Text(f"{char}",
                     color=BLUE_COLOR,
                     font_size=font_size,
                     font="Times New Roman")
            )for char in binary]).arrange(RIGHT, buff=0)
        for binary in binary_val]
        binary_obj = VGroup(*binary_obj_ls).arrange(RIGHT, buff=box_buff)
        binary_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1),
            Text("二进制", font_size=font_size, color=BLUE_COLOR, font="Dream Han Serif CN")
        ).next_to(binary_obj, LEFT, buff=lab_buff)
        binary_all_obj = VGroup(binary_lab, binary_obj).shift(RIGHT*move_right+UP*1.5)

        self.play(
            *[Write(obj) for obj in binary_obj_ls],
            Write(binary_lab)
        )
        self.wait()

        """
        2-展示格雷编码过程
        """
        # 格雷码
        def gray_coding(binary: str)->str:
            # 格雷编码
            binary = np.array(list(map(int, binary)))  # 转为ndarray对象便于计算
            res = np.concatenate((binary[0:1],
                                  np.bitwise_xor(binary[1:], binary[:-1])))
            return ''.join(map(str, res))
        gray_val = [
            gray_coding(binary)
        for binary in binary_val]

        gray_obj_ls = [
            VGroup(*[VGroup(
                Rectangle(width=box_width / 4, height=box_height,
                          fill_opacity=0,
                          stroke_color=BLUE_COLOR, stroke_opacity=1, stroke_width=1),
                Text(f"{char}",
                     color=BLUE_COLOR,
                     font_size=font_size,
                     font="Times New Roman")
            ) for char in gray]).arrange(RIGHT, buff=0)
            for gray in gray_val]
        gray_obj = VGroup(*gray_obj_ls).arrange(RIGHT, buff=box_buff)
        gray_lab = VGroup(
            Rectangle(width=label_width, height=box_height, fill_color=YELLOW_COLOR, fill_opacity=1),
            Text("格雷码", font_size=font_size, color=BLUE_COLOR, font="Dream Han Serif CN")
        ).next_to(gray_obj, LEFT, buff=lab_buff)
        gray_all_obj = VGroup(gray_lab, gray_obj).shift(RIGHT*move_right)

        # 箭头
        array_buff = 0.05
        code_first_arrow_obj = [
            Arrow(
                start=binary_obj_ls[idx][0].get_center()+DOWN*box_height/2,
                end=gray_obj_ls[idx][0].get_center()+UP*box_height/2,
                buff=array_buff,
                color=RED_COLOR,
                stroke_width=1,
                max_tip_length_to_length_ratio=0.1,
                tip_shape=StealthTip
            )
        for idx in range(len(binary_val))]
        code_follow_arrow_obj = [
            VGroup(*[
                VGroup(
                    Line(
                        start=binary_obj_ls[idx][bit-1].get_center()+DOWN*(box_height/2+array_buff)+RIGHT*array_buff,
                        end=binary_obj_ls[idx][bit].get_center()+DOWN*(box_height/2+array_buff),
                        color=RED_COLOR,
                        stroke_width=1
                    ),
                    Arrow(
                        start=binary_obj_ls[idx][bit].get_center() + DOWN * box_height/2,
                        end=gray_obj_ls[idx][bit].get_center() + UP * box_height/2,
                        buff=array_buff,
                        color=RED_COLOR,
                        stroke_width=1,
                        max_tip_length_to_length_ratio=0.1,
                        tip_shape=StealthTip
                    )
                )
            for bit in range(1, len(binary_val[0]))])
        for idx in range(len(binary_val))]
        # 异或运算真值表
        table_obj = ThreeLineTable(
            data=[
                    ['A', 'B', 'A⊕B'],
                    ['0', '0', '0'],
                    ['0', '1', '1'],
                    ['1', '0', '1'],
                    ['1', '1', '0']
            ],
            vertical_lines=[2],
            col_widths=[0.8,0.8,1.3],
            base_line_width=0.5,
            font_color=BLUE_COLOR,
            line_color=BLUE_COLOR,
            header_font_size=font_size,
            content_font_size=font_size,
            h_buff=0,
            v_buff=0.3
        ).to_edge(RIGHT, buff=1).shift(UP*2.5*0.3)
        # 代码
        code_code_obj = VGroup(
            Code(code_string=r'''
# python =====================================================
import numpy as np
def gray_coding(binary: str)->str:
    # 格雷编码
    binary = np.array(list(map(int, binary)))  # 便于计算
    res = np.concatenate((binary[0:1],
                          np.bitwise_xor(binary[1:], 
                                         binary[:-1])))
    return ''.join(map(str, res))''', language="python", **code_kargs),
            Code(code_string=r'''
% matlab =====================================================
%% 格雷编码
gray_ls = cat(2, population_ls(:,1), ...
    xor(population_ls(:,1:end-1), population_ls(:,2:end)));''', language='matlab', **code_kargs)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).to_edge(RIGHT, buff=0.1)

        # manim
        # 首位不变
        self.play(Write(gray_lab),
                  *[Write(obj) for obj in code_first_arrow_obj], run_time=0.5)
        self.play(*[Write(gray_obj_ls[idx][0]) for idx in range(len(gray_val))], run_time=0.5)
        self.wait()
        # 异或运算
        self.play(Write(table_obj), run_time=0.5)
        for bit in range(1, len(gray_val[0])):
            self.play(*[Write(code_follow_arrow_obj[idx][bit-1]) for idx in range(len(gray_val))], run_time=0.5)
            self.play(*[Write(gray_obj_ls[idx][bit]) for idx in range(len(gray_val))], run_time=0.5)
            self.wait(0.2)
        self.play(Uncreate(table_obj), run_time=0.5)
        self.play(Write(code_code_obj))
        self.wait()

        """
        3-展示格雷解码过程
        """
        # 二进制
        aft_binary_obj_ls = deepcopy(binary_obj_ls)
        aft_binary_obj = VGroup(*aft_binary_obj_ls)
        aft_binary_lab = binary_lab.copy()
        aft_binary_all_obj = VGroup(aft_binary_lab, aft_binary_obj).shift(2*1.5*DOWN)

        # 箭头
        decode_first_arrow_obj = [
            Arrow(
                start=gray_obj_ls[idx][0].get_center() + DOWN * box_height / 2,
                end=aft_binary_obj_ls[idx][0].get_center() + UP * box_height / 2,
                buff=array_buff,
                color=RED_COLOR,
                stroke_width=1,
                max_tip_length_to_length_ratio=0.1,
                tip_shape=StealthTip
            )
            for idx in range(len(binary_val))]
        decode_follow_arrow_obj = [
            VGroup(*[
                VGroup(
                    Line(
                        start=aft_binary_obj_ls[idx][bit - 1].get_center() + UP * (
                                    box_height / 2 + array_buff) + RIGHT * array_buff,
                        end=gray_obj_ls[idx][bit].get_center() + DOWN * (box_height / 2 + array_buff),
                        color=RED_COLOR,
                        stroke_width=1
                    ),
                    Arrow(
                        start=gray_obj_ls[idx][bit].get_center() + DOWN * box_height / 2,
                        end=aft_binary_obj_ls[idx][bit].get_center() + UP * box_height / 2,
                        buff=array_buff,
                        color=RED_COLOR,
                        stroke_width=1,
                        max_tip_length_to_length_ratio=0.1,
                        tip_shape=StealthTip
                    )
                )
                for bit in range(1, len(binary_val[0]))])
            for idx in range(len(binary_val))]

        # 代码
        decode_code_obj = VGroup(
            Code(code_string=r'''
# python =====================================================
import numpy as np
def gray_coding(binary: str)->str:
    # 格雷编码
    binary = np.array(list(map(int, binary)))  # 便于计算
    res = np.concatenate((binary[0:1],
                          np.bitwise_xor(binary[1:], 
                                         binary[:-1])))
    return ''.join(map(str, res))
def gray_decoding(gray: str)->str:
    # 格雷解码
    gray = np.array(list(map(int, gray)))  # 便于计算
    mask = np.triu(np.ones(len(gray), dtype=int))
    res = np.mod(gray @ mask, 2)    # 模二加法
    return ''.join(map(str, res))''', language="python", **code_kargs),
            Code(code_string=r'''
% matlab =====================================================
%% 格雷编码
gray_ls = cat(2, population_ls(:,1), ...
    xor(population_ls(:,1:end-1), population_ls(:,2:end)));
%% 格雷解码
[~,m] = size(gray_ls);
mask = triu(ones(m));
population_ls = mod(gray_ls * mask, 2);''', language='matlab', **code_kargs)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).to_edge(RIGHT, buff=0.1)

        # manim
        # 首位不变
        self.play(Write(aft_binary_lab),
                  *[Write(obj) for obj in decode_first_arrow_obj], run_time=0.5)
        self.play(*[Write(aft_binary_obj_ls[idx][0]) for idx in range(len(gray_val))], run_time=0.5)
        self.wait()
        # 异或运算
        for bit in range(1, len(gray_val[0])):
            self.play(*[Write(decode_follow_arrow_obj[idx][bit - 1]) for idx in range(len(gray_val))], run_time=0.5)
            self.play(*[Write(aft_binary_obj_ls[idx][bit]) for idx in range(len(gray_val))], run_time=0.5)
            self.wait(0.2)
        self.play(ReplacementTransform(code_code_obj, decode_code_obj))
        self.wait()
