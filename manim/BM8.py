from manim import *
import numpy as np
from config import *
from custom_class import ThreeLineTable

class BM8(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        font_size = 18
        """
        0-展示十进制、二进制、格雷码表格
        """
        def gray_coding(binary: str)->str:
            # 格雷编码
            binary = np.array(list(map(int, binary)))  # 转为ndarray对象便于计算
            res = np.concatenate((binary[0:1],
                                  np.bitwise_xor(binary[1:], binary[:-1])))
            return ''.join(map(str, res))
        table_data = [
            [num, f"{bin(num)[2:]:>04}", gray_coding(f"{bin(num)[2:]:>04}")]
        for num in range(1, 16)]
        table_obj = ThreeLineTable(
            data=[["十进制", "二进制", "格雷码"]]+table_data,
            vertical_lines=[1,2],
            col_widths=[1, 1.3, 1.3],
            base_line_width=0.5,
            font_color=BLUE_COLOR,
            line_color=BLUE_COLOR,
            header_font_size=font_size+1,
            content_font_size=font_size,
            h_buff=0,
            v_buff=0.3
        ).shift(UP*3)
        self.add(table_obj)
