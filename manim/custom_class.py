"""自定义类，用于实现非内置的manim图形"""

from manim import *
import numpy as np
from config import *
from typing import Callable


class Pie(Sector):
    """饼图"""
    def interpolate(self, mobject1, mobject2, alpha, path_func=straight_path):
        if not (isinstance(mobject1, Pie) and isinstance(mobject2, Pie)):
            return super().interpolate(mobject1, mobject2, alpha, path_func=path_func)

        for attr in (
            'start_angle', 'angle',
            'inner_radius', 'outer_radius',
        ):
            v1 = getattr(mobject1, attr)
            v2 = getattr(mobject2, attr)
            setattr(self, attr, path_func(v1, v2, alpha))

        self.arc_center = path_func(
            mobject1.get_arc_center(),
            mobject2.get_arc_center(),
            alpha
        )
        self.interpolate_color(mobject1, mobject2, alpha)
        self.clear_points()
        self.generate_points()
        return self


class ThreeLineTable(VGroup):
    def __init__(self,
                 data:list,
                 col_widths:list|None=None,
                 h_buff:float=0.5,
                 v_buff:float=0.4,
                 font_color:ManimColor=BLACK,
                 line_color:ManimColor=BLACK,
                 base_line_width=0.1,    # 基线（细线）宽度
                 header_font_size=24,
                 content_font_size=20,
                 alignments=None,
                 vertical_lines:list[int]=None,
                 font:str="Dream Han Serif CN",
                 **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.h_buff = h_buff
        self.v_buff = v_buff
        self.header_font_size = header_font_size
        self.content_font_size = content_font_size

        self.base_line_width = base_line_width

        self.font_color = font_color
        self.line_color = line_color

        self.font = font

        if not data or not all(len(row) == len(data[0]) for row in data):
            raise ValueError("表格数据必须是非空且每行长度一致的二维列表")

        self.num_cols = len(data[0])
        self.num_rows = len(data)

        # 对齐方式
        if alignments is None:
            self.alignments = ['c'] * self.num_cols
        else:
            if len(alignments) != self.num_cols:
                raise ValueError("对齐方式列表长度必须与列数相同")
            self.alignments = alignments

        # 处理列左侧竖线配置
        self.vertical_lines = self._handle_vertical_lines(vertical_lines)

        # 处理列宽
        self.col_widths = self._handle_col_widths(col_widths)

        # 创建表格元素
        self.headers = self._create_headers()
        self.cells = self._create_cells()

        self._calculate_dimensions()  # 计算表格尺寸参数
        self._create_rules()  # 包含横线和竖线

        # 排列表格元素
        self._arrange_elements()

        # 添加所有元素到组
        self.add(self.top_rule, self.mid_rule, self.bottom_rule)
        self.add(*self.v_lines)  # 添加竖线
        self.add(*self.headers)
        self.add(*self.cells)

    def _handle_vertical_lines(self, vertical_lines):
        """处理列左侧竖线配置"""
        if vertical_lines is None:
            return []
        # 验证列索引有效性
        for col in vertical_lines:
            if not (0 <= col < self.num_cols):
                raise ValueError(f"无效的列索引{col}，列索引应在0到{self.num_cols - 1}之间")
        return list(set(vertical_lines))  # 去重

    def _handle_col_widths(self, col_widths):
        """处理列宽，若未指定则自动计算"""
        if col_widths is None:
            # 自动计算列宽（基于最长文本）
            col_widths = []
            for col in range(self.num_cols):
                max_width = 0
                for row in range(self.num_rows):
                    text = TextMobject(str(self.data[row][col]), font="Times New Roman")
                    width = text.get_width()
                    if width > max_width:
                        max_width = width
                col_widths.append(max_width + self.h_buff)
        else:
            if len(col_widths) != self.num_cols:
                raise ValueError("列宽列表长度必须与列数相同")
        return col_widths

    def _calculate_dimensions(self):
        """计算表格关键尺寸参数"""
        self.total_width = sum(self.col_widths) - self.h_buff  # 总宽度
        self.total_height = self.num_rows * self.v_buff  # 总高度
        # 计算每列左侧的x坐标位置
        self.col_left_x = []
        current_x = -self.total_width / 2
        for width in self.col_widths:
            self.col_left_x.append(current_x)
            current_x += width

    def _create_headers(self):
        """创建表头单元格"""
        headers = []
        for col in range(self.num_cols):
            text = Text(
                str(self.data[0][col]),
                font=self.font,
                font_size=self.header_font_size,
                color=self.font_color
            )
            headers.append(text)
        return VGroup(*headers)

    def _create_cells(self):
        """创建内容单元格"""
        cells = []
        for row in range(1, self.num_rows):  # 从第二行开始是内容
            row_cells = []
            for col in range(self.num_cols):
                text = Text(
                    str(self.data[row][col]),
                    font=self.font,
                    font_size=self.content_font_size,
                    color=self.font_color
                )
                row_cells.append(text)
            cells.extend(row_cells)
        return VGroup(*cells)

    def _create_rules(self):
        """创建三条横线和指定的竖线"""
        # 横线
        self.top_rule = Line(
            LEFT * (self.total_width / 2),
            RIGHT * (self.total_width / 2),
            stroke_width=2*self.base_line_width,
            color=self.line_color
        )

        self.mid_rule = Line(
            LEFT * (self.total_width / 2),
            RIGHT * (self.total_width / 2),
            stroke_width=self.base_line_width,
            color=self.line_color
        )

        self.bottom_rule = Line(
            LEFT * (self.total_width / 2),
            RIGHT * (self.total_width / 2),
            stroke_width=2*self.base_line_width,
            color=self.line_color
        )

        # 竖线（列左侧的细线）
        self.v_lines = VGroup()
        line_height = self.total_height# - self.v_buff / 2  # 竖线总高度
        for col in self.vertical_lines:
            x_pos = self.col_left_x[col]
            v_line = Line(
                UP * (self.v_buff / 2),
                DOWN * (line_height - self.v_buff / 2),
                stroke_width=self.base_line_width,
                color=self.line_color
            )
            v_line.shift(RIGHT * x_pos)
            self.v_lines.add(v_line)

    def _arrange_elements(self):
        """排列表格所有元素"""
        # 排列表头
        for i, header in enumerate(self.headers):
            # 计算列中心x坐标
            col_center_x = self.col_left_x[i] + self.col_widths[i] / 2
            header.move_to(col_center_x * RIGHT)

            # 根据对齐方式调整位置
            if self.alignments[i] == 'l':
                header.shift(self.col_widths[i] / 2 * LEFT)
            elif self.alignments[i] == 'r':
                header.shift(self.col_widths[i] / 2 * RIGHT)

        # 排列内容单元格
        for row in range(self.num_rows - 1):
            for col in range(self.num_cols):
                idx = row * self.num_cols + col
                cell = self.cells[idx]
                # 计算列中心x坐标和行中心y坐标
                col_center_x = self.col_left_x[col] + self.col_widths[col] / 2
                row_center_y = -(row + 1) * self.v_buff

                cell.move_to(col_center_x * RIGHT + row_center_y * UP)

                # 根据对齐方式调整位置
                if self.alignments[col] == 'l':
                    cell.shift(self.col_widths[col] / 2 * LEFT)
                elif self.alignments[col] == 'r':
                    cell.shift(self.col_widths[col] / 2 * RIGHT)

        # 定位横线
        self.top_rule.move_to(0.5 * self.v_buff * UP)
        self.mid_rule.move_to(0.5 * self.v_buff * DOWN)
        self.bottom_rule.move_to(((self.num_rows - 1) * self.v_buff + 0.5 * self.v_buff) * DOWN)

class FlowBox(VGroup):
    # 产生流程图box
    def __init__(self,
            text:str,   # 文本
            font_size:int = 18,  # 字号
            font_color:ManimColor = BLUE_COLOR, # 文本颜色
            shape:int = 0,  # 形状，0-矩形，1-圆角矩形，2-平行四边形，3-菱形
            box_width:float = 2,  # 宽度
            box_height:float = 0.4,   # 高度
            stroke_width:float = 1,   # 描边宽度
            stroke_color:ManimColor = BLUE_COLOR,   # 描边颜色
            stroke_opacity: float = 1,  # 描边不透明度
            fill_color:ManimColor = YELLOW, # 填充颜色
            fill_opacity: float = 0,    #填充不透明度
            **kwargs
    ):
        self.box_width = box_width
        self.box_height = box_height
        self.shape = shape
        self.center_skew = 0   # 平行四边形水平中心偏移量
        text_obj = Text(
            text=text,
            font_size=font_size,
            font="Dream Han Serif CN",
            color=font_color
        )
        match shape:
            case 0:
                box_obj = Rectangle(width=box_width, height=box_height)
            case 1:
                box_obj = RoundedRectangle(corner_radius=self.box_height/2, width=box_width, height=box_height)
            case 2:
                angle = 15 * DEGREES
                skew = box_height * np.tan(angle)
                self.center_skew = skew/2
                box_obj = Polygon([-box_width/2+skew, box_height/2, 0], [box_width/2, box_height/2, 0],
                                  [box_width/2-skew, -box_height/2, 0], [-box_width/2, -box_height/2, 0])
            case 3:
                box_obj = Polygon([0, box_height/2, 0], [box_width/2, 0, 0],
                                  [0, -box_height/2, 0], [-box_width/2, 0, 0])
            case _:
                raise ValueError(f"形状参数{shape}非法")
        box_obj.set_stroke(color=stroke_color, width=stroke_width, opacity=stroke_opacity)
        box_obj.set_fill(color=fill_color, opacity=fill_opacity)
        super().__init__(text_obj, box_obj)

class FlowPoint(VGroup):
    # 虚拟空对象，便于统一FlowArrow连接的对象
    def __init__(self):
        super().__init__(Dot(fill_opacity=0, radius=0))
        self.box_height = 0
        self.box_width = 0
        self.center_skew = 0

class FlowArrow(VGroup):
    # 产生箭头
    def __init__(self,
                 obj1:FlowBox|FlowPoint,    # 起点对象
                 obj2:FlowBox|FlowPoint,    # 终点对象
                 direct1=DOWN,   # 起点对象方向
                 direct2=UP,     # 终点对象方向
                 line_color:ManimColor=BLUE_COLOR,
                 stroke_width:float=1,
    ):
        self.obj1 = obj1
        self.obj2 = obj2
        self.direct1 = direct1
        self.direct2 = direct2

        self.line_color = line_color
        self.stroke_width = stroke_width

        super().__init__(self.get_arrow())

    def get_arrow(self)->VGroup:
        # 产生箭头
        point_ls:list[np.ndarray] = []

        tip_width = 0.05
        tip_length = 0.1
        tip_buff=0.03

        for obj, direct in [(self.obj1, self.direct1), (self.obj2, self.direct2)]:
            if direct[0] == 0:  # direct in (UP, DOWN)
                point_ls.append(obj.get_center()+direct*obj.height/2)
            else:
                point_ls.append(obj.get_center()+direct*(obj.box_width/2-obj.center_skew))
        if point_ls[0][0] == point_ls[1][0] or point_ls[0][1] == point_ls[1][1]:
            # 水平或垂直直线，直接返回单个箭头
            return VGroup(
                Line(
                    start=point_ls[0],
                    end=point_ls[1]+self.direct2*tip_buff,
                    color=self.line_color,
                    stroke_width=self.stroke_width,
                    buff=0,
                ).add_tip(tip_shape=StealthTip, tip_length=tip_length, tip_width=tip_width)
            )
        else:
            # 需要转折
            if self.direct1[0] == 0:    # self.direction1 in (UP, DOWN)
                trans_point = [point_ls[0][0], point_ls[1][1], 0]
            else:
                trans_point = [point_ls[1][0], point_ls[0][1], 0]
            return VGroup(
                Line(
                    start=point_ls[0],
                    end=trans_point,
                    color=self.line_color,
                    stroke_width=self.stroke_width
                ),
                Line(
                    start=trans_point,
                    end=point_ls[1]+self.direct2*tip_buff,
                    color=self.line_color,
                    stroke_width=self.stroke_width,
                ).add_tip(tip_shape=StealthTip, tip_length=tip_length, tip_width=tip_width)
            )

class Histogram(VGroup):
    # 直方图，仅包含直方图
    def __init__(
            self,
            data:np.ndarray[float],    # 数据
            axis_range:list[float, float], # 坐标轴范围
            axis_step:float=1,              # 坐标轴刻度步长
            bin_num:int=10,     # 划分数量
            size:list[float, float]=(2, 3),  # 直方图大小
            axes_color:ManimColor=WHITE,
            font_size:float=18,

            bar_fill_color:ManimColor=WHITE,
            bar_fill_opacity:float=1,
            bar_stroke_color:ManimColor=WHITE,
            bar_stroke_opacity:float=1,
            bar_stroke_width:float=1.,  # 描边宽度

            box_stroke_color:ManimColor=WHITE,
            box_stroke_width:float=0.3, # 散点盒描边宽度
            boxline_color:ManimColor=WHITE, # 散点盒内线条颜色
            boxline_width:float=0.1,    # 散点盒内线宽度

            box_buff:float = 0.2,   # 散点盒离轴线距离
            box_width:float = 0.4,  # 散点盒宽度
            **kwargs
    ):
        super().__init__(**kwargs)
        # 数据存储
        self.data = data
        self.y_range = axis_range
        self.y_step = axis_step
        self.bin_num = bin_num
        self.size = size
        self.axes_color = axes_color
        self.font_size = font_size

        # 直方图参数
        self.bar_fill_color = bar_fill_color
        self.bar_fill_opacity = bar_fill_opacity
        self.bar_stroke_color = bar_stroke_color
        self.bar_stroke_opacity = bar_stroke_opacity
        self.bar_stroke_width = bar_stroke_width

        # 散点图参数
        self.box_stroke_color = box_stroke_color
        self.box_stroke_width = box_stroke_width
        self.boxline_color = boxline_color
        self.boxline_width = boxline_width
        self.box_buff = box_buff
        self.box_width = box_width

        self.setup_data()
        self.setup_fig()

    def setup_data(self):
        # 数据预处理
        self.counts, self.bin_edges = np.histogram(self.data, bins=self.bin_num, range=self.y_range)
        self.bin_widths = np.diff(self.bin_edges)
        self.x_range = [-max(self.counts),0]

    def setup_fig(self):
        # 坐标轴
        axes = Axes(
            x_range=self.x_range,
            y_range=[*self.y_range, self.y_step],
            x_length=self.size[0],
            y_length=self.size[1],
            x_axis_config={
                "stroke_width": 0,
                "include_numbers": False,
                "include_ticks":False,
                "include_tip":False,
            },
            y_axis_config={
                "include_ticks": True,
                "include_tip":False,
                "include_numbers":True,
                "label_direction":RIGHT,
                "font_size":self.font_size
            }
        )
        num_axes = NumberLine(
            x_range=axes.y_range,
            length=axes.y_length,
            include_numbers=True,
            include_ticks=True,
            label_direction=RIGHT,
            include_tip=False,
            font_size=self.font_size,
            rotation=90*DEGREES
        )
        num_axes.shift(axes.c2p(0,0,0)-num_axes.n2p(0))

        # 柱子
        bar = VGroup()
        for idx, (x_value, y_value) in enumerate(zip(self.counts, self.bin_edges[:-1])):
            bar.add(Polygon(
                axes.c2p(0, y_value, 0),
                axes.c2p(0, y_value+self.bin_widths[idx], 0),
                axes.c2p(-x_value, y_value+self.bin_widths[idx], 0),
                axes.c2p(-x_value, y_value, 0),
                fill_color=self.bar_fill_color,
                fill_opacity=self.bar_fill_opacity,
                stroke_width=self.bar_stroke_width,
                stroke_color=self.bar_stroke_color,
                stroke_opacity=self.bar_stroke_opacity
            ))
        # 散点
        buff = self.box_buff  # 散点与轴线间距
        box_width = self.box_width # 散点图宽度
        box = VGroup(Polygon(
            axes.c2p(0, self.y_range[0], 0) + RIGHT * buff,
            axes.c2p(0, self.y_range[1], 0) + RIGHT * buff,
            axes.c2p(0, self.y_range[1], 0) + RIGHT * (buff + box_width),
            axes.c2p(0, self.y_range[0], 0) + RIGHT * (buff + box_width),
            color=self.box_stroke_color,
            stroke_width=self.box_stroke_width,
        ))
        for num in self.data:
            if self.y_range[0] <= num <= self.y_range[1]:
                box.add(Line(
                    start=axes.c2p(0, num, 0) + RIGHT * buff,
                    end=axes.c2p(0, num, 0) + RIGHT * (buff + box_width),
                    color=self.boxline_color,
                    stroke_width=self.boxline_width,
                ))

        self.add(bar, box, num_axes) # bar在前，axes在后，防止bar遮挡axes
        self.obj_axes = axes    # 把axes作为属性传出去，便于其他对象访问
        self.obj_numaxes = num_axes # 把num_axes也跟着传出去，用的时候方便点


# contour图，用于二维寻优可视化
class Contour(VGroup):
    def __init__(
            self,
            func: Callable[[float], float], # 绘制函数
            x_range: list[float],
            y_range: list[float],
            x_length: float,
            y_length: float,
            x_axis_config: dict = {},
            y_axis_config: dict = {},
            axis_config: dict = {"include_tip": False},
            z_range: list[float] = [0, 1, 0.1],
            colors: list[ManimColor] = [WHITE, BLACK],    # 颜色表
            include_color_bar: bool = True,         # 包含颜色条
            sep_num: int = 100, #  分隔数目
            **kwargs
    ):
        self.func = func
        self.x_range = x_range
        self.y_range = y_range
        self.x_length = x_length
        self.y_length = y_length
        self.z_range = z_range
        self.colors = colors
        self.include_color_bar = include_color_bar
        self.sep_num = sep_num

        self.path_point = VGroup()
        self.now_point = VGroup()

        self.axes = ThreeDAxes(
            x_range=x_range,
            x_length=x_length,
            y_range=y_range,
            y_length=y_length,
            x_axis_config=x_axis_config,
            y_axis_config=y_axis_config,
            z_range=z_range,
            z_length=0.1,
            axis_config=axis_config,
            **kwargs
        )

        super().__init__()
        self._add_fig()
        self._add_axes()
        self._add_color_bar()

    def _add_fig(self):
        self.surface = Surface(
            lambda u, v: self.axes.c2p(u, v, self.func(u, v)),
            u_range=[*self.x_range[:-1],],
            v_range=[*self.y_range[:-1],],
            resolution=(self.sep_num, self.sep_num),
        )
        self.surface.set_fill_by_value(
            axes=self.axes,
            colorscale=self.colors,
        )
        self.add(self.surface)
        return self.surface

    def _add_axes(self):
        self.add(self.axes)
        return self.axes

    def _add_color_bar(self, add_color_bar: bool=True):
        self.color_bar = VGroup()
        height = self.y_length/2
        width = 0.3
        sep_num = 150
        if add_color_bar:
            grad_vgp = VGroup()
            for color_now in color_gradient(self.colors[::-1], 150):
                grad_vgp.add(Rectangle(
                    color=color_now,
                    width=width,
                    height=height/sep_num,
                ))
            grad_vgp.arrange(DOWN, buff=0)
            self.color_bar.add(
                Rectangle(
                    height=height,
                    width=width,
                    stroke_color=WHITE,
                    stroke_width=1,
                    fill_opacity=0
                ).move_to(grad_vgp.get_center()),
                grad_vgp
            )
            num_line = NumberLine(
                x_range=self.z_range,
                length=height,
                include_ticks=True,
                include_numbers=True,
                font_size=14,
                include_tip=False,
                label_direction=RIGHT,
                rotation=90*DEGREES
            ).next_to(grad_vgp, direction=RIGHT, buff=0.2)
            self.color_bar.add(num_line)
            self.color_bar.next_to(self.axes, direction=RIGHT, buff=0.6)
            self.add(self.color_bar)
            return self.color_bar

    # 添加约束条件范围提示
    def add_band(
            self,
            band_path: list[tuple[float, float]],
            **kwargs
    )->Polygon:
        self.band = Polygon(
            *[self.axes.c2p(*point, 0) for point in band_path],
            **kwargs,
        )
        self.add(self.band)
        return self.band

    # 用于刷新路径点，放在循环里，减少多次产生图像的不必要运行
    def update_path_point(
            self,
            path_point: list[tuple[float, float]],
            radius=0.04,
            color=YELLOW_COLOR,
            fill_opacity=0.6,
            line_stroke_width=1,
            **kwargs
    )->VGroup:
        # 精英个体历代路线
        self.remove(self.path_point)
        self.path_point = VGroup(*[
            Line(
                start=self.axes.c2p(*path_point[front_idx], 0),
                end=self.axes.c2p(*path_point[front_idx+1]),
                color=color,
                stroke_width=line_stroke_width
            )
        for front_idx in range(len(path_point)-1)])
        self.path_point.add(VGroup(*[
            Star(
                outer_radius=radius,
                color=color,
                fill_opacity=fill_opacity,
                **kwargs
            ).move_to(self.axes.c2p(*point, 0))
        for point in path_point]))
        self.add(self.path_point)
        return self.path_point

    def update_now_point(
            self,
            point_ls: list[tuple[float, float]],
            radius=0.02,
            color=YELLOW_COLOR,
            fill_opacity=0.9,
            **kwargs
    )->VGroup:
        # 当代
        self.remove(self.now_point)
        self.now_point = VGroup(*[
            Dot(
                point=self.axes.c2p(*point, 0),
                radius=radius,
                color=color,
                fill_opacity=fill_opacity,
                **kwargs
            )
        for point in point_ls])
        self.add(self.now_point)
        return self.now_point
