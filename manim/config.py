from manim import *
from manim import ManimColor

BLUE_COLOR = ManimColor("#1541e6")
YELLOW_COLOR = ManimColor("#f5d42c")
RED_COLOR = ManimColor("#c92200")
CYAN_COLOR = ManimColor("#35dcff")
PURPLE_COLOR = ManimColor("#565bf3")

flash_time = 4/60   # 快闪时间，4Frame，60FpS

code_kargs = {  # Code类配置参数
    'formatter_style':'paraiso-light',
    'add_line_numbers':False,
    'paragraph_config':{'font_size':10,
                        'line_spacing':0.3},
    'background_config':{'corner_radius':0,
                         'fill_color':WHITE,
                         'stroke_color':BLUE,
                         'buff':0.1}
}