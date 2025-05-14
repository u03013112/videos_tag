# 颜色分组方案
# 色相环 方案，每隔30度分一组，(0~30],(30~60],(60~90],(90~120],(120~150],(150~180],(180~210],(210~240],(240~270],(270~300],(300~330],(330~360]
# 饱和度 高饱和度（80%~100%），中饱和度（40%~79%），低饱和度（0%~39%）
# 明度 高明度（70%~100%），中明度（30%~69%），低明度（0%~29%）
# 将颜色分为 12 * 3 * 3 = 108 种颜色
# 命名为 c0，c1，c2，...，c107
# 然后遍历 RGB [0,0,0]~[255,255,255] ，每一种颜色 判断是否属于 某一种颜色
# 需要记录：1、是否有颜色没有匹配到任何颜色分组；2、是否有颜色匹配到多个颜色分组；
# 输出 1和2 的计数

from colorsys import rgb_to_hls

import os
import cv2
import pandas as pd
import numpy as np
import datetime
import joblib

def get_hsl_color_ranges():
    color_ranges = {}
    for i in range(12):  # Hue groups (0 to 179, 15 degrees per group)
        hue_lower = i * 15
        hue_upper = (i + 1) * 15 - 1
        for j in range(3):  # Saturation groups (0 to 255)
            if j == 0:
                sat_lower, sat_upper = int(80 / 100 * 255), 255
            elif j == 1:
                sat_lower, sat_upper = int(40 / 100 * 255), int(79 / 100 * 255)
            else:
                sat_lower, sat_upper = 0, int(39 / 100 * 255)
            for k in range(3):  # Lightness groups (0 to 255)
                if k == 0:
                    light_lower, light_upper = int(70 / 100 * 255), 255
                elif k == 1:
                    light_lower, light_upper = int(30 / 100 * 255), int(69 / 100 * 255)
                else:
                    light_lower, light_upper = 0, int(29 / 100 * 255)
                
                color_tag = f'c{i * 9 + j * 3 + k}'
                lower_bound = np.array([hue_lower, light_lower, sat_lower], dtype=np.uint8)
                upper_bound = np.array([hue_upper, light_upper, sat_upper], dtype=np.uint8)
                color_ranges[color_tag] = (lower_bound, upper_bound)
    
    return color_ranges

def findColorTagfromFrame(frame_dir,filename):
    frames = []
    sorted_filenames = sorted(os.listdir(frame_dir))
    for frame_filename in sorted_filenames:
        if frame_filename.endswith('.jpg'):
            frame_path = os.path.join(frame_dir, frame_filename)
            frame = cv2.imread(frame_path)
            frames.append(frame)
    
    color_ranges = get_hsl_color_ranges()
    color_ratios = {color: 0 for color in color_ranges.keys()}
    # 遍历每一帧,计算颜色分组像素占比
    for frame in frames:
        # 将图像转换为HSL颜色空间
        hls_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        
        # 计算每种颜色的像素占比
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hls_frame, lower, upper)
            ratio = cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])
            color_ratios[color] += ratio / len(frames)
    
    # 计算其他颜色的占比
    other_ratio = 1 - sum(color_ratios.values())

    # 输出dataframe，列名为：filename, color, ratio
    result_df = pd.DataFrame([
        {'filename': filename, 'color': color, 'ratio': ratio}
        for color, ratio in color_ratios.items()
    ] + [{'filename': filename, 'color': 'other', 'ratio': other_ratio}])
    
    
    return result_df



# def main():
    

if __name__ == "__main__":
    # main()
    # print(get_hsl_color_ranges())
    print(findColorTagfromFrame('/src/data/videos/20250421_1_frames', '20250421_1.mp4'))
