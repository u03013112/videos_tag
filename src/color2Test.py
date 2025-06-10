from colorsys import rgb_to_hls

import os
import cv2
import pandas as pd
import numpy as np
import datetime
import joblib
# 分108种颜色
def get_hsl_color_ranges():
    color_ranges = {}
    for i in range(12):  # Hue groups (0 to 179, 15 degrees per group)
        hue_lower = i * 15
        hue_upper = (i + 1) * 15 - 1
        if hue_upper == 179:
            hue_upper = 180
        for j in range(3):  # Saturation groups (0 to 255)
            if j == 0:
                sat_lower, sat_upper = int(79 / 100 * 255)+1, 255
            elif j == 1:
                sat_lower, sat_upper = int(39 / 100 * 255)+1, int(79 / 100 * 255)
            else:
                sat_lower, sat_upper = 0, int(39 / 100 * 255)
            for k in range(3):  # Lightness groups (0 to 255)
                if k == 0:
                    light_lower, light_upper = int(69 / 100 * 255)+1, 255
                elif k == 1:
                    light_lower, light_upper = int(29 / 100 * 255)+1, int(69 / 100 * 255)
                else:
                    light_lower, light_upper = 0, int(29 / 100 * 255)
                
                color_tag = f'c{i * 9 + j * 3 + k}'
                lower_bound = np.array([hue_lower, light_lower, sat_lower], dtype=np.uint8)
                upper_bound = np.array([hue_upper, light_upper, sat_upper], dtype=np.uint8)
                color_ranges[color_tag] = (lower_bound, upper_bound)
    
    return color_ranges

# 分24种颜色
def get_hsl_color_ranges2():
    color_ranges = {}
    count = 0
    for i in range(6):
        hue_lower = i * 30
        hue_upper = (i + 1) * 30 - 1
        if hue_upper == 179:
            hue_upper = 180
        for j in range(2):
            if j == 0:
                sat_lower, sat_upper = int(50 / 100 * 255)+1, 255
            else:
                sat_lower, sat_upper = 0, int(50 / 100 * 255)

            for k in range(2):
                if k == 0:
                    light_lower, light_upper = int(50 / 100 * 255)+1, 255
                else:
                    light_lower, light_upper = 0, int(50 / 100 * 255)
                
                color_tag = f'c{count}'
                count += 1
                lower_bound = np.array([hue_lower, light_lower, sat_lower], dtype=np.uint8)
                upper_bound = np.array([hue_upper, light_upper, sat_upper], dtype=np.uint8)
                color_ranges[color_tag] = (lower_bound, upper_bound)
    
    return color_ranges

def findColorTagfromFrame(frame_dir,filename,color_ranges):
    frames = []
    sorted_filenames = sorted(os.listdir(frame_dir))
    for frame_filename in sorted_filenames:
        if frame_filename.endswith('.jpg'):
            frame_path = os.path.join(frame_dir, frame_filename)
            frame = cv2.imread(frame_path)
            frames.append(frame)
    
    # color_ranges = get_hsl_color_ranges()
    # color_ranges = get_hsl_color_ranges_for_test()
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
            # print('cv2.countNonZero(mask):',cv2.countNonZero(mask))
            # print('frame.shape[0] * frame.shape[1]:',frame.shape[0] * frame.shape[1])
            # print('ratio:',ratio)
            # print(color_ratios[color])

            # # 创建反掩码
            # inverse_mask = cv2.bitwise_not(mask)
            
            # # 获取掩码外的像素坐标
            # out_of_range_pixels = np.where(inverse_mask != 0)
            
            # # 打印掩码外的像素的 HLS 值
            # for y, x in zip(out_of_range_pixels[0], out_of_range_pixels[1]):
            #     h, l, s = hls_frame[y, x]
            #     print(f"Pixel at ({y}, {x}) - H: {h}, L: {l}, S: {s}")

            # return
    
    # 计算其他颜色的占比
    other_ratio = 1 - sum(color_ratios.values())

    # 输出dataframe，列名为：filename, color, ratio
    result_df = pd.DataFrame([
        {'filename': filename, 'color': color, 'ratio': ratio}
        for color, ratio in color_ratios.items()
    ] + [{'filename': filename, 'color': 'other', 'ratio': other_ratio}])
    
    
    return result_df

def addTag(videoInfoDf,color_ranges):
    colorTagDf = pd.DataFrame()
    for i in range(len(videoInfoDf)):
        filename = videoInfoDf.loc[i, 'filename']
        frames_dir = videoInfoDf.loc[i, 'frames_dir']

        colorTagDf0 = findColorTagfromFrame(frames_dir, filename, color_ranges)
        colorTagDf0 = colorTagDf0.pivot(index='filename', columns='color', values='ratio').reset_index()
        colorTagDf = pd.concat([colorTagDf, colorTagDf0], ignore_index=True)

    videoInfoDf = pd.merge(videoInfoDf, colorTagDf, on='filename', how='left')

    return videoInfoDf

def makeCsv(color_ranges):
    csvDir = '/src/data/csv/color2Tag'
    if not os.path.exists(csvDir):
        os.makedirs(csvDir)

    downloadCsvDir = '/src/data/csv/download'
    downloadCsvFilenameList = os.listdir(downloadCsvDir)
    downloadCsvFilenameList = sorted(downloadCsvFilenameList)
    # print('downloadCsvFilenameList:')
    # print(downloadCsvFilenameList)

    for downloadCsvFilename in downloadCsvFilenameList:
        if downloadCsvFilename.endswith('.csv'):
            # print('downloadCsvFilename:', downloadCsvFilename)
            # 找到downloadCsvFilename的文件名，不包含路径
            # 检查这个文件名在csvDir中是否存在，如果存在就跳过
            csvFilename = os.path.join(csvDir, downloadCsvFilename)
            if os.path.exists(csvFilename):
                # print(f"{csvFilename} already exists, skipping.")
                continue

            videoInfoDf = pd.read_csv(os.path.join(downloadCsvDir, downloadCsvFilename))
            videoInfoDf = addTag(videoInfoDf,color_ranges)

            videoInfoDf.to_csv(csvFilename, index=False)
            print(f"Processed {downloadCsvFilename} and saved to {csvFilename}")

def predict(data,color_ranges):
    testDf = data.copy()
    model_dir = '/src/data/models/'
    model_path = os.path.join(model_dir, 'color2_decision_tree1.pkl')
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist.")
        return
    model = joblib.load(model_path)

    color_features = list(color_ranges.keys())
    X = testDf[color_features]  # 输入特征
    y = np.zeros(testDf.shape[0])
    y[testDf['cost'] > 3500] = 1
    
    print('共有', len(y), '条数据')
    print('畅销素材数量:', len(y[y == 1]))
    print('畅销比例:', len(y[y == 1]) / len(y))

    # 获取预测概率
    probabilities = model.predict_proba(X)
    # 输出每个样本属于每个类别的概率
    # print(probabilities)
    # 自定义阈值
    threshold = 0.2
    testDf.loc[:, 'predicted_class'] = (probabilities[:, 1] >= threshold).astype(int)
    testDf.loc[:, 'probabilities'] = probabilities[:, 1]
    testDf.loc[:, 'y_test'] = y

    # 计算测试集中预测为畅销素材的数量
    predicted_hot_count = len(testDf[testDf['predicted_class'] == 1])
    print('测试集中预测为畅销素材的数量:', predicted_hot_count)
    # 计算测试集中预测为畅销素材并且真的畅销素材的数量
    true_predicted_hot_count = len(testDf[(testDf['predicted_class'] == 1) & (y == 1)])
    print('测试集中预测为畅销素材并且真的畅销素材的数量:', true_predicted_hot_count)
    # 计算测试集中预测为不畅销素材的数量
    predicted_not_hot_count = len(testDf[testDf['predicted_class'] == 0])
    print('测试集中预测为不畅销素材的数量:', predicted_not_hot_count)
    # 计算测试集中预测为不畅销但是真实畅销素材的数量
    true_predicted_not_hot_count = len(testDf[(testDf['predicted_class'] == 0) & (y == 1)])
    print('测试集中预测为不畅销但是真实畅销素材的数量:', true_predicted_not_hot_count)
    # 计算测试集中预测为畅销素材的准确率
    accuracy1 = true_predicted_hot_count / predicted_hot_count if predicted_hot_count > 0 else 0
    print('测试集中预测为畅销素材的准确率:', accuracy1)
    # 计算测试集中预测为不畅销素材的准确率
    accuracy2 = true_predicted_not_hot_count / predicted_not_hot_count if predicted_not_hot_count > 0 else 0
    print('测试集中预测为不畅销素材的准确率:', accuracy2)
    if accuracy2 == 0:
        print('测试集中没有预测为不畅销素材')
        return
    print('测试集中预测为畅销素材准确率/不畅销素材准确率:', accuracy1 / accuracy2)
    
    

def main(dayStr = None):

    if dayStr is None:
        today = datetime.datetime.now()
    else:
        today = datetime.datetime.strptime(dayStr, '%Y%m%d')

    # 如果不是周一，什么都不做
    if today.weekday() != 0:
        print("今天不是周一，不执行数据准备。")
        return
    
    df = pd.DataFrame()

    # 这里的周一其实是上周一，周一至周日是上周，这样才能获得完整数据。
    monday = today - datetime.timedelta(days=7)
    sunday = monday + datetime.timedelta(days=6)
    lastMonday = monday - datetime.timedelta(days=7)
    lastSunday = monday - datetime.timedelta(days=1)
    
    mondayStr = monday.strftime('%Y%m%d')
    sundayStr = sunday.strftime('%Y%m%d')
    lastMondayStr = lastMonday.strftime('%Y%m%d')
    lastSundayStr = lastSunday.strftime('%Y%m%d')

    print('当前周一：', mondayStr)
    # print('当前周日：', sundayStr)
    # print('上周一：', lastMondayStr)
    # print('上周日：', lastSundayStr)

    color_ranges = get_hsl_color_ranges2()
    makeCsv(color_ranges)

    data = pd.read_csv(f'/src/data/csv/color2Tag/{mondayStr}.csv')
    predict(data, color_ranges)

# 历史数据补充，如果有需要补充的历史数据，调佣这个函数，并且调整时间范围
def historyData():
    startDayStr = '20250101'
    endDayStr = '20250609'

    startDay = datetime.datetime.strptime(startDayStr, '%Y%m%d')
    endDay = datetime.datetime.strptime(endDayStr, '%Y%m%d')

    for i in range((endDay - startDay).days + 1):
        day = startDay + datetime.timedelta(days=i)
        dayStr = day.strftime('%Y%m%d')
        print(dayStr)
        main(dayStr)

def debug():
    color_ranges = get_hsl_color_ranges2()
    csvDir = '/src/data/csv/color2Tag'

    tagCsvFilenameList = os.listdir(csvDir)
    tagCsvFilenameList = sorted(tagCsvFilenameList)

    df = pd.DataFrame()
    for tagCsvFilename in tagCsvFilenameList:
        if tagCsvFilename.endswith('.csv'):
            print('tagCsvFilename:', tagCsvFilename)
            data = pd.read_csv(os.path.join(csvDir, tagCsvFilename))
            data['day'] = tagCsvFilename.split('.')[0]
            df = pd.concat([df, data], ignore_index=True)

    # df['day'] = pd.to_datetime(df['day'], format='%Y%m%d')
    # df['month'] = df['day'].dt.month
    # 直接将day的最后2个字符去除，作为月份
    df['month'] = df['day'].str[:-2]

    monthList = df['month'].unique()
    monthList = sorted(monthList)
    print('monthList:', monthList)

    for month in monthList:
        monthDf = df[df['month'] == month]
        print(f"Month {month}")

        predict(monthDf, color_ranges)
        print('-' * 50)



if __name__ == '__main__':
    # historyData()
    # main('20250609')

    debug()
    