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



# 为了测试，只分配一个颜色范围
def get_hsl_color_ranges_for_test():
    color_ranges = {}
    # 将所有可能的 HLS 值组合都分配到一个标签 'c1'
    # Hue: 0 to 179
    # Lightness: 0 to 255
    # Saturation: 0 to 255
    # hue_lower, hue_upper = 0, 179
    hue_lower, hue_upper = 0, 180
    light_lower, light_upper = 0, 255
    sat_lower, sat_upper = 0, 255
    
    color_tag = 'c1'
    lower_bound = np.array([hue_lower, light_lower, sat_lower], dtype=np.uint8)
    upper_bound = np.array([hue_upper, light_upper, sat_upper], dtype=np.uint8)
    color_ranges[color_tag] = (lower_bound, upper_bound)
    
    return color_ranges

def color_check(color_ranges):
    # 图像的宽度和每个颜色条的高度
    img_width = 100
    bar_height = 5
    
    # 计算图像的总高度
    img_height = len(color_ranges) * bar_height
    
    # 创建空白图像
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # 绘制每个颜色条
    for idx, (color_tag, (lower, upper)) in enumerate(color_ranges.items()):
        for x in range(img_width):
            # 计算当前像素的 HLS 值
            hue = int(lower[0] + (upper[0] - lower[0]) * (x / img_width))
            lightness = int(lower[1] + (upper[1] - lower[1]) * (x / img_width))
            saturation = int(lower[2] + (upper[2] - lower[2]) * (x / img_width))
            
            # 设置颜色条的颜色
            color = np.array([hue, lightness, saturation], dtype=np.uint8)
            img[idx * bar_height:(idx + 1) * bar_height, x] = cv2.cvtColor(
                np.array([[color]], dtype=np.uint8), cv2.COLOR_HLS2BGR)[0][0]
    
    # # 显示图像
    # cv2.imshow('Color Ranges', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存图像
    output_path = '/src/data/color_ranges.png'
    cv2.imwrite(output_path, img)
    print(f"Color ranges image saved to {output_path}")

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

from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
def visualize_tree(model, feature_names):
    plt.figure(figsize=(40, 40))
    # plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei'] 
    plot_tree(model, feature_names=feature_names, filled=True, rounded=True)
    plt.savefig("/src/data/decision_tree_color2.png")  # 保存为 decision_tree.png

from sklearn.metrics import precision_score, recall_score, r2_score
from sklearn.tree import DecisionTreeClassifier
def fit_predict_cost_with_decision_tree(data,color_ranges,testWeeks=4):
    # 提取颜色比例特征
    color_features = list(color_ranges.keys())
    print('color_features:')
    print(color_features)

    X = data[color_features]  # 输入特征

    y = np.zeros(data.shape[0])
    y[data['cost'] > 3500] = 1
    
    print('共有', len(y), '条数据')
    print('畅销素材数量:', len(y[y == 1]))
    print('')

    earliestDayList = data['earliest_day'].tolist()
    earliestDayList = sorted(set(earliestDayList))
    print('earliestDayList:', earliestDayList)
    # 将最后testWeeks周的数据作为测试集
    lastDayStr = earliestDayList[-1]
    lastDay = datetime.datetime.strptime(lastDayStr, '%Y%m%d')
    testStartDay = lastDay - datetime.timedelta(weeks=testWeeks)
    testStartDayStr = testStartDay.strftime('%Y%m%d')
    print('testStartDayStr:', testStartDayStr)

    # 将数据拆分成训练集和测试集，将earliest_day大于等于20250324的作为测试集
    train_mask = data['earliest_day'] <= testStartDayStr

    trainDf = data[train_mask].copy()
    testDf = data[~train_mask].copy()

    X_train = X[train_mask]
    y_train = y[train_mask]

    # 测试集
    X_test = X[~train_mask]
    y_test = y[~train_mask]

    print('训练集数量:', len(X_train))
    print('训练集中畅销素材数量:', len(y_train[y_train == 1]))

    # print('测试集数量:', len(X_test))
    # print('测试集中畅销素材数量:', len(y_test[y_test == 1]))

    

    # 初始化决策树分类模型
    model = DecisionTreeClassifier(
        max_depth=3,
        # max_leaf_nodes=20  # 限制叶子节点的最大数量
    )
    
    # 拟合模型
    model.fit(X_train, y_train)
    
    # # 预测畅销素材
    # testDf.loc[:, 'predicted_class'] = model.predict(X_test)

    # 获取预测概率
    probabilities = model.predict_proba(X_test)
    # 输出每个样本属于每个类别的概率
    # print(probabilities)
    # 自定义阈值
    threshold = 0.2
    testDf.loc[:, 'predicted_class'] = (probabilities[:, 1] >= threshold).astype(int)
    testDf.loc[:, 'probabilities'] = probabilities[:, 1]
    testDf.loc[:, 'y_test'] = y_test
    
    # 计算查准率和查全率
    precision = precision_score(y_test, testDf['predicted_class'])
    recall = recall_score(y_test, testDf['predicted_class'])
    
    # 计算 R2
    r2 = r2_score(y_test, testDf['predicted_class'])

    # 获取训练集预测概率
    train_probabilities = model.predict_proba(X_train)
    # 使用与测试集相同的阈值
    trainDf.loc[:, 'predicted_class'] = (train_probabilities[:, 1] >= threshold).astype(int)
    trainDf.loc[:, 'probabilities'] = train_probabilities[:, 1]
    trainDf.loc[:, 'y_train'] = y_train
    
    # 计算训练集的查准率和查全率
    train_precision = precision_score(y_train, trainDf['predicted_class'])
    train_recall = recall_score(y_train, trainDf['predicted_class'])
    
    # 计算训练集的 R2
    train_r2 = r2_score(y_train, trainDf['predicted_class'])

    # 保存模型至/src/data/models/
    model_dir = '/src/data/models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'color1_decision_tree.pkl')
    joblib.dump(model, model_path)
    print(f'模型已保存至: {model_path}')

    visualize_tree(model, color_features)
    
    print('训练集 precision:', train_precision)
    print('训练集 recall:', train_recall)
    print('训练集 r2:', train_r2)
    
    print('测试集 precision:', precision)
    print('测试集 recall:', recall)
    print('测试集 r2:', r2)

    # 计算训练集中预测为畅销素材的数量
    predicted_hot_count = len(trainDf[trainDf['predicted_class'] == 1])
    print('训练集中预测为畅销素材的数量:', predicted_hot_count)
    
    # 计算训练集中预测为畅销素材并且真的畅销素材的数量
    true_predicted_hot_count = len(trainDf[(trainDf['predicted_class'] == 1) & (y_train == 1)])
    print('训练集中预测为畅销素材并且真的畅销素材的数量:', true_predicted_hot_count)
    # 计算训练集中预测为不畅销素材的数量
    predicted_not_hot_count = len(trainDf[trainDf['predicted_class'] == 0])
    print('训练集中预测为不畅销素材的数量:', predicted_not_hot_count)
    
    # 计算训练集中预测为不畅销但是真实畅销素材的数量
    true_predicted_not_hot_count = len(trainDf[(trainDf['predicted_class'] == 0) & (y_train == 1)])
    print('训练集中预测为不畅销但是真实畅销素材的数量:', true_predicted_not_hot_count)
    
    # 计算训练集中预测为畅销素材的准确率
    accuracy1 = true_predicted_hot_count / predicted_hot_count if predicted_hot_count > 0 else 0
    print('训练集中预测为畅销素材的准确率:', accuracy1)
    # 计算训练集中预测为不畅销素材的准确率
    accuracy2 = true_predicted_not_hot_count / predicted_not_hot_count if predicted_not_hot_count > 0 else 0
    print('训练集中预测为不畅销素材的准确率:', accuracy2)

    print('训练集中预测为畅销素材准确率/不畅销素材准确率:', accuracy1 / accuracy2)

    # 计算测试集中预测为畅销素材的数量
    predicted_hot_count = len(testDf[testDf['predicted_class'] == 1])
    print('测试集中预测为畅销素材的数量:', predicted_hot_count)
    # 计算测试集中预测为畅销素材并且真的畅销素材的数量
    true_predicted_hot_count = len(testDf[(testDf['predicted_class'] == 1) & (y_test == 1)])
    print('测试集中预测为畅销素材并且真的畅销素材的数量:', true_predicted_hot_count)
    # 计算测试集中预测为不畅销素材的数量
    predicted_not_hot_count = len(testDf[testDf['predicted_class'] == 0])
    print('测试集中预测为不畅销素材的数量:', predicted_not_hot_count)
    # 计算测试集中预测为不畅销但是真实畅销素材的数量
    true_predicted_not_hot_count = len(testDf[(testDf['predicted_class'] == 0) & (y_test == 1)])
    print('测试集中预测为不畅销但是真实畅销素材的数量:', true_predicted_not_hot_count)
    # 计算测试集中预测为畅销素材的准确率
    accuracy1 = true_predicted_hot_count / predicted_hot_count if predicted_hot_count > 0 else 0
    print('测试集中预测为畅销素材的准确率:', accuracy1)
    # 计算测试集中预测为不畅销素材的准确率
    accuracy2 = true_predicted_not_hot_count / predicted_not_hot_count if predicted_not_hot_count > 0 else 0
    print('测试集中预测为不畅销素材的准确率:', accuracy2)
    print('测试集中预测为畅销素材准确率/不畅销素材准确率:', accuracy1 / accuracy2)
    
    # return model, data, precision, recall, r2


def train(color_ranges):

    csvDir = '/src/data/csv/color2'
    if not os.path.exists(csvDir):
        os.makedirs(csvDir)

    filename = f'{csvDir}/videoWithColor2Tag.csv'
    if os.path.exists(filename):
        print('文件已存在，直接读取')
    else:
        downloadCsvDir = '/src/data/csv/download'
        downloadCsvFilenameList = os.listdir(downloadCsvDir)
        downloadCsvFilenameList = sorted(downloadCsvFilenameList)
        print('downloadCsvFilenameList:')
        print(downloadCsvFilenameList)
        
        df = pd.DataFrame()

        for filename in downloadCsvFilenameList:
            if filename.endswith('.csv'):
                print('filename:', filename)
                videoInfoDf = pd.read_csv(os.path.join(downloadCsvDir, filename))
                # print(videoInfoDf)
                videoInfoDf = addTag(videoInfoDf,color_ranges)
                
                df = pd.concat([df, videoInfoDf], ignore_index=True)

        df.to_csv(filename, index=False)

    # 读取数据
    data = pd.read_csv(filename, dtype={'earliest_day': str})
    
    
    fit_predict_cost_with_decision_tree(data,color_ranges,testWeeks=4)



if __name__ == "__main__":
    color_ranges = get_hsl_color_ranges2()
    print(color_ranges)
    # color_check(color_ranges)

    train(color_ranges)
