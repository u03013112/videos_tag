# 简单颜色标签1
# TODO:
# 1、将训练集与测试集分别进行 查准率、查全率、R2 计算
# 2、调整阈值，这里限制了决策树的深度，所以分类必然是有限的，可以遍历几个分类的阈值，分别计算查准率、查全率、R2
# 3、模型保存 与 预测，将这个部分 可以独立出来，每周进行一下评测。


import os
import cv2
import pandas as pd
import numpy as np
import datetime
import joblib

# 定义颜色范围
color_ranges = {
    'c1': [(0, 50, 50), (15, 255, 255)],
    'c2':[(170, 50, 50), (180, 255, 255)],
    'c3': [(16, 50, 50), (25, 255, 255)],
    'c4': [(26, 50, 50), (35, 255, 255)],
    'c5': [(36, 50, 50), (95, 255, 255)],    # 扩展至青绿色
    'c6': [(96, 50, 50), (135, 255, 255)],   # 覆盖蓝青色过渡区
    'c7': [(136, 50, 50), (169, 255, 255)],  # 包含蓝紫色到品红色
    'c8': [(0, 0, 0), (180, 255, 50)],
    'c9': [(0, 0, 200), (180, 30, 255)],
    'c10': [(0, 0, 51), (180, 30, 199)]       # 新增灰度过渡带
}

# 为视频帧找到颜色标签
def findColorTagfromFrame(frame_dir,filename):
    # 遍历文件夹中的所有图片
    frames = []
    sorted_filenames = sorted(os.listdir(frame_dir))
    for frame_filename in sorted_filenames:
        if frame_filename.endswith('.jpg'):
            # print('frame_filename:', frame_filename)
            frame_path = os.path.join(frame_dir, frame_filename)
            frame = cv2.imread(frame_path)
            frames.append(frame)

    color_tags = list(color_ranges.keys())
    color_ratios = {color: 0 for color in color_tags}

    for frame in frames:
        # 将图像转换为HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 计算每种颜色的像素占比
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
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

def addTag(videoInfoDf):
    colorTagDf = pd.DataFrame()
    for i in range(len(videoInfoDf)):
        filename = videoInfoDf.loc[i, 'filename']
        frames_dir = videoInfoDf.loc[i, 'frames_dir']

        colorTagDf0 = findColorTagfromFrame(frames_dir, filename)
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
    plt.savefig("/src/data/decision_tree.png")  # 保存为 decision_tree.png

from sklearn.metrics import precision_score, recall_score, r2_score
from sklearn.tree import DecisionTreeClassifier
def fit_predict_cost_with_decision_tree(data,testWeeks=4):
    # 提取颜色比例特征
    color_features = data.columns[6:-1]  # 颜色比例特征列名
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


def train():

    csvDir = '/src/data/csv/color1'
    if not os.path.exists(csvDir):
        os.makedirs(csvDir)

    filename = f'{csvDir}/videoWithColor1Tag.csv'
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
                videoInfoDf = addTag(videoInfoDf)
                
                df = pd.concat([df, videoInfoDf], ignore_index=True)

        df.to_csv(f'{csvDir}/videoWithColor1Tag.csv', index=False)

    # 读取数据
    data = pd.read_csv(f'{csvDir}/videoWithColor1Tag.csv', dtype={'earliest_day': str})
    
    
    fit_predict_cost_with_decision_tree(data)
    
def main(dayStr = None):
    if dayStr is None:
        today = datetime.datetime.now()
    else:
        today = datetime.datetime.strptime(dayStr, '%Y%m%d')

    # 如果不是周一，什么都不做
    if today.weekday() != 0:
        print("今天不是周一，不执行数据准备。")
        return
    
    print(dayStr)
    monday = today - datetime.timedelta(days=7)
    mondayStr = monday.strftime('%Y%m%d')

    downloadCsvDir = '/src/data/csv/download'
    filename = f'{downloadCsvDir}/{mondayStr}.csv'
    if os.path.exists(filename):
        videoInfoDf = pd.read_csv(filename)
    else:
        print(f"Error: {filename} does not exist.")
        return
    videoInfoDf = addTag(videoInfoDf)

    # 读取模型
    model_dir = '/src/data/models/'
    model_path = os.path.join(model_dir, 'color1_decision_tree.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        print(f"Error: {model_path} does not exist.")
        return

    # 预测
    color_features = videoInfoDf.columns[6:-1]  # 颜色比例特征列名
    X = videoInfoDf[color_features]  # 输入特征
    # 获取预测概率
    probabilities = model.predict_proba(X)
    
    videoInfoDf.loc[:, 'probabilities'] = probabilities[:, 1]
    videoInfoDf.loc[:, 'predicted_class'] = (probabilities[:, 1] >= 0.2).astype(int)
    videoInfoDf.loc[:, 'y'] = 0
    videoInfoDf.loc[videoInfoDf['cost'] > 3500, 'y'] = 1

    # 将数据进行保存备份
    csvDir = '/src/data/csv/color1'
    if not os.path.exists(csvDir):
        os.makedirs(csvDir)
    videoInfoDf.to_csv(f'{csvDir}/{mondayStr}.csv', index=False)
    print(f"结论文件保存在 /src/data/csv/color1/{mondayStr}.csv")

    # 计算查准率和查全率
    precision = precision_score(videoInfoDf['y'], videoInfoDf['predicted_class'])
    recall = recall_score(videoInfoDf['y'], videoInfoDf['predicted_class'])
    # 计算 R2
    r2 = r2_score(videoInfoDf['y'], videoInfoDf['predicted_class'])
    print('查准率:', precision)
    print('查全率:', recall)
    print('R2:', r2)

# 历史数据补充，如果有需要补充的历史数据，调佣这个函数，并且调整时间范围
def historyData():
    startDayStr = '20250101'
    endDayStr = '20250430'

    startDay = datetime.datetime.strptime(startDayStr, '%Y%m%d')
    endDay = datetime.datetime.strptime(endDayStr, '%Y%m%d')

    for i in range((endDay - startDay).days + 1):
        day = startDay + datetime.timedelta(days=i)
        dayStr = day.strftime('%Y%m%d')
        # print(dayStr)
        main(dayStr)

if __name__ == '__main__':
    train()
    # historyData()
    # main('20250428')
