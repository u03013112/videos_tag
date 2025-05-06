import os
import sys
import pandas as pd
import datetime
import cv2


# 下载视频并制作帧
def downloadVideos(df, filenamePrefix = ''):
    trainDf = pd.DataFrame()

    for i in range(len(df)):
        index = i + 1
        filename = f'{filenamePrefix}_{index}.mp4'
        name = df.iloc[i]['material_name']
        cost = df.iloc[i]['cost']
        video_url = df.iloc[i]['video_url']
        earliest_day = df.iloc[i]['earliest_day']

        # 下载视频到本地 videos目录下
        video_dir = '/src/data/videos'
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        video_path = os.path.join(video_dir, filename)
        video_frames_dir = os.path.join(video_dir, f'{filenamePrefix}_{index}_frames')
        # if not os.path.exists(video_path):
        # 改为如果帧目录不存在，则下载视频，重新制作帧
        if not os.path.exists(video_frames_dir):
            print(f"Downloading video {index}/{len(df)}: {filename}")
            print(f"video_url: {video_url}")
            # 下载视频
            os.system(f'curl -o {video_path} {video_url}')
            os.makedirs(video_frames_dir)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = int(fps)
            # if fps < 10:
            #     fps = 30
            print('fps:',fps)

            frame_count = 0
            frame_name_count = 1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % fps == 0:
                    frame_filename = os.path.join(video_frames_dir, f'frame_{frame_name_count}.jpg')
                    frame_name_count += 1
                    cv2.imwrite(frame_filename, frame,[cv2.IMWRITE_JPEG_QUALITY, 60])
                    # 最多保存30帧
                    if frame_name_count > 30:
                        break

                frame_count += 1
            cap.release()
            # 删除视频文件
            os.remove(video_path)

        # 创建一个新的 DataFrame行
        new_row = pd.DataFrame([{
            'filename': filename,
            'frames_dir': video_frames_dir,
            'video_url': video_url,
            'name': name,
            'cost': cost,
            'earliest_day': earliest_day
        }])

        # 使用 pd.concat() 将新行添加到现有 DataFrame
        trainDf = pd.concat([trainDf, new_row], ignore_index=True)

    return trainDf

def main(dayStr = None):

    if dayStr is None:
        today = datetime.datetime.now()
    else:
        today = datetime.datetime.strptime(dayStr, '%Y%m%d')

    # 如果不是周一，什么都不做
    if today.weekday() != 0:
        print("今天不是周一，不执行数据准备。")
        return
    
    monday = today - datetime.timedelta(days=7)
    mondayStr = monday.strftime('%Y%m%d')

    csvfile = f'/src/data/csv/raw/{mondayStr}.csv'
    if os.path.exists(csvfile):
        df = pd.read_csv(csvfile)
    else:
        print(f"Error: {csvfile} does not exist.")
        return
    
    downloadDf = downloadVideos(df, mondayStr)

    downloadCsvDir = '/src/data/csv/download'
    if not os.path.exists(downloadCsvDir):
        os.makedirs(downloadCsvDir)

    downloadDf.to_csv(f'{downloadCsvDir}/{mondayStr}.csv', index=False)
    print(f"视频下载完成，文件保存在 /src/data/csv/download/{mondayStr}.csv")

# 历史数据补充，如果有需要补充的历史数据，调佣这个函数，并且调整时间范围
def historyData():
    startDayStr = '20250101'
    endDayStr = '20250429'

    startDay = datetime.datetime.strptime(startDayStr, '%Y%m%d')
    endDay = datetime.datetime.strptime(endDayStr, '%Y%m%d')

    for i in range((endDay - startDay).days + 1):
        day = startDay + datetime.timedelta(days=i)
        dayStr = day.strftime('%Y%m%d')
        print(dayStr)
        main(dayStr)

if __name__ == '__main__':
    # historyData()

    main('20250106')
    


