import os
import sys
import pandas as pd
import datetime
import cv2

sys.path.append('/src')
from src.maxCompute import execSql

def getDataFromMaxCompute(installDayStartStr, installDayEndStr, earliestDayStartStr, earliestDayEndStr):
    mediasourceList = [
        # 'Applovin',
        # 'Facebook',
        'Google',
        # 'Mintegral',
        # 'Moloco',
        # 'Snapchat',
        # 'Twitter',
        # 'tiktok',
    ]
    sql = f'''
select
    material_name,
    video_url,
    earliest_day,
    sum(cost_value_usd) as cost
from rg_bi.dws_material_overseas_data_public
where
    app = '502'
    and material_type = '视频'
    and install_day between {installDayStartStr} and {installDayEndStr}
    and earliest_day between {earliestDayStartStr} and {earliestDayEndStr}
    and mediasource in ({','.join(f"'{source}'" for source in mediasourceList)})
    and country in ('US')
group by
    material_name,
    video_url,
    earliest_day
order by
    cost desc
;
    '''
    print(sql)
    data = execSql(sql)
    return data


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
    print('当前周日：', sundayStr)
    print('上周一：', lastMondayStr)
    print('上周日：', lastSundayStr)

    weekDf = getDataFromMaxCompute(mondayStr, sundayStr, lastMondayStr, lastSundayStr)
    
    csvDir = '/src/data/csv/raw'
    if not os.path.exists(csvDir):
        os.makedirs(csvDir)

    weekDf.to_csv(f'{csvDir}/{mondayStr}.csv', index=False)

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
    historyData()

    # main()
    


