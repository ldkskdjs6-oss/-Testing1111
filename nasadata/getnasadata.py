import requests
import pandas as pd
import io

def get_nasa_power_data_manual(lat, lon, start, end):
    base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    start_str = start.strftime('%Y%m%d')
    end_str = end.strftime('%Y%m%d')
   
    params = {
        'parameters': 'ALLSKY_SFC_SW_DWN,T2M,WS2M',  # 请求的参数：总太阳辐射，气温，风速
        'community': 'RE',  # 社区类型
        'longitude': lon,
        'latitude': lat,
        'start': start_str,
        'end': end_str,
        'format': 'CSV'
    }
   
    print(f" 正在请求 NASA API (范围: {start_str}-{end_str})...")
    response = requests.get(base_url, params=params)
   
    if response.status_code != 200:
        raise Exception(f"NASA API 请求失败: {response.text}")
   
    content = response.text
    header_marker = "-END HEADER-"
    if header_marker not in content:
        df = pd.read_csv(io.StringIO(content))
    else:
        lines = content.split('\n')
        header_line_index = 0
        for i, line in enumerate(lines):
            if header_marker in line:
                header_line_index = i + 1
                break
        data_str = '\n'.join(lines[header_line_index:])
        df = pd.read_csv(io.StringIO(data_str))
   
    df['index'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1) + ':00:00', format='%Y-%m-%d-%H:%M:%S')
    df = df.set_index('index')
    df.index = df.index.tz_localize('UTC')
    return df
import pandas as pd
from datetime import datetime

# 设置经纬度和日期范围
LATITUDE = 22.95
LONGITUDE = 115.65
start_date = datetime(2024, 12, 1)  # 开始日期
end_date = datetime(2025, 10, 30)  # 结束日期

# 调用获取数据的函数
nasa_data = get_nasa_power_data_manual(LATITUDE, LONGITUDE, start_date, end_date)

# 打印前几行数据查看
print(nasa_data.head())

# 保存数据为 CSV 文件
output_filename = 'nasa_power_data.csv'
nasa_data.to_csv(output_filename, index=True)
print(f"数据已保存为 {output_filename}")

