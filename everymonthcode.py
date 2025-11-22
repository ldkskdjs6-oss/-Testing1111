import pandas as pd
import pvlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import requests
import io
# ================= 1. 基础配置区域 =================
# 你的 NASA API KEY
NASA_API_KEY = '2uf1G3FW8KI1dfvUy6AFcMf5hNqJwLA7Ztm9XkjJ'
# 陆丰明阳电站坐标
LATITUDE = 22.95
LONGITUDE = 115.65
# 电站参数
TILT = 25
AZIMUTH = 180
ALBEDO = 0.2
MODULE_EFF = 0.18
INVERTER_EFF = 0.96
A_TOTAL = 10000
FILE_PATH = "陆丰明阳光伏发电站数据3.csv"
# ================= 2. 智能字体设置 =================
def set_cloud_font():
    """解决云平台中文乱码问题"""
    font_list = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'PingFang SC', 'Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False
    found_font = None
    import matplotlib.font_manager
    system_fonts = set([f.name for f in matplotlib.font_manager.fontManager.ttflist])
    for font in font_list:
        if font in system_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            found_font = font
            break
    if found_font:
        print(f"提示：已启用中文字体 '{found_font}'")
        return True
    else:
        print("提示：未检测到常用中文字体，切换为英文显示。")
        return False
has_chinese_font = set_cloud_font()
# ================= 3. 手动获取 NASA 数据 (修复版) =================
def get_nasa_power_data_manual(lat, lon, start, end):
    base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    start_str = start.strftime('%Y%m%d')
    end_str = end.strftime('%Y%m%d')
   
    params = {
        'parameters': 'ALLSKY_SFC_SW_DWN,T2M,WS2M',
        'community': 'RE',
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
# ================= 4. 主程序逻辑 =================
print(">>> 步骤1：正在读取并清洗实测数据...")
if not os.path.exists(FILE_PATH):
    print(f"错误：找不到文件 '{FILE_PATH}'。请检查上传是否成功。")
    exit()
df = pd.read_csv(FILE_PATH, header=None, encoding='utf-8', dtype=str)
df.columns = ['date_str', 'energy_str']
def clean_energy(x):
    if pd.isna(x): return np.nan
    s = str(x).replace(' ', '').replace(',', '').strip()
    if s.count('.') > 1:
        parts = s.split('.')
        s = parts[0] + '.' + ''.join(parts[1:])
    try: return float(s)
    except: return np.nan
def parse_date(s):
    try:
        s = str(s).strip().replace('年', '-').replace('月', '-').replace('日', '')
        return pd.to_datetime(s, format='%Y-%m-%d')
    except: return pd.NaT
df['Measured_Energy'] = df['energy_str'].apply(clean_energy)
df['Date'] = df['date_str'].apply(parse_date)
df = df.dropna(subset=['Date', 'Measured_Energy'])
data_measured = df.set_index('Date').sort_index()
start_date = data_measured.index.min()
end_date = data_measured.index.max()
print(f" 数据范围：{start_date.date()} 至 {end_date.date()}，共 {len(data_measured)} 天")
# --- NASA ---
print(f"\n>>> 步骤2：正在下载 NASA 气象数据...")
try:
    nasa_data = get_nasa_power_data_manual(
        lat=LATITUDE,
        lon=LONGITUDE,
        start=start_date,
        end=end_date + pd.Timedelta(days=1)
    )
    print(" NASA 数据下载成功！")
    nasa_data = nasa_data.replace(-999, 0)
except Exception as e:
    print(f"!!! NASA 下载失败: {e}")
    raise e
# --- 模拟 ---
print("\n>>> 步骤3：正在进行光伏模拟...")
ghi = nasa_data['ALLSKY_SFC_SW_DWN']
temp_air = nasa_data['T2M']
wind_speed = nasa_data['WS2M']
location = pvlib.location.Location(LATITUDE, LONGITUDE, tz='UTC')
times = nasa_data.index
solpos = location.get_solarposition(times)
erbs = pvlib.irradiance.erbs(ghi, solpos['zenith'], times)
dni = erbs['dni']
dhi = erbs['dhi']
poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=TILT,
    surface_azimuth=AZIMUTH,
    dni=dni,
    ghi=ghi,
    dhi=dhi,
    solar_zenith=solpos['zenith'],
    solar_azimuth=solpos['azimuth'],
    albedo=ALBEDO
)
temp_cell = pvlib.temperature.pvsyst_cell(
    poa['poa_global'], temp_air=temp_air, wind_speed=wind_speed
)
gamma_pmp = -0.004
P_dc_unit = MODULE_EFF * poa['poa_global'] * (1 + gamma_pmp * (temp_cell - 25))
P_ac_unit = INVERTER_EFF * P_dc_unit
# 1. 转到北京时间
P_ac_local = P_ac_unit.tz_convert('Asia/Shanghai')
# 2. 按天求和
daily_energy_raw = P_ac_local.resample('D').sum()
# 3. 【核心修复】移除时区信息，变成"墙上时间"，以便和 CSV 对齐
daily_energy_raw.index = daily_energy_raw.index.tz_localize(None)
# 4. 计算总电站发电量
simulated_energy = (daily_energy_raw / 1000) * A_TOTAL
# 5. 对齐索引
simulated_energy = simulated_energy.reindex(data_measured.index, method='nearest')
# --- 画图 --- 
print("\n>>> 步骤4：计算误差并画图...") 
merged = pd.concat([data_measured['Measured_Energy'], simulated_energy], axis=1).dropna() 
merged.columns = ['Measured', 'Simulated'] 

# 按月份分组
monthly_groups = merged.resample('M')

# 对每个月的数据分别绘制并保存
for month, month_data in monthly_groups:
    if len(month_data) > 0:
        rmse = np.sqrt(mean_squared_error(month_data['Measured'], month_data['Simulated']))
        r2 = r2_score(month_data['Measured'], month_data['Simulated'])
        print(f"="*30)
        print(f" 月份: {month.strftime('%Y-%m')} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
        print(f"="*30)

        plt.figure(figsize=(14, 7))

        if has_chinese_font:
            label_meas = '实测发电量 (CSV)'
            label_sim = 'NASA气象模拟值'
            title_text = f"光伏拟合对比：实测 vs NASA卫星仿真\n(地点：陆丰, R²={r2:.3f}, 月份：{month.strftime('%Y-%m')})"
            xlabel_text = "日期"
            ylabel_text = "日发电量 (kWh)"
        else:
            label_meas = 'Measured Energy (CSV)'
            label_sim = 'NASA Simulated (Satellite)'
            title_text = f"PV Fitting: Measured vs NASA Simulation\n(Location: Lufeng, R²={r2:.3f}, Month: {month.strftime('%Y-%m')})"
            xlabel_text = "Date"
            ylabel_text = "Daily Energy (kWh)"
        
        plt.plot(month_data.index, month_data['Measured'], label=label_meas, color='blue', alpha=0.7)
        plt.plot(month_data.index, month_data['Simulated'], label=label_sim, color='red', linestyle='--', alpha=0.8)
        plt.title(title_text, fontsize=16)
        plt.xlabel(xlabel_text)
        plt.ylabel(ylabel_text)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存图片，文件名包含月份
        save_name = f"Lufeng_PV_Fitting_Result_{month.strftime('%Y-%m')}.png"
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，释放内存
        print(f"\n[成功] 图片已保存为: {save_name}")
