# ==================== 替换从这里开始 ====================
# 只需要改这部分！其他全部保持师兄原代码不动！

import pandas as pd
import pvlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# ==================== 【终极稳健版】只替换这部分 ====================
import pandas as pd
import numpy as np

file_path = "陆峰明阳光伏发电站数据3.csv"   # 你的文件名

# 第一步：直接读，不设任何假设
df = pd.read_csv(file_path, header=None, encoding='utf-8', dtype=str)

# 手动命名列
df.columns = ['date_str', 'energy_str']

# 第二步：超级容错地清洗电量列（解决 18109..6 这类问题）
def clean_energy_value(x):
    if pd.isna(x):
        return np.nan
    # 去掉空格、换行、逗号
    s = str(x).replace(' ', '').replace(',', '').strip()
    # 把多个小数点只保留第一个
    if s.count('.') > 1:
        parts = s.split('.')
        s = parts[0] + '.' + ''.join(parts[1:])
    # 如果变成空的或者全是点，就返回nan
    if s == '' or s == '.':
        return np.nan
    try:
        return float(s)
    except:
        return np.nan

df['Measured_Energy'] = df['energy_str'].apply(clean_energy_value)

# 第三步：解析中文日期（2025年10月1日）
def parse_chinese_date(s):
    try:
        s = str(s).strip()
        s = s.replace('年', '-').replace('月', '-').replace('日', '')
        return pd.to_datetime(s, format='%Y-%m-%d')
    except:
        return pd.NaT

df['Date'] = df['date_str'].apply(parse_chinese_date)

# 第四步：去掉任何解析失败的行
df = df.dropna(subset=['Date', 'Measured_Energy']).copy()
df = df[['Date', 'Measured_Energy']].sort_values('Date')

# 第五步：完全对齐你师兄原来的 data 格式（必须是 DataFrame + 列名 'Measured_Energy'）
data = df.set_index('Date').copy()
data = data[['Measured_Energy']]           # 确保只有这一列
data.index = data.index.normalize()        # 去掉时分秒

print(f"成功读取并清洗完成！共 {len(data)} 天有效数据")
print(f"时间范围：{data.index.min().date()} 至 {data.index.max().date()}")
print("已自动修复类似 '18109..6' 这类错误")
# ==================================================================

# —— 到这里结束！下面这行和师兄原来完全一样 ——
# data 就是师兄原来通过 Excel 得到的结果
# ==================== 替换到这里结束 ====================
latitude = 31.0          # 纬度
longitude = 121.0        # 经度
tz = 'Asia/Shanghai'
tilt = 25                # 倾角
azimuth = 180            # 方位角（180=朝南）
albedo = 0.2             # 地面反照率
module_efficiency = 0.18 # 组件效率
inverter_efficiency = 0.96
gamma_pmp = -0.004       # 功率温度系数
A_total = 10000        # 🌞 假设光伏板总面积（平方米）
times = pd.date_range(start=data.index.min(), end=data.index.max() + pd.Timedelta(days=1)-pd.Timedelta(hours=1),
                      freq='1H', tz=tz)
location = pvlib.location.Location(latitude, longitude, tz=tz, altitude=0, name='Lufeng')
solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
cs = location.get_clearsky(times, model='ineichen')
poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=tilt,
    surface_azimuth=azimuth,
    dni=cs['dni'],
    ghi=cs['ghi'],
    dhi=cs['dhi'],
    solar_zenith=solpos['zenith'],
    solar_azimuth=solpos['azimuth'],
    albedo=albedo
)
daytime = solpos['apparent_elevation'] > 0
poa = poa.where(daytime, 0)
temp_cell = pvlib.temperature.pvsyst_cell(
    poa['poa_global'], temp_air=25, wind_speed=1, u_c=29, u_v=0
)
P_dc = module_efficiency * poa['poa_global'] * (1 + gamma_pmp * (temp_cell - 25))
P_ac = inverter_efficiency * P_dc
sim_energy_m2 = P_ac.resample('D').sum() / 1000  # kWh/m²
sim_energy_total = sim_energy_m2 * A_total       # kWh（整站总发电量）
sim_energy_total.index = sim_energy_total.index.tz_localize(None)
sim_energy_total = sim_energy_total.reindex(data.index, method='nearest')

merged = pd.concat([data['Measured_Energy'], sim_energy_total], axis=1)
merged.columns = ['Measured_kWh', 'Simulated_kWh']
rmse = np.sqrt(mean_squared_error(merged['Measured_kWh'], merged['Simulated_kWh']))
mae = mean_absolute_error(merged['Measured_kWh'], merged['Simulated_kWh'])
r2 = r2_score(merged['Measured_kWh'], merged['Simulated_kWh'])
print("========== 模型拟合结果 ==========")
print(f"假设总面积 A = {A_total:,} m²")
print(f"RMSE = {rmse:.2f}")
print(f"MAE  = {mae:.2f}")
print(f"R²   = {r2:.3f}")
print("=================================")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 画图 + 万能显示（替换你原来的画图代码）===================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 6))
plt.plot(merged.index, merged['Measured_kWh'], label='实测发电量', color='blue', linewidth=2)
plt.plot(merged.index, merged['Simulated_kWh'], label=f'模型预测发电量 (A={A_total:,}m²)', 
         color='red', linestyle='--', linewidth=2)
plt.title("光伏发电模型 vs 实测发电量（陆丰明阳光伏站）", fontsize=16)
plt.xlabel("日期", fontsize=12)
plt.ylabel("日发电量 (kWh)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 万能保存 + 万能显示（关键！）
plt.savefig("拟合结果.png", dpi=300, bbox_inches='tight')
print("图片已保存：拟合结果.png")

# 下面这三行选一行就行（根据你的环境）
# 1. 如果你在 Jupyter / OpenBayes → 用这个：
from IPython.display import display, Image
display(Image("拟合结果.png"))

