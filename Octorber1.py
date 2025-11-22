# 最终提交版：完全符合文档要求的真实气象模型
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
import requests
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize_scalar

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取CSV（稳健处理所有格式问题）
df = pd.read_csv("陆峰明阳光伏发电站数据2.csv", header=None, encoding='utf-8', dtype=str)
df.columns = ['date_str', 'energy_str']
df['energy_str'] = df['energy_str'].str.replace(' ', '').str.replace(',', '')
df['Measured_Energy'] = pd.to_numeric(df['energy_str'], errors='coerce')
df['Date'] = pd.to_datetime(df['date_str'].str.replace('年','-').str.replace('月','-').str.replace('日',''), 
                            format='%Y-%m-%d', errors='coerce')
df = df.dropna(subset=['Date', 'Measured_Energy']).set_index('Date')
data = df[['Measured_Energy']].sort_index()
data.index = data.index.normalize().tz_localize('Asia/Shanghai')

# 2. 参数（与文档完全一致）
latitude, longitude = 22.908, 115.920
tilt, azimuth, albedo = 25, 180, 0.2
module_efficiency = 0.18
inverter_efficiency = 0.96
gamma_pmp = -0.004
api_key = "2uf1G3FW8KI1dfvUy6AFcMf5hNqJwLA7Ztm9XkjJ"

# 3. 拉取真实气象（NASA POWER）
times = pd.date_range(start=data.index.min(), 
                      end=data.index.max() + pd.Timedelta(days=1) - pd.Timedelta(hours=1),
                      freq='H', tz='Asia/Shanghai')

url = (f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
       f"parameters=ALLSKY_SFC_SW_DWN,T2M,WS10M&community=RE"
       f"&latitude={latitude}&longitude={longitude}"
       f"&start={times[0]:%Y%m%d}&end={times[-1]:%Y%m%d}"
       f"&format=JSON&api_key={api_key}")

resp = requests.get(url).json()['properties']['parameter']
weather = pd.DataFrame({
    'ghi': pd.Series(resp['ALLSKY_SFC_SW_DWN'], dtype=float),
    'temp_air': pd.Series(resp['T2M'], dtype=float),
    'wind_speed': pd.Series(resp['WS10M'], dtype=float)
}, index=times)
# ========== 关键修复：清理 NASA POWER 的 -999 缺测值 ==========
# 把 -999 替换为 NaN，再向前/向后填充（气象数据一天内的缺测用邻近值填合理）
weather = weather.replace(-999.0, np.nan)     # -999 → NaN
weather = weather.replace(-9999.0, np.nan)    # 有时也出现 -9999

# 按列填充（GHI、温度、风速分别处理）
weather['ghi']        = weather['ghi'].fillna(method='ffill').fillna(method='bfill')
weather['temp_air']   = weather['temp_air'].fillna(method='ffill').fillna(method='bfill')
weather['wind_speed'] = weather['wind_speed'].fillna(method='ffill').fillna(method='bfill')

# 可选：如果还有少量残留 NaN，用列均值填（稳健）
weather = weather.fillna(weather.mean())

print(f"气象数据清洗完成，GHI 范围：{weather['ghi'].min():.1f} ~ {weather['ghi'].max():.1f} W/m²")
# ================================================================
weather['dni'] = weather['ghi'] * 0.82
weather['dhi'] = weather['ghi'] - weather['dni']

# 4. pvlib 建模（与文档流程完全一致）
loc = pvlib.location.Location(latitude, longitude, tz='Asia/Shanghai')
solpos = loc.get_solarposition(times)
poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=tilt, surface_azimuth=azimuth,
    solar_zenith=solpos['apparent_zenith'], solar_azimuth=solpos['azimuth'],
    dni=weather['dni'], ghi=weather['ghi'], dhi=weather['dhi'], albedo=albedo)

daytime = solpos['apparent_elevation'] > 0
poa = poa.where(daytime, 0)

temp_cell = pvlib.temperature.pvsyst_cell(poa['poa_global'], weather['temp_air'], weather['wind_speed'])
p_dc = poa['poa_global'] * module_efficiency * (1 + gamma_pmp * (temp_cell - 25))
p_ac = p_dc * inverter_efficiency

# 5. 自动拟合面积
def error(A):
    sim = p_ac.resample('D').sum() / 1000 * A
    merged = pd.concat([data['Measured_Energy'], sim.reindex(data.index)], axis=1).dropna()
    return mean_squared_error(merged.iloc[:,0], merged.iloc[:,1])

A_opt = minimize_scalar(error, bounds=(8000, 20000)).x

# 6. 最终模拟结果
sim_daily = p_ac.resample('D').sum() / 1000 * A_opt
sim_daily = sim_daily.reindex(data.index)

merged = pd.concat([data['Measured_Energy'], sim_daily], axis=1).dropna()
merged.columns = ['Measured_kWh', 'Simulated_kWh']
r2 = r2_score(merged['Measured_kWh'], merged['Simulated_kWh'])

# 7. 打印与出图（风格与文档一致）
print(f"最优拟合面积 A = {A_opt:,.0f} m²")
print(f"R² = {r2:.4f}")

plt.figure(figsize=(12,6))
plt.plot(merged.index, merged['Measured_kWh'], 'o-', label='实测发电量', linewidth=2)
plt.plot(merged.index, merged['Simulated_kWh'], '--', label=f'模型预测发电量 (A={A_opt:,.0f}m²)', linewidth=2)
plt.title('陆丰明阳光伏电站发电量实测与真实气象模型预测对比')
plt.xlabel('日期')
plt.ylabel('日发电量 (kWh)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("陆丰明阳_真实气象模型_最终版.png", dpi=300, bbox_inches='tight')
plt.show()