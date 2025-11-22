import pandas as pd
import pvlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
file_path = "陆丰明阳光伏站发电电量.xlsx"
data = pd.read_excel(file_path, skiprows=1)
data.columns = ['Date', 'Measured_Energy']
data = data.dropna(subset=['Date', 'Measured_Energy'])
def convert_date(x):
    try:
        return pd.Timestamp('1899-12-30') + pd.to_timedelta(int(float(x)), unit='D')
    except:
        return pd.to_datetime(x, errors='coerce')
data['Date'] = data['Date'].apply(convert_date)
data = data.dropna(subset=['Date'])
data = data.set_index('Date')
data.index = data.index.normalize()  
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

plt.figure(figsize=(10,5))
plt.plot(merged.index, merged['Measured_kWh'], label='实测发电量', color='blue', linewidth=2)
plt.plot(merged.index, merged['Simulated_kWh'], label=f'模型预测发电量 (A={A_total:,}m²)', color='red', linestyle='--')
plt.title("光伏发电模型 vs 实测发电量（陆丰明阳光伏站）")
plt.xlabel("日期")
plt.ylabel("日发电量 (kWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
