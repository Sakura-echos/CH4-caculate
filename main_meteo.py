#!/usr/bin/env python
# coding=utf-8

import os
os.environ['R_HOME'] = r'C:\Users\jjl\.conda\envs\daima\Lib\R'
os.environ['R_USER'] = r'C:\Users\jjl\.conda\envs\daima\Lib\site-packages\rpy2'
import sys
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from romeomemo.massbalance import mixturemodel as mm
from romeomemo.meteorology import anemo, wcomponent
from romeomemo.utilities import conversion, grids, plotting, projection
from romeomemo.utilities import readconfig, readmemoascii

from romeomemo.functions import estimate

import gstools as gs
from pykrige.ok import OrdinaryKriging

# 这是一个 Python 代码，用于污染传输模型的质量平衡记录处理和网格计算，其中包括以下步骤：
#
# 读取配置文件；
# 从配置文件中提取必要的参数，如质量平衡记录文件路径、气象数据文件路径、飞行代码等；
# 初始化网格计算所需的一些参数，如水平和垂直网格步长、起始和结束时间、对数风速剖面、污染源的经纬度坐标等；
# 根据提供的经纬度坐标计算距离；
# 处理 GPS 测量高度与仪器进气口对齐；
# 计算污染源和监测帷幕间的垂直距离；
# 创建目标网格；
# 读取气象数据，并进行相关的预处理；
# 计算风速，并根据风速的方向计算其在垂直方向上的分量；
# 计算网格内的平均风速，并生成风场文件。
# 总体而言，这段代码是一个质量平衡记录处理和网格计算的流程，它可以用于计算污染传输过程中的浓度分布情况。
def main(cfg_path):

    cfg = readconfig.load_cfg(cfg_path)  # 读取配置文件

    memo_file = cfg.memo_file  # 提取质量平衡记录文件路径
    meteo_file = cfg.meteo_file  # 提取气象数据文件路径
    flight_code = cfg.flight_code  # 提取飞行代码

    # 水平和垂直网格步长
    dx = cfg.dx
    dz = cfg.dz

    ## 质量平衡的起始和结束时间
    start = cfg.start
    end = cfg.end

    ## REBS 参数
    NoXP = cfg.rebs_NoXP
    b = cfg.rebs_b

    ## 对数风速剖面
    wind_z = cfg.wind_z
    wind_OL = cfg.wind_OL
    wind_u_star = cfg.wind_u_star

    ## 污染源的经纬度坐标
    source_lon = cfg.source_lon
    source_lat = cfg.source_lat

    ## 气象配置
    meteo_file = cfg.meteo_file

    # 定义目标路径
    target_path = cfg.model_path

    ## 初始化文件
    wind_fname = "_".join(["MET", flight_code])  # 生成风场文件名
    fwind_results = os.path.join(target_path, wind_fname + ".nc")  # 生成风场文件路径
    cfg.fwind_results = fwind_results

    memo = readmemoascii.memoCDF(memo_file, start, end)  # 读取质量平衡记录文件
    memo_df = memo.massbalance_data(NoXP, b)  # 从质量平衡记录文件中提取数据

    con_ab = memo_df["con_ab"]  # 每个质量平衡记录的污染物浓度
    ## 将 GPS 测量高度与仪器进气口对齐
    # GPS 测量高度与仪器进气口的距离为 72 cm
    agl = memo_df["Altitude"]

    ## 计算污染源和监测帷幕间的垂直距离
    lon = memo_df["Longitude"]
    lat = memo_df["Latitude"]
    dist = memo_df["Distance"]

    proc_gps = grids.ProcessCoordinates(lon, lat)  # 对 GPS 测量的经纬度坐标进行处理
    m, b, r_value = proc_gps.regression_coeffs()  # 计算 GPS 测量的经纬度坐标与距离之间的线性关系

    perp_distance = grids.perpendicular_distance(lon, lat, source_lon, source_lat)  # 计算污染源和监测帷幕间的垂直距离

    xmin = 0.0  # x 轴最小值
    zmin = 0.0  # z 轴最小值
    nx = int(np.floor(np.max(dist) / dx) + 1)  # x 轴上的网格数


    ## CREATE TARGET GRID
    target_grid = grids.Grid(nx, nz, dx, dz, xmin, zmin)
    range_x = target_grid.xdist_range()
    range_z = target_grid.alt_range()

    krige_ppm = pd.DataFrame(index=dist.index)
    krige_ppm["dist"] = dist
    krige_ppm["agl"] = agl
    krige_ppm["obs_data"] = con_ab

    max_dist = np.max(krige_ppm["dist"])
    agl_series = pd.Series(krige_ppm["agl"]).diff()
    agl_max = agl_series.max()

    ## Projected Wind profile
    meteo_df = anemo.meteorology_data(meteo_file, start, end)
    sw_corr_factor = [wcomponent.swise_correction(wd, m) for wd in meteo_df["wd"]]
    meteo_df["sw_factor"] = sw_corr_factor

    meteo_df["stream_wind"] = meteo_df["ws"] * meteo_df["sw_factor"]
    meteo_df["dist"] = krige_ppm["dist"]
    meteo_df["agl"] = krige_ppm["agl"]
    meteo_df = meteo_df.dropna()

    ## Domininant perpendicular wind
    mean_perp_wind = meteo_df["stream_wind"].mean()
    std_perp_wind = meteo_df["stream_wind"].std()
    ste_perp_wind = std_perp_wind / np.sqrt(len(meteo_df))

    dom_wind_mean = np.full((nz,nx), mean_perp_wind)
    dom_wind_ss = np.full((nz,nx), std_perp_wind**2)
    dom_wind_se = np.full((nz,nx), ste_perp_wind)
    print("Average mean wind speed: {:.4f}m/s".format(mean_perp_wind))
    print("Standard deviation: {:.4f}m/s".format(std_perp_wind))
    print("Standard error: {:.4f}m/s".format(ste_perp_wind))

    ## Logarithmic profile wind
    stab_param = wind_z / wind_OL

    if np.isclose(stab_param, 0, atol=0.5):
        print("Neutral")
        log_wind = wcomponent.neutral_prof(mean_perp_wind, wind_z, wind_u_star,
                                           range_z)
        std_log = wcomponent.neutral_prof(std_perp_wind, wind_z, wind_u_star,
                                          range_z)
    elif stab_param > 0.5:
        print("Stable")
        log_wind = wcomponent.stable_prof(mean_perp_wind, wind_z,
                                          wind_u_star,wind_OL, range_z)
        std_log = wcomponent.stable_prof(std_perp_wind, wind_z, wind_u_star,
                                         wind_OL, range_z)
    elif stab_param < -0.5:
        print("Unstable")
        log_wind = wcomponent.unstable_prof(mean_perp_wind, wind_z,
                                            wind_u_star, wind_OL, range_z)
        std_log = wcomponent.unstable_prof(std_perp_wind, wind_z, wind_u_star,
                                           wind_OL, range_z)
    else:
        print("Check stability parameter value:{:.4f}".format(stab_param))

    min_wind = np.min(log_wind[log_wind >= 0])
    log_wind[log_wind < 0] = min_wind

    log_mean_curt = np.ones((nz, nx)) * log_wind[:, None]
    std_mean_curt = np.ones((nz, nx)) * std_log[:, None]


    print("================================")
    print("Writing meteorology files.\n\n")

    wind_results = xr.Dataset({
                "z" : xr.DataArray(range_z, dims=("z",)),
                "x" : xr.DataArray(range_x, dims=("x",)),
                "scalar_wind_mean" : xr.DataArray(dom_wind_mean, dims=("z", "x")),
                "scalar_wind_std" : xr.DataArray(dom_wind_ss, dims=("z", "x")),
                "scalar_wind_ste" : xr.DataArray(dom_wind_se, dims=("z", "x")),
                "log_wind_mean" : xr.DataArray(log_mean_curt, dims=("z", "x")),
                "log_wind_std" : xr.DataArray(std_mean_curt, dims=("z", "x")),
                })

    wind_results.to_netcdf(fwind_results)

    print("Meteorology curtain file written")

    return

if __name__ ==  "__main__":
    try:
        config_path = sys.argv[1]
    except IndexError:
        raise RuntimeError("Please supply a config file.")
    main(config_path)

