#!/usr/bin/env python
# coding=utf-8

import os
import sys
os.environ['R_HOME'] = r'C:\Users\jjl\.conda\envs\daima\Lib\R'
os.environ['R_USER'] = r'C:\Users\jjl\.conda\envs\daima\Lib\site-packages\rpy2'
import numpy as np
import pandas as pd
import xarray as xr

from romeomemo.utilities import conversion, grids, plotting, projection
from romeomemo.utilities import extract_fields, readconfig, readmemoascii

# 这段代码定义了一个名为main的函数，该函数接受一个配置文件路径cfg_path作为参数，并返回一个结果。
#
# 函数内部执行以下操作：
#
# 1、使用readconfig.load_cfg函数从指定路径读取配置文件，将其存储在变量cfg中。
# 2、从配置文件中获取memo_file、meteo_file和flight_code等信息，并存储在相应的变量中。
# 3、从配置文件中获取水平和垂直网格步长等信息，存储在变量dx和dz中。
# 4、获取摄影测量开始和结束时间等信息。
# 5、获取REBS模型参数，包括NoXP和b。
# 6、获取源的位置信息。
# 7、获取气象配置信息。
# 8、定义目标路径。
# 9、初始化krige_fname和fkrige_results变量。
# 10、从克里金结果的netCDF文件中读取数据。
# 11、计算网格步长，最小值以及提取预测结果和方差。
# 12、从memo文件中提取质量平衡数据。
# 13、将GPS高度与仪器入口高度对齐。
# 14、计算源和幕之间的垂直距离。
# 15、使用DroneSampling函数提取预测结果和方差。
# 16、将提取的预测结果和方差插值到memo数据帧上，存储在interp_df中。
# 17、准备要用于绘图的数据，包括时间序列和散点图。
# 18、返回结果。
def main(cfg_path):
    # 从指定路径读取配置文件
    cfg = readconfig.load_cfg(cfg_path)

    # 从配置文件中获取 memo 文件、气象文件和飞行编号等信息
    memo_file = cfg.memo_file
    meteo_file = cfg.meteo_file
    flight_code = cfg.flight_code

    # horizontal and vertical grid steps
    # 从配置文件中获取水平和垂直网格步长等信息
    dx = cfg.dx
    dz = cfg.dz

    ## start and end time of mass balance
    ## 摄影测量开始和结束时间
    start = cfg.start
    end = cfg.end

    ## REBS parameters
    ## REBS 模型参数
    NoXP = cfg.rebs_NoXP
    b = cfg.rebs_b

    ## LOCATION OF THE SOURCE
    ## 源的位置
    source_lon = cfg.source_lon
    source_lat = cfg.source_lat

    ## Meteorology configuration
    ## 气象配置
    meteo_file = cfg.meteo_file

    # Define target path
    # 定义目标路径
    target_path = cfg.model_path

    ## INITIALIZE FILES
    ## 初始化文件
    krige_fname = "_".join(["OK", flight_code])
    fkrige_results = os.path.join(target_path, krige_fname + ".nc")
    cfg.fkrige_results = fkrige_results

    ## Load netCDF file from kriging results
    ## 从克里金结果的 netCDF 文件中读取数据
    krige_results =  xr.open_dataset(fkrige_results)
    nx = krige_results.dims["x"]
    nz = krige_results.dims["z"]

    # 计算网格步长
    dx = np.around(krige_results.x[1] - krige_results.x[0], 2)
    dz = np.around(krige_results.z[1] - krige_results.z[0], 2)

    # 计算网格最小值
    xmin = np.asarray(krige_results.x)[0] - dx/2
    zmin = np.asarray(krige_results.z)[0] - dz/2

    # 提取预测结果和方差
    obs_pred = np.asarray(krige_results.ppm_mean)
    obs_var = np.asarray(krige_results.ppm_variance)

    # 从 memo 文件中提取质量平衡数据
    memo = readmemoascii.memoCDF(cfg.memo_file, cfg.start, cfg.end)
    memo_df = memo.massbalance_data(NoXP, b)
    con_ab = memo_df["con_ab"]
    ## Align altitude of GPS with inlet of instrument
    # Distance between the GPS and the altitude is 72 cm
    # 调整 GPS 高度与仪器入口的高度对齐
    agl = memo_df["Altitude"]

    ## COMPUTE PERPENDICULAR DISTANCE BETWEEN SOURCE AND CURTAIN
    ## 计算源和幕之间的垂直距离
    dtm = memo_df.index
    lon = memo_df["Longitude"]
    lat = memo_df["Latitude"]
    dist = memo_df["Distance"]

    obs_interp = extract_fields.DroneSampling(xmin, zmin, dx, dz, nx, nz, obs_pred)
    var_interp = extract_fields.DroneSampling(xmin, zmin, dx, dz, nx, nz, obs_var)

    obs_points = obs_interp.interpolate_points(dist, agl)
    var_points = var_interp.interpolate_points(dist, agl)

    interp_df = pd.DataFrame(index = memo_df.index)
    interp_df["obs_pred"] = obs_points
    interp_df["var_pred"] = var_points

    dtm = memo_df.index
    obs = con_ab
    obs_pred = interp_df["obs_pred"]
    var_pred = interp_df["var_pred"]

    tseries_fig = plotting.extract_con_plots(dtm, obs, obs_pred, var_pred)
    scatter_fig = plotting.plot_scatter(obs, obs_pred, var_pred)

    return
