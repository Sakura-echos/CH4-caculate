#!/usr/bin/env python
# coding=utf-8

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr


import gstools as gs
from pykrige.ok import OrdinaryKriging

from romeomemo.massbalance import mixturemodel as mm
from romeomemo.meteorology import anemo, wcomponent
from romeomemo.utilities import conversion, grids, plotting, projection
from romeomemo.utilities import readconfig, readmemoascii

from romeomemo.functions import estimate


# 该代码是用于进行基于空间变异的数据插值（kriging）分析，
# 生成一个目标网格。其中，通过读取配置文件（.cfg文件）来设置一些输入参数。
# 参数设置包括源位置经纬度，起始和结束时间，网格步长等。
# 然后，代码读取质量平衡记录和气象记录，并处理这些记录以获取所需的空间和环境参数。
# 接着，对GPS数据进行回归分析以确定与控制点之间的垂直距离。
# 最后，使用标准克里金插值法进行数据插值，并生成nc文件保存插值结果。
def main(cfg_path):
    # 读入配置文件cfg_path
    cfg = readconfig.load_cfg(cfg_path)
    # memo_file是mass balance data的文件名，meteo_file是meteorology data的文件名，flight_code是用于命名输出结果的航班代码
    memo_file = cfg.memo_file
    meteo_file = cfg.meteo_file
    flight_code = cfg.flight_code
    # dx和dz是x和z方向上的格子大小
    dx = cfg.dx
    dz = cfg.dz
    # mass balance的起止时间
    start = cfg.start
    end = cfg.end
    # REBS参数
    NoXP = cfg.rebs_NoXP
    b = cfg.rebs_b
    # 源的经纬度
    source_lon = cfg.source_lon
    source_lat = cfg.source_lat
    # meteorology文件
    meteo_file = cfg.meteo_file
    # 输出文件的路径
    target_path = cfg.model_path
    # 初始化krige的结果文件
    krige_fname = "_".join(["OK", flight_code])
    fkrige_results = os.path.join(target_path, krige_fname + ".nc")
    cfg.fkrige_results = fkrige_results
    # 读入mass balance数据并生成mass balance dataframe
    memo = readmemoascii.memoCDF(memo_file, start, end)
    memo_df = memo.massbalance_data(NoXP, b)
    # 取出浓度数据
    con_ab = memo_df["con_ab"]
    # 对海拔进行校准，距离为72cm
    agl = memo_df["Altitude"]
    # 读入经纬度和距离
    lon = memo_df["Longitude"]
    lat = memo_df["Latitude"]
    dist = memo_df["Distance"]
    # 读入环境参数
    pres = memo_df["Pressure"]
    temp = memo_df["Temperature"]
    # 替换NAN值为STP值
    pres[pres == -999.99] = 1013.25
    temp[temp == -999.99] = 273.15
    # 转化为浓度值
    # con_ab = conversion.ppm2micro(ppm_ab, temp=temp, pres=pres, mass=16.04)
    # 初始化经纬度处理类
    proc_gps = grids.ProcessCoordinates(lon, lat)
    # 计算回归系数，返回拟合直线的斜率m和截距b以及拟合优度r_value
    m, b, r_value = proc_gps.regression_coeffs()
    # 计算从源到站点的垂直距离
    perp_distance = grids.perpendicular_distance(lon, lat, source_lon, source_lat)
    # 设置2D空间的边界和格子大小
    xmin = 0.0
    zmin = 0.0
    nx = int(np.floor(np.max(dist) / dx) + 1)
    nz = int(np.floor(np.max(agl) / dz) + 1)

    target_grid = grids.Grid(nx, nz, dx, dz, xmin, zmin)  # 创建网格对象target_grid，包括nx, nz, dx, dz, xmin, zmin参数
    range_x = target_grid.xdist_range()  # 获取x方向距离范围
    range_z = target_grid.alt_range()  # 获取z方向距离范围

    if not os.path.exists(target_path):  # 如果文件夹不存在
        os.makedirs(target_path)  # 则创建文件夹

    save_arrays = True  # 定义一个布尔型变量save_arrays为True
    if os.path.isfile(fkrige_results):  # 如果文件fkrige_results已存在
        print("Overwite interpolated arrays in %s? " % target_path)  # 输出字符串信息
        s = input("y/[n] \n")  # 提示输入
        save_arrays = (s == "y")  # 根据用户输入更新save_arrays的值

    if save_arrays:  # 如果save_arrays为True

        krige_ppm = pd.DataFrame(index=dist.index)  # 创建pandas的DataFrame对象krige_ppm，以dist.index为行索引
        krige_ppm["dist"] = dist  # 将dist的值赋给krige_ppm的“dist”列
        krige_ppm["agl"] = agl  # 将agl的值赋给krige_ppm的“agl”列
        krige_ppm["obs_data"] = con_ab  # 将con_ab的值赋给krige_ppm的“obs_data”列

        max_dist = np.max(krige_ppm["dist"])  # 计算krige_ppm的“dist”列的最大值并赋值给max_dist
        # max_agl = np.max(krige_ppm["dist"])
        # max_disp = np.hypot(max_dist, max_agl) / 2
        agl_series = pd.Series(krige_ppm["agl"]).diff()  # 创建一个以krige_ppm的“agl”列的差值为元素的pandas的Series对象agl_series
        agl_max = agl_series.max()  # 获取agl_series的最大值

        ## 测量标准克里金插值
        X_one = np.asarray(krige_ppm["dist"])  # 将krige_ppm的“dist”列转换为ndarray类型并赋值给X_one
        Y_one = np.asarray(krige_ppm["agl"])  # 将krige_ppm的“agl”列转换为ndarray类型并赋值给Y_one
        Z_one = np.asarray(krige_ppm["obs_data"])  # 将krige_ppm的“obs_data”列转换为ndarray类型并赋值给Z_one

        init_gp1_l1 = mm.init_lengthscale(krige_ppm["dist"], 1)  # 初始横向长度尺度为dist的方差除以1的平方根
        init_gp1_l2 = mm.init_lengthscale(krige_ppm["agl"], 1)  # 初始纵向长度尺度为agl的方差除以1的平方根
        init_gp1_sf = mm.init_variance(krige_ppm["obs_data"], 1)  # 初始方差为obs_data的方差除以1的平方根
        init_gp1_sn = 0.25 * init_gp1_sf  # 初始噪声标准差为方差的1/4

        nu = 1.5  # Matern内核的nu参数
        gp1_kernel = mm.matern_kernel(init_gp1_sf, [init_gp1_l1, init_gp1_l2], nu, agl_max, max_dist)  # 创建Matern内核
        gp1_sf, gp1_l1, gp1_l2 = mm.EM_hyperparam(X_one, Y_one, Z_one, gp1_kernel, init_gp1_sn)  # 用EM算法确定内核超参数
        gp1_sn = np.mean(gp1_sf)  # 计算噪声标准差的平均值

        print("Horizontal length scale: {:.4f}".format(gp1_l1))
        print("Vertical length scale: {:.4f}".format(gp1_l2))

        gp1_l1 = 2.9610
        gp1_l2 = 2.1607

        gp1_cov = gs.Matern(dim=2, var=gp1_sf, len_scale=[gp1_l1, gp1_l2],
                            nugget = gp1_sn, nu=nu)

        krige_one = OrdinaryKriging(X_one, Y_one, Z_one, variogram_model=gp1_cov)
        pred_one, var_one = krige_one.execute("grid", range_x, range_z)

        # Kriging wind data (Standard ordinary kriging)
        # 应用标准普通克里金法对风速数据进行插值
        meteo_df = anemo.meteorology_data(meteo_file, start, end)  # 获取气象数据
        sw_corr_factor = [wcomponent.swise_correction(wd, m) for wd in meteo_df["wd"]]  # 对数据进行风向校正
        meteo_df["sw_factor"] = sw_corr_factor  # 将风向校正因子添加到数据中

        meteo_df["stream_wind"] = meteo_df["ws"] * meteo_df["sw_factor"]  # 计算流场风速
        meteo_df["dist"] = krige_ppm["dist"]  # 将距离数据添加到气象数据中
        meteo_df["agl"] = krige_ppm["agl"]  # 将高度数据添加到气象数据中
        meteo_df = meteo_df.dropna()  # 删除含有空值的行

        init_gpw_l1 = mm.init_lengthscale(meteo_df["dist"], 1)  # 初始化横向长度尺度
        init_gpw_l2 = mm.init_lengthscale(meteo_df["agl"], 1)  # 初始化纵向长度尺度
        init_gpw_sf = mm.init_variance(meteo_df["stream_wind"], 1)  # 初始化流场风速方差
        init_gpw_sn = 0.25 * init_gpw_sf  # 初始化噪声方差

        gpw_kernel = mm.matern_kernel(init_gpw_sf, [init_gpw_l1, init_gpw_l2],
                                      nu, 1e-3, max_dist)  # 初始化高斯过程核函数
        gpw_sf, gpw_l1, gpw_l2 = mm.EM_hyperparam(meteo_df["dist"],
                                                  meteo_df["agl"],
                                                  meteo_df["stream_wind"],
                                                  gpw_kernel, init_gpw_sn)  # 用期望最大化算法计算高斯过程核函数超参数
        gpw_sn = 0.25 * gpw_sf  # 计算噪声方差

        gpw_cov = gs.Matern(dim=2, var=gpw_sf, len_scale=[gpw_l1, gpw_l2],
                            nugget=gpw_sn, nu=nu)  # 构建高斯过程协方差函数

        wind_krige = OrdinaryKriging(meteo_df["dist"], meteo_df["agl"],
                                     meteo_df["stream_wind"], variogram_model=gpw_cov)  # 创建普通克里金对象
        wind_prof, wind_ss = wind_krige.execute("grid", range_x, range_z)  # 进行网格插值，并得到插值结果和半方差结果

        ## Emission computation
        # 计算排量
        mean_pres = np.mean(pres)  # 平均气压
        mean_temp = np.mean(temp)  # 平均温度
        mic_meas = conversion.ppm2micro(pred_one, mass=16.04, temp=mean_temp, pres=mean_pres)  # 将预测的浓度值从ppm转换为微克/立方米

        krige_meas = estimate.kriging_estimate(mic_meas, wind_prof)  # 计算克里金估计浓度
        okpw_flux = krige_meas * cfg.dx * cfg.dz * 1e-6  # 计算排放通量，dx和dz为网格尺寸的水平和垂直分辨率

        ## Uncertainty computation
        # 不确定性计算
        # 计算网格点之间的欧几里得距离平方
        square_dist = target_grid.square_matrix()
        range_covx = np.arange(square_dist.shape[0])
        range_covz = np.arange(square_dist.shape[1])

        # 计算gp1_cov的协方差矩阵
        ppm_cov = gp1_cov.covariance(square_dist)
        # 将ppm_cov转换为污染物浓度的协方差矩阵
        conc_cov = conversion.ppm2micro(ppm_cov, mass=16.04, temp=mean_temp, pres=mean_pres)
        conc_cov = conversion.ppm2micro(conc_cov, mass=16.04, temp=mean_temp, pres=mean_pres)

        # 计算gpw_cov的协方差矩阵
        wind_cov = gpw_cov.covariance(square_dist)

        # 计算Kriging估计的不确定性
        okpw_var = estimate.kriging_uncertainty(mic_meas, wind_prof, conc_cov, wind_cov)
        # 计算Kriging估计的标准偏差
        okpw_std = np.sqrt(okpw_var) * cfg.dx * cfg.dz * 1e-6

        print("================================")
        print("\n\nOKPW estimate is: {:.4f}".format(okpw_flux))
        print("\nOKPW Uncertainty estimate is: {:.4f}".format(okpw_std))

        ## OKSW
        ## Domininant perpendicular wind
        # mean_perp_wind = meteo_df["stream_wind"].mean()
        # std_perp_wind = meteo_df["stream_wind"].std()

        # sw_mean = np.full((nz,nx), mean_perp_wind)
        # sw_ss = np.full((nz,nx), std_perp_wind**2)

        # oksw_mean = estimate.kriging_estimate(mic_meas, sw_mean)
        # oksw_flux = oksw_mean * dx * dz * 1e-6

        # oksw_var = estimate.kriging_uncertainty(mic_meas, sw_mean, conc_cov, std_perp_wind)
        # oksw_std = np.sqrt(oksw_var) * dx * dz * 1e-6

        # print("================================")
        # print("\n\nOKPW estimate is: {:.4f}".format(oksw_flux))
        # print("\nOKPW Uncertainty estimate is: {:.4f}".format(oksw_std))


        krige_results = xr.Dataset({
                     "z" : xr.DataArray(range_z, dims=("z",)),
                     "x" : xr.DataArray(range_x, dims=("x",)),
                     "cov_z" : xr.DataArray(range_covz, dims=("cov_z")),
                     "cov_x" : xr.DataArray(range_covx, dims=("cov_x")),
                     "ppm_mean" : xr.DataArray(pred_one, dims=("z", "x")),
                     "ppm_variance" : xr.DataArray(var_one, dims=("z", "x")),
                     "wind_mean" : xr.DataArray(wind_prof, dims=("z", "x")),
                     "wind_variance" : xr.DataArray(wind_ss, dims=("z", "x")),
                     "concentration" : xr.DataArray(mic_meas, dims=("z","x")),
                     "ppm_covmat" : xr.DataArray(ppm_cov, dims=("cov_z", "cov_x")),
                     "conc_covmat" : xr.DataArray(conc_cov, dims=("cov_z", "cov_x")),
                     "wind_covmat" : xr.DataArray(wind_cov, dims=("cov_z", "cov_x")),
                     })

        krige_results.attrs["curtain_r2"] = r_value
        krige_results.attrs["perp_distance"] = perp_distance
        krige_results.attrs["mass_emission"] = okpw_flux
        krige_results.attrs["std_mass_emission"] = okpw_std
        krige_results.attrs["flight_code"] = flight_code
        krige_results.attrs["source_lon"] = source_lon
        krige_results.attrs["source_lat"] = source_lat
        krige_results.attrs["mean_pres"] = mean_pres
        krige_results.attrs["mean_temp"] = mean_temp

        krige_results.to_netcdf(fkrige_results, mode="w")

        print("Successfully written the emission file")

    else:
        pass

    return cfg



if __name__ ==  "__main__":
    try:
        config_path = sys.argv[1]
    except IndexError:
        raise RuntimeError("Please supply a config file.")

    print("Computing methane fluxes using mass balance...")
    cfg = main(config_path)

    # 从配置中获取保存路径
    target_path = cfg.model_path
    # 获取 fkrige 结果的路径
    fkrige_results = cfg.fkrige_results

    ## 从 kriging 结果中加载 netCDF 文件
    krige_results = xr.open_dataset(fkrige_results)

    # 获取 x 和 z 的维度
    nx = krige_results.dims["x"]
    nz = krige_results.dims["z"]

    # 计算每个网格的 x 和 z 坐标之间的距离
    dx = np.around(krige_results.x[1] - krige_results.x[0], 2)
    dz = np.around(krige_results.z[1] - krige_results.z[0], 2)

    # 获取 xmin 和 zmin 的值
    xmin = np.asarray(krige_results.x)[0] - dx / 2
    zmin = np.asarray(krige_results.z)[0] - dz / 2

    # 获取观测值预测值、浓度和风速的均值和方差
    obs_pred = np.asarray(krige_results.ppm_mean)
    obs_var = np.asarray(krige_results.ppm_variance)
    mic_meas = np.asarray(krige_results.concentration)
    wind_mean = np.asarray(krige_results.wind_mean)
    wind_var = np.asarray(krige_results.wind_variance)

    # 获取源的经纬度、纵向贡献和垂线距离
    source_lon = krige_results.attrs["source_lon"]
    source_lat = krige_results.attrs["source_lat"]
    r_value = krige_results.attrs["curtain_r2"]
    perp_distance = krige_results.attrs["perp_distance"]

    # 从配置中获取 NoXP 和 b
    NoXP = cfg.rebs_NoXP
    b = cfg.rebs_b

    # 加载 memo 文件
    memo = readmemoascii.memoCDF(cfg.memo_file, cfg.start, cfg.end)
    # 从 memo 中获取质量平衡数据
    memo_df = memo.massbalance_data(NoXP, b)

    # 获取浓度和海拔的相关系数
    con_ab = memo_df["con_ab"]

    ## Align altitude of GPS with inlet of instrument
    # 将GPS高度与仪器进气口对齐
    agl = memo_df["Altitude"]

    ## COMPUTE PERPENDICULAR DISTANCE BETWEEN SOURCE AND CURTAIN
    # 计算源和幕之间的垂直距离
    dtm = memo_df.index
    lon = memo_df["Longitude"]
    lat = memo_df["Latitude"]
    dist = memo_df["Distance"]

    # 存储生成的图片路径
    im_meas_scatter = os.path.join(target_path, "measurement_scatter.png")
    im_obs_pred = os.path.join(target_path, "ppm_meas.png")
    im_obs_pred_scatter = os.path.join(target_path, "ppm_meas_scatter.png")
    im_obs_var = os.path.join(target_path, "ppm_ss.png")
    im_mic_meas = os.path.join(target_path, "mic_meas.png")
    im_proj_meas = os.path.join(target_path, "projected_conc_map.png")
    im_wind_prof = os.path.join(target_path, "wind_prof.png")
    im_above_bg = os.path.join(target_path, "above_background.png")

    # 是否需要保存生成的图片
    save_images = True

    if os.path.isfile(im_meas_scatter):
        print("Overwrite the plots in %s?" % target_path)
        s = input("y/[n] \n")
        save_images = (s=="y")

    if save_images:
        curtain_plot = plotting.GridPlotting(nx, nz, dx, dz, xmin, zmin)

        ## Scatter plot displaying measured values with respect to height
        fig = plotting.geo_scatter(dtm, dist, agl, con_ab, "Distance [m]",
                                   "Altitude [m]", "CH$_4$ [ppm]")

        ## curtain plot of prediction field of methane molar fraction
        fig2 = curtain_plot.curtainPlots(obs_pred, units = "CH$_4$ [ppm]",
                                         title = "Predicted Measured Molar Fraction")

        ## curtain plot of prediction variance field of methane molar fraction
        fig3 = curtain_plot.curtainPlots(obs_var, units = "CH$_4$ [ppm$^2$]",
                                         title = "Prediction variance")

        ## curtain plot of prediction field overlaid with measurement points
        fig4 = curtain_plot.curtainScatterPlots(dist, agl, con_ab, obs_pred,
                                                units = "CH$_4$ [ppm]", title =
                                                "Predicted Measured Molar Fraction")

        ## curtain plot displaying krige measured concentration
        fig5 = curtain_plot.curtainPlots(mic_meas, units = "CH$_4$ [$\mu$g/m$^3$]",
                                         title = "Predicted Measured Concentration")

        ## Wind profile plot
        fig6 = curtain_plot.curtainPlots(wind_mean, units="[m/s]",
                                         title="Kriging streamwise wind")

        ## Map plot of point source and projected points
        fig7 = curtain_plot.projected_map(source_lon, source_lat, lon, lat,
                                          con_ab, r_value, perp_distance,
                                          units="CH4 [ppm]")


        ## Time series of methane elevations from background
        fig8 = plotting.measurevsbg(dtm, con_ab, agl)


        ## SAVE FIGURE
        fig.savefig(im_meas_scatter, dpi=300)
        fig2.savefig(im_obs_pred, dpi=300)
        fig3.savefig(im_obs_var, dpi=300)
        fig4.savefig(im_obs_pred_scatter, dpi=300)
        fig5.savefig(im_mic_meas, dpi=300)
        fig6.savefig(im_wind_prof, dpi=300)
        fig7.savefig(im_proj_meas, dpi=300)
        fig8.savefig(im_above_bg, dpi=300)
