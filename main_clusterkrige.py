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
from romeomemo.utilities import conversion, grids, plotting
from romeomemo.utilities import readconfig, readmemoascii

from romeomemo.functions import estimate

# 下面是一个名为main的函数，它接受一个参数cfg_path，表示配置文件的路径。
# 该函数包含了一系列变量，用于存储配置文件中指定的各种参数。
# 该函数的主要任务是读取指定的文件并进行处理，其中包括读取配置文件、加载数据文件、初始化文件、计算相关参数等。
def main(cfg_path):
    
    cfg = readconfig.load_cfg(cfg_path)

    memo_file = cfg.memo_file
    rug_file = cfg.rug_file
    meteo_file = cfg.meteo_file
    flight_code = cfg.flight_code

    # horizontal and vertical grid steps
    dx = cfg.dx
    dz = cfg.dz

    ## start and end time of mass balance
    start = cfg.start
    end = cfg.end

    ## REBS parameters
    NoXP = cfg.rebs_NoXP
    b = cfg.rebs_b

    ## LOCATION OF THE SOURCE
    source_lon = cfg.source_lon
    source_lat = cfg.source_lat

    ## Meteorology configuration
    meteo_file = cfg.meteo_file

    # Define target path
    target_path = cfg.model_path

    ## INITIALIZE FILES
    krige_fname = "_".join(["CK", flight_code])
    fkrige_results = os.path.join(target_path, krige_fname + "v2.nc")
    cfg.fkrige_results = fkrige_results

    ### UNCOMMENT BLOCK FOR PROCESSNG QCL FILES ###
    ##################################################
    memo = readmemoascii.memoCDF(memo_file, start, end)
    memo_df = memo.massbalance_data(NoXP, b)

    con_ab = memo_df["con_ab"]

    ## Load spatial parameters
    lon = memo_df["Longitude"]
    lat = memo_df["Latitude"]
    dist = memo_df["Distance"]

    ## Load environmental parameters
    pres = memo_df["Pressure"]
    temp = memo_df["Temperature"]

    agl = memo_df["Altitude"]
    ##################################################

    ## UNCOMMENT BLOCK FOR PROCESSING RUG FILES ##
    ##############################################
    # memo = readmemoascii.memoCDF(memo_file, start, end)
    # memo_qcl_df = memo.massbalance_data(NoXP, b)

    # memo = readmemoascii.memoCDF(rug_file, start, end)
    # memo_df = memo.massbalance_rugdata(NoXP, b)

    # memo_qcl_df = memo_qcl_df[memo_qcl_df.index.isin(memo_df.index)]
    # memo_df = memo_df[memo_df.index.isin(memo_qcl_df.index)]

    # con_ab = memo_df["con_ab"]
    # ## Align altitude of GPS with inlet of instrument
    # # Distance between the GPS and the altitude is 72 cm
    # # agl = memo_df["Altitude"]

    # ## Load spatial parameters
    # lon = memo_df["Longitude"]
    # lat = memo_df["Latitude"]
    # dist = memo_df["Distance"]

    # # ## Load environmental parameters
    # pres = memo_qcl_df["Pressure"]
    # temp = memo_qcl_df["Temperature"]

    # agl = memo_qcl_df["Altitude"]
    ###############################################

    ### UNCOMMENT BLOCK FOR PROCESSING FIT FILES ##
    # memo = readmemoascii.memoCDF(memo_file, start, end)
    # memo_df = memo.massbalance_data(NoXP, b)

    # try:
    #     fit_file = cfg.fit_file
    # except NameError:
    #     raise FileNotFoundError(f"Fitted file does not exist")

    # fit_ds = xr.open_dataset(fit_file)
    # fit_df = fit_ds.to_dataframe()
    # fit_df = fit_df.dropna()
    # con_ab = fit_df["y_ac"]
    # lon = fit_df["longitude"]
    # lat = fit_df["latitude"]
    # alt = fit_df["altitude"]

    # dist = fit_df["distance"]

    # ## Load environmental parameters
    # pres = memo_df["Pressure"]
    # temp = memo_df["Temperature"]
    # agl = memo_df["Altitude"]
    # ###############################################

    ## Replace NANs with STP values
    pres[pres == -999.99] = 1013.25  # 将NaN值替换为STP值（1013.25）
    temp[temp == -999.99] = 273.15  # 将NaN值替换为273.15

    ## Convert mole fraction to concentration
    # con_ab = conversion.ppm2micro(ppm_ab, temp=temp, pres=pres, mass=16.04)

    proc_gps = grids.ProcessCoordinates(lon, lat)  # 创建处理经纬度的对象
    m, b, r_value = proc_gps.regression_coeffs()  # 获取回归系数

    perp_distance = grids.perpendicular_distance(lon, lat, source_lon, source_lat)  # 获取经纬度坐标与给定坐标之间的垂直距离

    xmin = 0.0  # x轴的最小值
    zmin = 0.0  # z轴的最小值
    nx = int(np.floor(np.max(dist) / dx) + 1)  # x轴方向的节点数
    nz = int(np.floor(np.max(agl) / dz) + 1)  # z轴方向的节点数

    ## CREATE TARGET GRID
    target_grid = grids.Grid(nx, nz, dx, dz, xmin, zmin)  # 创建目标网格
    range_x = target_grid.xdist_range()  # x轴的范围
    range_z = target_grid.alt_range()  # z轴的范围

    if not os.path.exists(target_path):  # 如果目标路径不存在，则创建一个目标文件夹
        os.makedirs(target_path)

    save_arrays = True  # 保存数组的标志
    if os.path.isfile(fkrige_results):  # 如果文件已经存在
        print("Overwite interpolated arrays in %s? " % target_path)  # 显示询问消息
        s = input("y/[n] \n")  # 提示用户输入y或n
        save_arrays = (s == "y")  # 如果用户输入y，那么将save_arrays设置为True
    # save_arrys讲解：
    # 如果 save_arrays 变量为真，则开始执行克里金插值。
    # 首先将观测站点的经纬度坐标和污染物浓度转换为 numpy 数组 obs_x 和 obs_y。
    # 创建高斯混合模型对象 gmm，并将 obs_x 和 obs_y 作为训练数据。
    # 训练高斯混合模型，并获取簇的概率和分类标签，以及权重。
    # 创建网格上的点的坐标 post_points，并计算每个点属于每个簇的概率。
    # 将数据保存到 pandas DataFrame krige_ppm 中，包括距离、高度、污染物浓度、分类标签和概率。
    if save_arrays:  # 如果需要保存数组

        obs_x = np.asarray(list(zip(dist, agl)))  # 观测站点的经纬度坐标
        obs_y = np.asarray(con_ab)  # 观测站点的污染物浓度

        gmm = mm.GMM(n_clusters=2)  # 创建高斯混合模型对象，其中n_clusters为聚类数目
        gmm.set_training_values(obs_x, obs_y)  # 设置训练数据
        gmm.train()  # 训练高斯混合模型

        prob_clust, hard_clust = gmm.cluster_data()  # 获取簇的概率和分类标签
        weights = gmm.weights()  # 获取权重

        ## Compute posteriori membership probabilities
        post_points = []
        for j, z in enumerate(range_z):
            for i, x in enumerate(range_x):
                post_points.append([x, z])
        post_points = np.asarray(post_points)  # 创建网格上的点的坐标

        membership_prob = gmm.prob_membership(post_points)  # 计算每个点属于簇的概率

        prob_one = membership_prob[:,1].reshape(nz, nx)  # 将计算出来的每个点属于簇一的概率重新变为原始网格大小
        prob_two = membership_prob[:,0].reshape(nz, nx)  # 将计算出来的每个点属于簇二的概率重新变为原始网格大小
        # opp_clust = np.where((hard_clust==0)|(hard_clust==1), hard_clust^1, hard_clust)

        # 创建一个 DataFrame 对象，用于存储普通克里金插值的结果
        # 这些代码创建了一个名为“krige_ppm”的DataFrame对象，
        # 用于存储普通克里金插值的结果。
        # DataFrame对象具有五列：分别为“dist”、“agl”、“obs_data”、“hard_cluster”和“prob”，
        # 这些列包含了各种克里金插值需要的数据。
        krige_ppm = pd.DataFrame(index=dist.index)
        krige_ppm["dist"] = dist
        krige_ppm["agl"] = agl
        krige_ppm["obs_data"] = con_ab
        krige_ppm["hard_cluster"] = hard_clust
        krige_ppm["prob"] = prob_clust[:,1]

        # 这些代码计算了所需的参数值。max_dist 是 krige_ppm 中“dist”列的最大值。
        max_dist = np.max(krige_ppm["dist"])
        agl_series = pd.Series(krige_ppm["agl"]).diff() # agl_series 是“agl”列的差分序列，表示相邻值之间的高度差。
        agl_max = agl_series.max() # agl_max 是 agl_series 的最大值，表示最大高度差。

        ## Component zero
        # 这些代码是用于分离第一类（hard_cluster == 1）的数据。
        # 选择数据中 hard_cluster 列为 1 的行，将其赋值给 cl_one 变量
        cl_one = krige_ppm[krige_ppm["hard_cluster"] == 1]

        # 将 cl_one 数据中的 "dist" 列转换为 numpy 数组，存储到变量 X_one 中
        X_one = np.asarray(cl_one["dist"])

        # 将 cl_one 数据中的 "agl" 列转换为 numpy 数组，存储到变量 Y_one 中
        Y_one = np.asarray(cl_one["agl"])

        # 将 cl_one 数据中的 "obs_data" 列转换为 numpy 数组，存储到变量 Z_one 中
        Z_one = np.asarray(cl_one["obs_data"])

        # 选择数据中 hard_cluster 列为 1 的行，将其赋值给 cl_one 变量
        cl_one = krige_ppm[krige_ppm["hard_cluster"] == 1]

        # 将 cl_one 数据中的 "dist" 列转换为 numpy 数组，存储到变量 X_one 中
        X_one = np.asarray(cl_one["dist"])

        # 将 cl_one 数据中的 "agl" 列转换为 numpy 数组，存储到变量 Y_one 中
        Y_one = np.asarray(cl_one["agl"])

        # 将 cl_one 数据中的 "obs_data" 列转换为 numpy 数组，存储到变量 Z_one 中
        Z_one = np.asarray(cl_one["obs_data"])

        # 使用 mm 模块中的函数初始化 GP1 模型的 l1, l2, sf 和 sn，存储到变量 init_gp1_l1, init_gp1_l2, init_gp1_sf 和 init_gp1_sn 中
        init_gp1_l1 = mm.init_lengthscale(cl_one["dist"], cl_one["prob"])
        init_gp1_l2 = mm.init_lengthscale(cl_one["agl"], cl_one["prob"])
        init_gp1_sf = mm.init_variance(cl_one["obs_data"], cl_one["prob"])
        init_gp1_sn = 0.25 * init_gp1_sf

        # 初始化 GP1 的核函数，存储到变量 gp1_kernel 中
        nu = 1.5
        gp1_kernel = mm.matern_kernel(init_gp1_sf, [init_gp1_l1, init_gp1_l2], nu, agl_max, max_dist)

        # 使用 EM 算法优化 GP1 模型的 l1, l2 和 sf，存储到变量 gp1_l1, gp1_l2 和 gp1_sf 中
        gp1_sf, gp1_l1, gp1_l2 = mm.EM_hyperparam(X_one, Y_one, Z_one, gp1_kernel, init_gp1_sn)

        # 计算 GP1 模型的 sn，存储到变量 gp1_sn 中
        gp1_sn = np.mean(gp1_sf / np.asarray(cl_one["prob"]))

        # 输出 GP1 模型的水平长度尺度和竖直长度尺度
        print("Horizontal length scale GP1: {:.4f}".format(gp1_l1))
        print("Vertical length scale GP1: {:.4f}".format(gp1_l2))

        # 构建 GP1 的协方差矩阵
        gp1_cov = gs.Matern(dim=2, var=gp1_sf, len_scale=[gp1_l1, gp1_l2], nugget=gp1_sn, nu=nu)

        # 使用 OrdinaryKriging 方法构建 krige_one 模型
        krige_one = OrdinaryKriging(X_one, Y_one, Z_one, variogram_model=gp1_cov)
        pred_one, var_one = krige_one.execute("grid", range_x, range_z)

        ## Component two
        # 这段代码的作用是用于构建第二个高斯过程。主要步骤如下：
        # 首先通过计算 1-krige_ppm[“hard_cluster”] 和 1-krige_ppm[“prob”] 得到新的 hard_cluster 和 prob 的互补值，以得到第二个高斯过程的样本点。
        # 将互补值 comp_prob 添加到 krige_ppm 数据集中。
        #
        # 从 krige_ppm 中选择 hard_cluster 等于 0 的行作为第二个高斯过程的输入样本点。将 dist、agl 和 obs_data 列转换为 numpy 数组 X_two、Y_two 和 Z_two。
        #
        # 初始化高斯过程的长度尺度 init_gp2_l1、init_gp2_l2 和方差 init_gp2_sf。
        #
        # 计算高斯过程的噪声方差 init_gp2_sn。
        #
        # 通过使用 init_gp2_sf、init_gp2_l1、init_gp2_l2 和协方差函数中的 nu、agl_max 和 max_dist 来构建 Matern 核 gp2_kernel。
        #
        # 通过调用 EM_hyperparam 函数来学习高斯过程的超参数 gp2_sf、gp2_l1 和 gp2_l2。
        #
        # 计算高斯过程的噪声方差 gp2_sn。
        #
        # 根据学习到的 gp2_sf、gp2_l1、gp2_l2 和 gp2_sn 构建高斯过程的协方差矩阵 gp2_cov。
        #
        # 用 X_two、Y_two、Z_two 和 gp2_cov 来构建 OrdinaryKriging 对象 krige_two。
        #
        # 调用 krige_two 对象的 execute 方法来预测网格上的值 pred_two 和方差 var_two。
        comp_clust = 1 - krige_ppm["hard_cluster"]
        comp_prob = 1 - krige_ppm["prob"]
        krige_ppm["comp_prob"] = comp_prob
        cl_two = krige_ppm[comp_clust==1]
        X_two = np.asarray(cl_two["dist"])
        Y_two = np.asarray(cl_two["agl"])
        Z_two = np.asarray(cl_two["obs_data"])

        init_gp2_l1 = mm.init_lengthscale(cl_two["dist"], cl_two["comp_prob"])
        init_gp2_l2 = mm.init_lengthscale(cl_two["agl"], cl_two["comp_prob"])
        init_gp2_sf = mm.init_variance(cl_two["obs_data"], cl_two["comp_prob"])
        init_gp2_sn = 0.25 * init_gp2_sf
        init_gp2_sn /= np.asarray(cl_two["comp_prob"])

        gp2_kernel = mm.matern_kernel(init_gp2_sf, [init_gp2_l1, init_gp2_l2],
                                      nu, agl_max, max_dist)
        gp2_sf, gp2_l1, gp2_l2 = mm.EM_hyperparam(X_two, Y_two, Z_two, gp2_kernel, init_gp2_sn)
        gp2_sn = np.mean(gp2_sf/np.asarray(cl_two["comp_prob"]))

        print("Horizontal length scale GP2: {:.4f}".format(gp2_l1))
        print("Vertical length scale GP2: {:.4f}".format(gp2_l2))

        gp2_cov = gs.Matern(dim=2, var=gp2_sf, len_scale=[gp2_l1, gp2_l2],
                            nugget=gp2_sn, nu=nu)

        krige_two = OrdinaryKriging(X_two, Y_two, Z_two, variogram_model=gp2_cov)
        pred_two, var_two = krige_two.execute("grid", range_x, range_z)
        ##############################################################################
        # Kriging wind data (Standard ordinary kriging)
        # 从气象文件中读取气象数据
        meteo_df = anemo.meteorology_data(meteo_file, start, end)

        # 计算风向修正因子
        sw_corr_factor = [wcomponent.swise_correction(wd, m) for wd in meteo_df["wd"]]
        meteo_df["sw_factor"] = sw_corr_factor

        # 计算流速风
        meteo_df["stream_wind"] = meteo_df["ws"] * meteo_df["sw_factor"]

        # 将气象数据的距离和高度等属性设置为 krige_ppm 中相应的属性
        meteo_df["dist"] = krige_ppm["dist"]
        meteo_df["agl"] = krige_ppm["agl"]

        # 删除有缺失值的行
        meteo_df = meteo_df.dropna()

        # 初始化高斯过程的长度尺度、方差和信噪比
        init_gpw_l1 = mm.init_lengthscale(meteo_df["dist"], 1)
        init_gpw_l2 = mm.init_lengthscale(meteo_df["agl"], 1)
        init_gpw_sf = mm.init_variance(meteo_df["stream_wind"], 1)
        init_gpw_sn = 0.25 * init_gpw_sf

        # 构建高斯过程核函数
        gpw_kernel = mm.matern_kernel(init_gpw_sf, [init_gpw_l1, init_gpw_l2], nu, agl_max, max_dist)

        # 通过 EM 算法优化高斯过程的超参数
        gpw_sf, gpw_l1, gpw_l2 = mm.EM_hyperparam(meteo_df["dist"], meteo_df["agl"], meteo_df["stream_wind"],
                                                  gpw_kernel, init_gpw_sn)

        # 计算信噪比
        gpw_sn = 0.25 * gpw_sf

        # 构建高斯过程协方差函数
        gpw_cov = gs.Matern(dim=2, var=gpw_sf, len_scale=[gpw_l1, gpw_l2], nugget=gpw_sn, nu=nu)

        # 进行插值
        wind_krige = OrdinaryKriging(meteo_df["dist"], meteo_df["agl"], meteo_df["stream_wind"],
                                     variogram_model=gpw_cov)
        wind_prof, wind_ss = wind_krige.execute("grid", range_x, range_z)
        #######################################################################################
        ## Final prediction，对前面两个步骤进行整合，用来计算最终的排放通量的预测值和方差，以及对流风场。
        # 将两个硬聚类中的 Kriging 预测值按照它们在观测点上出现的概率进行加权平均，得到最终的预测值。
        obs_pred = (prob_one * pred_one) + (prob_two * pred_two)

        ## Final variance
        # 计算最终的方差，使用了方差的线性可加性。
        # 对于每个硬聚类中的方差和预测值，使用观测点上的出现概率将它们加权相加，
        # 得到一个总的加权平均值，再用它们之间的关系计算总的方差。
        res_one = prob_one * (var_one + pred_one**2)
        res_two = prob_two * (var_two + pred_two**2)

        obs_var = res_one + res_two - obs_pred ** 2

        ## Emission computation
        # 首先，计算出环境温度和压强的平均值，
        # 然后将最终的预测值（obs_pred）从单位为 ppm 转换为单位为 μg/m³ 的质量浓度。
        # 接着，使用刚刚计算出的横向和竖向风速的 Kriging 预测值'wind_profkrige_meas，
        # 然后将其乘以每个方格的面积d'x'和'dz'以及单位转换系数1e-6，得到最终的排放通量的预测值'flux_estimate'。
        mean_pres = np.mean(pres)
        mean_temp = np.mean(temp)
        mic_meas = conversion.ppm2micro(obs_pred, mass=16.04, temp=mean_temp, pres=mean_pres)

        krige_meas = estimate.kriging_estimate(mic_meas, wind_prof)
        flux_estimate = krige_meas * cfg.dx * cfg.dz * 1e-6

        ###########################################################################3
        ## Uncertainty computation,该段代码主要是计算一个空气污染源的排放量以及相应的不确定性。
        # 其中，一些主要使用的函数的功能如下：
        #
        # square_matrix(): 计算距离矩阵；
        # covariance(): 计算高斯过程的协方差矩阵；
        # cluster_covariance(): 估算簇的协方差；
        # ppm2micro(): 将单位从ppm转换为μg/m³；
        # kriging_uncertainty(): 估算克里金法的不确定性；
        # sqrt(): 计算平方根。
        ## 计算距离矩阵
        square_dist = target_grid.square_matrix()
        ## 计算x和z方向的协方差矩阵的索引范围
        range_covx = np.arange(square_dist.shape[0])
        range_covz = np.arange(square_dist.shape[1])

        ## 计算两个高斯过程的协方差矩阵
        gp1_covmat = gp1_cov.covariance(square_dist)
        gp2_covmat = gp2_cov.covariance(square_dist)

        ## 估算两个簇的协方差
        sigma_c1 = estimate.cluster_covariance(prob_one, gp1_covmat)
        sigma_c2 = estimate.cluster_covariance(prob_two, gp2_covmat)

        ## 计算浓度协方差
        ppm_cov = sigma_c1 + sigma_c2
        conc_cov = conversion.ppm2micro(ppm_cov, mass=16.04, temp=mean_temp, pres=mean_pres)
        conc_cov = conversion.ppm2micro(conc_cov, mass=16.04, temp=mean_temp, pres=mean_pres)

        ## 计算风速协方差
        wind_cov = gpw_cov.covariance(square_dist)

        ## 估算克里金法的不确定性和通量标准差
        var_flux = estimate.kriging_uncertainty(mic_meas, wind_prof, conc_cov, wind_cov)
        sigma_flux = np.sqrt(var_flux) * cfg.dx * cfg.dz * 1e-6


        print("================================")
        print("\n\nEmission estimate is: {:.4f}".format(flux_estimate))
        print("\n\nUncertainty estimate is: {:.4f}".format(sigma_flux))


        # 这段Python代码创建了一个名为krige_results的xarray Dataset对象，包含了许多不同的DataArray数组对象。
        # 下面是每个数组对象的注释
        # "z"：一个DataArray，包含了z维度上的坐标点。
        # "x"：一个DataArray，包含了x维度上的坐标点。
        # "cov_z"：一个DataArray，包含了在z方向上用于Kriging的协方差距离范围。
        # "cov_x"：一个DataArray，包含了在x方向上用于Kriging的协方差距离范围。
        # "ppm_cl_one"：一个DataArray，包含了第一类污染物的预测浓度。
        # "ppm_cl_two"：一个DataArray，包含了第二类污染物的预测浓度。
        # "var_cl_one"：一个DataArray，包含了第一类污染物的方差。
        # "var_cl_two"：一个DataArray，包含了第二类污染物的方差。
        # "prob_cl_one"：一个DataArray，包含了第一类污染物的概率。
        # "prob_cl_two"：一个DataArray，包含了第二类污染物的概率。
        # "ppm_mean"：一个DataArray，包含了观测和预测浓度的平均值。
        # "ppm_variance"：一个DataArray，包含了观测和预测浓度的方差。
        # "wind_mean"：一个DataArray，包含了风速和方向的平均值。
        # "wind_variance"：一个DataArray，包含了风速和方向的方差。
        # "concentration"：一个DataArray，包含了测量的微生物浓度。
        # "cov_one"：一个DataArray，包含了第一类污染物的协方差矩阵。
        # "cov_two"：一个DataArray，包含了第二类污染物的协方差矩阵。
        # "ppm_covmat"：一个DataArray，包含了污染物浓度的协方差矩阵。
        # "conc_covmat"：一个DataArray，包含了微生物浓度的协方差矩阵。
        # "wind_covmat"：一个DataArray，包含了风速和方向的协方差矩阵。
        krige_results = xr.Dataset({
                     "z" : xr.DataArray(range_z, dims=("z",)),
                     "x" : xr.DataArray(range_x, dims=("x",)),
                     "cov_z" : xr.DataArray(range_covz, dims=("cov_z")),
                     "cov_x" : xr.DataArray(range_covx, dims=("cov_x")),
                     "ppm_cl_one" : xr.DataArray(pred_one, dims=("z", "x")),
                     "ppm_cl_two" : xr.DataArray(pred_two, dims=("z", "x")),
                     "var_cl_one" : xr.DataArray(var_one, dims=("z", "x")),
                     "var_cl_two" : xr.DataArray(var_two, dims=("z", "x")),
                     "prob_cl_one" : xr.DataArray(prob_one, dims=("z", "x")),
                     "prob_cl_two" : xr.DataArray(prob_two, dims=("z", "x")),
                     "ppm_mean" : xr.DataArray(obs_pred, dims=("z", "x")),
                     "ppm_variance" : xr.DataArray(obs_var, dims=("z", "x")),
                     "wind_mean" : xr.DataArray(wind_prof, dims=("z", "x")),
                     "wind_variance" : xr.DataArray(wind_ss, dims=("z", "x")),
                     "concentration" : xr.DataArray(mic_meas, dims=("z","x")),
                     "cov_one" : xr.DataArray(gp1_covmat, dims=("cov_z", "cov_x")),
                     "cov_two" : xr.DataArray(gp2_covmat, dims=("cov_z", "cov_x")),
                     "ppm_covmat" : xr.DataArray(ppm_cov, dims=("cov_z", "cov_x")),
                     "conc_covmat" : xr.DataArray(conc_cov, dims=("cov_z", "cov_x")),
                     "wind_covmat" : xr.DataArray(wind_cov, dims=("cov_z", "cov_x")),
                     })

        # 为 krige_results 对象添加属性
        # krige_results.attrs["curtain_r2"] = r_value # 添加属性 curtain_r2, 值为 r_value
        # krige_results.attrs["perp_distance"] = perp_distance # 添加属性 perp_distance, 值为 perp_distance
        # krige_results.attrs["mass_emission"] = flux_estimate # 添加属性 mass_emission, 值为 flux_estimate
        # krige_results.attrs["std_mass_emission"] = sigma_flux # 添加属性 std_mass_emission, 值为 sigma_flux
        # krige_results.attrs["flight_code"] = flight_code # 添加属性 flight_code, 值为 flight_code
        # krige_results.attrs["source_lon"] = source_lon # 添加属性 source_lon, 值为 source_lon
        # krige_results.attrs["source_lat"] = source_lat # 添加属性 source_lat, 值为 source_lat
        # krige_results.attrs["cl_one_lscale"] = [gp1_l1, gp1_l2] # 添加属性 cl_one_lscale, 值为 [gp1_l1, gp1_l2]
        # krige_results.attrs["cl_two_lscale"] = [gp2_l1, gp2_l2] # 添加属性 cl_two_lscale, 值为 [gp2_l1, gp2_l2]
        # krige_results.attrs["mean_pres"] = mean_pres # 添加属性 mean_pres, 值为 mean_pres
        # krige_results.attrs["mean_temp"] = mean_temp # 添加属性 mean_temp, 值为 mean_temp
        krige_results.attrs["curtain_r2"] = r_value
        krige_results.attrs["perp_distance"] = perp_distance
        krige_results.attrs["mass_emission"] = flux_estimate
        krige_results.attrs["std_mass_emission"] = sigma_flux
        krige_results.attrs["flight_code"] = flight_code
        krige_results.attrs["source_lon"] = source_lon
        krige_results.attrs["source_lat"] = source_lat
        krige_results.attrs["cl_one_lscale"] = [gp1_l1, gp1_l2]
        krige_results.attrs["cl_two_lscale"] = [gp2_l1, gp2_l2]
        krige_results.attrs["mean_pres"] = mean_pres
        krige_results.attrs["mean_temp"] = mean_temp

        # 将 krige_results 对象保存为 netCDF 格式
        krige_results.to_netcdf(fkrige_results)

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

    # 以下代码为读取QCL文件
    # 从指定路径读取配置文件，并将其传递给 main 函数
    cfg = main(config_path)

    # 从配置文件中获取 dx 和 dz
    dx = cfg.dx
    dz = cfg.dz

    # 初始化用于保存文件的路径
    target_path = cfg.model_path
    fkrige_results = cfg.fkrige_results

    # 从 kriging 结果中的 netCDF 文件加载数据
    krige_results = xr.open_dataset(fkrige_results)

    # 获取 krige_results 的维度
    nx = krige_results.dims["x"]
    nz = krige_results.dims["z"]

    # 获取 krige_results 中的 x 和 z 方向上的跨度
    dx = np.around(krige_results.x[1] - krige_results.x[0], 2)
    dz = np.around(krige_results.z[1] - krige_results.z[0], 2)

    # 计算 krige_results 的最小值，用于确定网格的左下角位置
    xmin = np.asarray(krige_results.x)[0] - dx / 2
    zmin = np.asarray(krige_results.z)[0] - dz / 2

    # 从 krige_results 中提取有用的信息
    obs_pred = np.asarray(krige_results.ppm_mean)
    obs_var = np.asarray(krige_results.ppm_variance)
    mic_meas = np.asarray(krige_results.concentration)
    wind_mean = np.asarray(krige_results.wind_mean)
    wind_var = np.asarray(krige_results.wind_variance)
    source_lon = krige_results.attrs["source_lon"]
    source_lat = krige_results.attrs["source_lat"]
    r_value = krige_results.attrs["curtain_r2"]
    perp_distance = krige_results.attrs["perp_distance"]

    # 从配置文件中获取 NoXP 和 b
    NoXP = cfg.rebs_NoXP
    b = cfg.rebs_b

    # 使用 readmemoascii 包读取 QCL 文件并指定开始和结束时间
    memo = readmemoascii.memoCDF(cfg.memo_file, cfg.start, cfg.end)

    # 使用 NoXP 和 b 处理 QCL 文件
    memo_df = memo.massbalance_data(NoXP, b)

    # 获取浓度和高度的信息
    con_ab = memo_df["con_ab"]
    # 将GPS的高度与仪器的进口对齐，仪器与GPS的距离为72cm
    agl = memo_df["Altitude"]

    # 获取时间戳、经度、纬度和距离信息
    dtm = memo_df.index
    lon = memo_df["Longitude"]
    lat = memo_df["Latitude"]
    dist = memo_df["Distance"]
    ##################################################

    ### UNCOMMENT BLOCK FOR PROCESSNG RUG FILES ###
    ##################################################
    # memo = readmemoascii.memoCDF(cfg.memo_file, cfg.start, cfg.end)
    # memo_qcl_df = memo.massbalance_data(NoXP, b)

    # memo = readmemoascii.memoCDF(cfg.rug_file, cfg.start, cfg.end)
    # memo_df = memo.massbalance_rugdata(NoXP, b)

    # memo_qcl_df = memo_qcl_df[memo_qcl_df.index.isin(memo_df.index)]
    # memo_df = memo_df[memo_df.index.isin(memo_qcl_df.index)]

    # con_ab = memo_df["con_ab"]

    # agl = memo_qcl_df["Altitude"]

    # ## Load spatial parameters
    # dtm = memo_df.index
    # lon = memo_df["Longitude"]
    # lat = memo_df["Latitude"]
    # dist = memo_df["Distance"]
    ##################################################

    # 定义存储路径
    im_meas_scatter = os.path.join(target_path, "measurement_scatter.png")
    im_obs_pred = os.path.join(target_path, "ppm_meas.png")
    im_obs_pred_scatter = os.path.join(target_path, "ppm_meas_scatter.png")
    im_obs_var = os.path.join(target_path, "ppm_ss.png")
    im_mic_meas = os.path.join(target_path, "mic_meas.png")
    im_proj_meas = os.path.join(target_path, "projected_conc_map.png")
    im_wind_prof = os.path.join(target_path, "wind_prof.png")
    im_above_bg = os.path.join(target_path, "above_background.png")
    im_area_conc = os.path.join(target_path, "area_concentrion.png")

    # 是否保存图像
    save_images = True
    # 如果测量散点图已存在，则询问是否覆盖
    if os.path.isfile(im_meas_scatter):
        print("是否覆盖%s中的图像?" % target_path)
        s = input("y/[n] \n")
        save_images = (s == "y")

    if save_images:
        # 创建GridPlotting实例
        curtain_plot = plotting.GridPlotting(nx, nz, dx, dz, xmin, zmin)

        # 绘制测量高度分布的散点图
        fig = plotting.geo_scatter(dtm, dist, agl, con_ab, "Distance [m]",
                                   "Altitude [m]", "CH$_4$ [ppm]")
        # 绘制预测的甲烷浓度幕布图
        fig2 = curtain_plot.curtainPlots(obs_pred, units="CH$_4$ [ppm]",
                                         title="Predicted Measured Mole Fraction")
        # 绘制预测甲烷浓度的方差幕布图
        fig3 = curtain_plot.curtainPlots(obs_var, units="CH$_4$ [ppm$^2$]",
                                         title="Prediction variance")
        # 绘制预测甲烷浓度与测量值散点图的幕布图
        fig4 = curtain_plot.curtainScatterPlots(dist, agl, con_ab, obs_pred,
                                                units="CH$_4$ [ppm]", title=
                                                "Predicted Measured Mole Fraction")
        # 绘制krige法预测的甲烷浓度的幕布图
        fig5 = curtain_plot.curtainPlots(mic_meas, units="CH$_4$ [$\mu$g/m$^3$]",
                                         title="Predicted Measured Concentration")
        # 绘制风速纵向分布的幕布图
        fig6 = curtain_plot.curtainPlots(wind_mean, units="[m/s]",
                                         title="Kriging streamwise wind")
        # 绘制点源与预测点的地图
        fig7 = curtain_plot.projected_map(source_lon, source_lat, lon, lat,
                                          con_ab, r_value, perp_distance,
                                          units="CH$_4$ - CH$_{4\mathrm{bg}}}$ [ppm]")
        # 绘制背景中甲烷浓度变化的时间序列图
        fig8 = plotting.measurevsbg(dtm, con_ab, agl)

        # 保存图像

        fig.savefig(im_meas_scatter, dpi=300)
        fig2.savefig(im_obs_pred, dpi=300)
        fig3.savefig(im_obs_var, dpi=300)
        fig4.savefig(im_obs_pred_scatter, dpi=300)
        fig5.savefig(im_mic_meas, dpi=300)
        fig6.savefig(im_wind_prof, dpi=300)
        fig7.savefig(im_proj_meas, dpi=300)
        fig8.savefig(im_above_bg, dpi=300)

