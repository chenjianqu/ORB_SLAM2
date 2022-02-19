/**
* This file is part of ORB-SLAM2.
* This file is a modified version of EPnP <http://cvlab.epfl.ch/EPnP/index.php>, see FreeBSD license below.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

/**
* Copyright (c) 2009, V. Lepetit, EPFL
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
*   either expressed or implied, of the FreeBSD Project
*/

#include <iostream>

#include "PnPsolver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include "Thirdparty/DBoW2/DUtils/Random.h"
#include <algorithm>

using namespace std;

namespace ORB_SLAM2{\


PnPsolver::PnPsolver(const Frame &F, const vector<MapPoint*> &vpMapPointMatches):
pws(0), us(0), alphas(0), pcs(0), max_num_of_corr(0), num_of_corr(0), mnInliersi(0),
mnIterations(0), mnBestInliers(0), N(0){
    mvpMapPointMatches = vpMapPointMatches;
    mvP2D.reserve(F.mvpMapPoints.size());
    mvSigma2.reserve(F.mvpMapPoints.size());
    mvP3Dw.reserve(F.mvpMapPoints.size());
    mvKeyPointIndices.reserve(F.mvpMapPoints.size());
    mvAllIndices.reserve(F.mvpMapPoints.size());

    int idx=0;
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];

        if(pMP)
        {
            if(!pMP->isBad())
            {
                const cv::KeyPoint &kp = F.mvKeysUn[i];

                mvP2D.push_back(kp.pt);
                mvSigma2.push_back(F.mvLevelSigma2[kp.octave]);

                cv::Mat Pos = pMP->GetWorldPos();
                mvP3Dw.emplace_back(Pos.at<float>(0),Pos.at<float>(1), Pos.at<float>(2));

                mvKeyPointIndices.push_back(i);
                mvAllIndices.push_back(idx);

                idx++;
            }
        }
    }

    // Set camera calibration parameters
    fu = F.fx;
    fv = F.fy;
    uc = F.cx;
    vc = F.cy;

    SetRansacParameters();
}

PnPsolver::~PnPsolver()
{
    delete [] pws;
    delete [] us;
    delete [] alphas;
    delete [] pcs;
}


void PnPsolver::SetRansacParameters(double probability, int minInliers, int maxIterations, int minSet, float epsilon, float th2)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;
    mRansacEpsilon = epsilon;
    mRansacMinSet = minSet;

    N = mvP2D.size(); // number of correspondences

    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    //根据输入参数，确定最小内点数量，最小为4，或N*epsilon
    int nMinInliers = N*mRansacEpsilon;
    if(nMinInliers<mRansacMinInliers)
        nMinInliers=mRansacMinInliers;
    if(nMinInliers<minSet)
        nMinInliers=minSet;
    mRansacMinInliers = nMinInliers;

    if(mRansacEpsilon<(float)mRansacMinInliers/N)
        mRansacEpsilon=(float)mRansacMinInliers/N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    //设置RANSAC的迭代次数
    int nIterations;
    if(mRansacMinInliers==N)
        nIterations=1;
    else
        nIterations = std::ceil(log(1-mRansacProb)/std::log(1-std::pow(mRansacEpsilon,3)));

    mRansacMaxIts = std::max(1,min(nIterations,mRansacMaxIts));

    //每个2D特征点对应不同的误差阈值
    mvMaxError.resize(mvSigma2.size());
    for(size_t i=0; i<mvSigma2.size(); i++)
        mvMaxError[i] = mvSigma2[i]*th2;
}



cv::Mat PnPsolver::find(vector<bool> &vbInliers, int &nInliers)
{
    bool bFlag;
    return iterate(mRansacMaxIts,bFlag,vbInliers,nInliers);
}

/**
 * 1,根据局内点计算mRi,mti
   2,根据mRi,mti,再次计算局内点,如果局内点数量大于设定值(mRansacMinInliers)，
   则将mRi,mti,给到mRefinedTcw，并返回1，否则返回0
 * @return
 */
bool PnPsolver::Refine()
{
    vector<int> vIndices;
    vIndices.reserve(mvbBestInliers.size());
    for(size_t i=0; i<mvbBestInliers.size(); i++){
        if(mvbBestInliers[i]){
            vIndices.push_back((int)i);
        }
    }
    //设置内点的数量
    set_maximum_number_of_correspondences(vIndices.size());
    //重置匹配点对数量为0
    reset_correspondences();
    //添加局内点匹配点对
    for(int idx : vIndices){
        add_correspondence(mvP3Dw[idx].x,mvP3Dw[idx].y,mvP3Dw[idx].z,mvP2D[idx].x,mvP2D[idx].y);
    }
    // Compute camera pose
    compute_pose(mRi, mti);
    // Check inliers
    CheckInliers();

    mnRefinedInliers =mnInliersi;
    mvbRefinedInliers = mvbInliersi;

    if(mnInliersi>mRansacMinInliers)
    {
        cv::Mat Rcw(3,3,CV_64F,mRi);
        cv::Mat tcw(3,1,CV_64F,mti);
        Rcw.convertTo(Rcw,CV_32F);
        tcw.convertTo(tcw,CV_32F);
        mRefinedTcw = cv::Mat::eye(4,4,CV_32F);
        Rcw.copyTo(mRefinedTcw.rowRange(0,3).colRange(0,3));
        tcw.copyTo(mRefinedTcw.rowRange(0,3).col(3));
        return true;
    }

    return false;
}

/**
 * 根据初始的R,t计算内点
 */
void PnPsolver::CheckInliers()
{
    mnInliersi=0;

    for(int i=0; i<N; i++)
    {
        cv::Point3f P3Dw = mvP3Dw[i];
        cv::Point2f P2D = mvP2D[i];
        //将3D点投影到像素坐标
        float Xc = mRi[0][0]*P3Dw.x+mRi[0][1]*P3Dw.y+mRi[0][2]*P3Dw.z+mti[0];
        float Yc = mRi[1][0]*P3Dw.x+mRi[1][1]*P3Dw.y+mRi[1][2]*P3Dw.z+mti[1];
        float invZc = 1./(mRi[2][0]*P3Dw.x+mRi[2][1]*P3Dw.y+mRi[2][2]*P3Dw.z+mti[2]);
        double ue = uc + fu * Xc * invZc;
        double ve = vc + fv * Yc * invZc;
        //计算误差
        float distX = P2D.x-ue;
        float distY = P2D.y-ve;
        float error2 = distX*distX+distY*distY;
        //判断内点
        if(error2<mvMaxError[i]){
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else{
            mvbInliersi[i]=false;
        }
    }
}


void PnPsolver::set_maximum_number_of_correspondences(int n)
{
    if (max_num_of_corr < n) {
        delete [] pws;
        delete [] us;
        delete [] alphas;
        delete [] pcs;

        max_num_of_corr = n;
        pws = new double[3 * max_num_of_corr];
        us = new double[2 * max_num_of_corr];
        alphas = new double[4 * max_num_of_corr];
        pcs = new double[3 * max_num_of_corr];
    }
}

void PnPsolver::reset_correspondences(void)
{
    num_of_corr = 0;
}

/**
* 增加一组 3D-2D 特征点
* @param X
* @param Y
* @param Z
* @param u
* @param v
*/
void PnPsolver::add_correspondence(double X, double Y, double Z, double u, double v)
{
    pws[3 * num_of_corr    ] = X;
    pws[3 * num_of_corr + 1] = Y;
    pws[3 * num_of_corr + 2] = Z;
    us[2 * num_of_corr    ] = u;
    us[2 * num_of_corr + 1] = v;
    num_of_corr++;
}

/**
* 计算四个控制点
*/
void PnPsolver::choose_control_points(void)
{
    // Take C0 as the reference points centroid:
    //设置第一个控制点为所有点的质心
    cws[0][0] = cws[0][1] = cws[0][2] = 0;
    for(int i = 0; i < num_of_corr; i++){
        for(int j = 0; j < 3; j++)
            cws[0][j] += pws[3 * i + j];
    }
    for(int j = 0; j < 3; j++)
        cws[0][j] /= num_of_corr;

    // Take C1, C2, and C3 from PCA on the reference points:
    //根据PCA计算剩下的3个点
    //首先构建去质心后的点的矩阵
    CvMat * PW0 = cvCreateMat(num_of_corr, 3, CV_64F);
    for(int i = 0; i < num_of_corr; i++){
        for(int j = 0; j < 3; j++)
            PW0->data.db[3 * i + j] = pws[3 * i + j] - cws[0][j];
    }
    //计算A^T * A
    double pw0tpw0[3 * 3];
    CvMat PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0);
    cvMulTransposed(PW0, &PW0tPW0, 1);
    //svd分解
    double dc[3], uct[3 * 3];
    CvMat DC      = cvMat(3, 1, CV_64F, dc);//特征值
    CvMat UCt     = cvMat(3, 3, CV_64F, uct);//特征向量
    cvSVD(&PW0tPW0, &DC, &UCt, nullptr, CV_SVD_MODIFY_A | CV_SVD_U_T);
    cvReleaseMat(&PW0);

    //计算剩余3个控制点
    for(int i = 1; i < 4; i++) {
        double k = std::sqrt(dc[i - 1] / num_of_corr);
        for(int j = 0; j < 3; j++)
            cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];
    }
}

/**
 * 计算用于EPnP算法的barycentric坐标
 */
void PnPsolver::compute_barycentric_coordinates(void)
{
    //计算由控制点的组成的矩阵
    double cc[3 * 3];
    CvMat CC = cvMat(3, 3, CV_64F, cc);
    for(int i = 0; i < 3; i++){
        for(int j = 1; j < 4; j++)
            cc[3 * i + j - 1] = cws[j][i] - cws[0][i];
    }
    //计算逆矩阵
    double cc_inv[3 * 3];
    CvMat CC_inv = cvMat(3, 3, CV_64F, cc_inv);
    cvInvert(&CC, &CC_inv, CV_SVD);
    //计算每一个3D-2D点对应的barycentric坐标
    double * ci = cc_inv;
    for(int i = 0; i < num_of_corr; i++) {
        double * pi = pws + 3 * i;//获取第i个3D点
        double * a = alphas + 4 * i;//获取第i个barycentric
        for(int j = 0; j < 3; j++)
            a[1 + j] =  ci[3 * j    ] * (pi[0] - cws[0][0]) +
                        ci[3 * j + 1] * (pi[1] - cws[0][1]) +
                        ci[3 * j + 2] * (pi[2] - cws[0][2]);
        a[0] = 1.0f - a[1] - a[2] - a[3];
    }
}

/**
 * EPnP算法的某个步骤,构造M矩阵用来求控制点在相机坐标系下的坐标
 * 构造M矩阵的第row列和row+1列
 * @param M 矩阵M
 * @param row 列号
 * @param as barycentric坐标矩阵
 * @param u 像素坐标u
 * @param v 像素坐标v
 */
void PnPsolver::fill_M(CvMat * M, const int row, const double * as, const double u, const double v) const{
    double * M1 = M->data.db + row * 12;//每个3D-2D点对 可以构造两个约束，每个等式有12个参数
    double * M2 = M1 + 12;
    for(int i = 0; i < 4; i++) {//对于每个barycentric坐标 alpha_ij
        M1[3 * i    ] = as[i] * fu;
        M1[3 * i + 1] = 0.0;
        M1[3 * i + 2] = as[i] * (uc - u);

        M2[3 * i    ] = 0.0;
        M2[3 * i + 1] = as[i] * fv;
        M2[3 * i + 2] = as[i] * (vc - v);
    }
}

void PnPsolver::compute_ccs(const double * betas, const double * ut)
{
    for(auto & cc : ccs)
        cc[0] = cc[1] = cc[2] = 0.0f;

    for(int i = 0; i < 4; i++) {
        const double * v = ut + 12 * (11 - i);
        for(int j = 0; j < 4; j++)
            for(int k = 0; k < 3; k++)
                ccs[j][k] += betas[i] * v[3 * j + k];
    }
}

void PnPsolver::compute_pcs(void)
{
    for(int i = 0; i < num_of_corr; i++) {
        double * a = alphas + 4 * i;
        double * pc = pcs + 3 * i;

        for(int j = 0; j < 3; j++)
            pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] + a[3] * ccs[3][j];
    }
}

/**
 * 使用EPnP计算位姿
 * @param R
 * @param t
 * @return
 */
double PnPsolver::compute_pose(double R[3][3], double t[3])
{
    //Step1:获得EPnP算法中的四个控制点（构成质心坐标系）
    choose_control_points();
    //Step2:计算世界坐标系下每个3D点用4个控制点线性表达时的系数alphas
    compute_barycentric_coordinates();
    //Step3:构造M矩阵
    CvMat * M = cvCreateMat(2 * num_of_corr, 12, CV_64F);
    for(int i = 0; i < num_of_corr; i++)
        fill_M(M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1]);
    // Step4：求解Mx = 0
    //计算 M^T * M
    double mtm[12 * 12];
    CvMat MtM = cvMat(12, 12, CV_64F, mtm);
    cvMulTransposed(M, &MtM, 1);
    // 通过（svd分解）求解齐次最小二乘解得到相机坐标系下四个不带尺度的控制点：ut
    // ut的每一行对应一组可能的解
    // 最小特征值对应的特征向量最接近待求的解，由于噪声和约束不足的问题，导致真正的解可能是多个特征向量的线性叠加
    double d[12], ut[12 * 12];
    CvMat D   = cvMat(12,  1, CV_64F, d);
    CvMat Ut  = cvMat(12, 12, CV_64F, ut);
    cvSVD(&MtM, &D, &Ut, nullptr, CV_SVD_MODIFY_A | CV_SVD_U_T);//CV_SVD_MODIFY_A:允许改变矩阵A,  CV_SVD_U_T:返回U转置而不是U
    cvReleaseMat(&M);
    // Step5:上述通过求解齐次最小二乘获得解不具有尺度，这里通过构造另外一个最小二乘（L*Betas = Rho）来求解尺度Betas
    // L_6x10 * Betas10x1 = Rho_6x1
    double l_6x10[6 * 10], rho[6];
    // Betas10        = [B00 B01 B11 B02 B12 B22 B03 B13 B23 B33]
    // |dv00, 2*dv01, dv11, 2*dv02, 2*dv12, dv22, 2*dv03, 2*dv13, 2*dv23, dv33|, 1*10
    // 4个控制点之间总共有6个距离，因此为6*10
    CvMat L_6x10 = cvMat(6, 10, CV_64F, l_6x10);
    compute_L_6x10(ut, l_6x10);
    //计算 Rho_6x1矩阵，即控制点之间两两距离
    CvMat Rho    = cvMat(6,  1, CV_64F, rho);
    compute_rho(rho);

    // 不管什么情况，都假设论文中N=4，并求解部分betas（如果全求解出来会有冲突）
    // 通过优化得到剩下的betas
    double Betas[4][4], rep_errors[4];
    double Rs[4][3][3], ts[4][3];
    // Betas10        = [B00 B01 B11 B02 B12 B22 B03 B13 B23 B33]
    // betas_approx_1 = [B00 B01     B02         B03]
    // 建模为除B11、B12、B13、B14四个参数外其它参数均为0进行最小二乘求解，求出B0、B1、B2、B3粗略解
    find_betas_approx_1(&L_6x10, &Rho, Betas[1]);
    //高斯牛顿法优化B0, B1, B2, B3
    gauss_newton(&L_6x10, &Rho, Betas[1]);
    // 最后计算R t
    rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);

    find_betas_approx_2(&L_6x10, &Rho, Betas[2]);
    gauss_newton(&L_6x10, &Rho, Betas[2]);
    rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);

    find_betas_approx_3(&L_6x10, &Rho, Betas[3]);
    gauss_newton(&L_6x10, &Rho, Betas[3]);
    rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);

    int index = 1;
    if (rep_errors[2] < rep_errors[1]) index = 2;
    if (rep_errors[3] < rep_errors[index]) index = 3;

    //赋值结果
    copy_R_and_t(Rs[index], ts[index], R, t);
    return rep_errors[index];//返回最小误差
}



/**
 * 根据公式构造6x10的L矩阵
 * @param ut ut是对MtM进行SVD分解得到的,其维度是12x12=144,每一行代表一个可能的解（4控制点在相机坐标系下的坐标）
 * @param l_6x10
 */
void PnPsolver::compute_L_6x10(const double * ut, double * l_6x10)
{
    //取出4个最小特征值对应的四组特征向量，作为基。
    const double * v[4];
    v[0] = ut + 12 * 11;
    v[1] = ut + 12 * 10;
    v[2] = ut + 12 *  9;
    v[3] = ut + 12 *  8;
    //根据公式计算dv
    double dv[4][6][3];
    for(int i = 0; i < 4; i++) {
        int a = 0, b = 1;
        for(int j = 0; j < 6; j++) {
            dv[i][j][0] = v[i][3 * a    ] - v[i][3 * b];
            dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
            dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];
            b++;
            if (b > 3) {
                a++;
                b = a + 1;
            }
        }
    }
    //根据公式计算L_6x10
    for(int i = 0; i < 6; i++) {
        double * row = l_6x10 + 10 * i;
        row[0] =        dot(dv[0][i], dv[0][i]);
        row[1] = 2.0f * dot(dv[0][i], dv[1][i]);
        row[2] =        dot(dv[1][i], dv[1][i]);
        row[3] = 2.0f * dot(dv[0][i], dv[2][i]);
        row[4] = 2.0f * dot(dv[1][i], dv[2][i]);
        row[5] =        dot(dv[2][i], dv[2][i]);
        row[6] = 2.0f * dot(dv[0][i], dv[3][i]);
        row[7] = 2.0f * dot(dv[1][i], dv[3][i]);
        row[8] = 2.0f * dot(dv[2][i], dv[3][i]);
        row[9] =        dot(dv[3][i], dv[3][i]);
    }
}

void PnPsolver::copy_R_and_t(const double R_src[3][3], const double t_src[3],
                             double R_dst[3][3], double t_dst[3]){
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++)
            R_dst[i][j] = R_src[i][j];
        t_dst[i] = t_src[i];
    }
}

/**
 * 计算两个3维点的距离的平方
 * @param p1
 * @param p2
 * @return
 */
double PnPsolver::dist2(const double * p1, const double * p2)
{
    return
    (p1[0] - p2[0]) * (p1[0] - p2[0]) +
    (p1[1] - p2[1]) * (p1[1] - p2[1]) +
    (p1[2] - p2[2]) * (p1[2] - p2[2]);
}
/**
 * 计算两个3D向量的点积
 * @param v1
 * @param v2
 * @return
 */
double PnPsolver::dot(const double * v1, const double * v2)
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

double PnPsolver::reprojection_error(const double R[3][3], const double t[3])
{
    double sum2 = 0.0;

    for(int i = 0; i < num_of_corr; i++) {
        double * pw = pws + 3 * i;
        double Xc = dot(R[0], pw) + t[0];
        double Yc = dot(R[1], pw) + t[1];
        double inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);
        double ue = uc + fu * Xc * inv_Zc;
        double ve = vc + fv * Yc * inv_Zc;
        double u = us[2 * i], v = us[2 * i + 1];

        sum2 += sqrt( (u - ue) * (u - ue) + (v - ve) * (v - ve) );
    }

    return sum2 / num_of_corr;
}

void PnPsolver::estimate_R_and_t(double R[3][3], double t[3])
{
    double pc0[3], pw0[3];

    pc0[0] = pc0[1] = pc0[2] = 0.0;
    pw0[0] = pw0[1] = pw0[2] = 0.0;

    for(int i = 0; i < num_of_corr; i++) {
        const double * pc = pcs + 3 * i;
        const double * pw = pws + 3 * i;

        for(int j = 0; j < 3; j++) {
            pc0[j] += pc[j];
            pw0[j] += pw[j];
        }
    }
    for(int j = 0; j < 3; j++) {
        pc0[j] /= num_of_corr;
        pw0[j] /= num_of_corr;
    }

    double abt[3 * 3], abt_d[3], abt_u[3 * 3], abt_v[3 * 3];
    CvMat ABt   = cvMat(3, 3, CV_64F, abt);
    CvMat ABt_D = cvMat(3, 1, CV_64F, abt_d);
    CvMat ABt_U = cvMat(3, 3, CV_64F, abt_u);
    CvMat ABt_V = cvMat(3, 3, CV_64F, abt_v);

    cvSetZero(&ABt);
    for(int i = 0; i < num_of_corr; i++) {
        double * pc = pcs + 3 * i;
        double * pw = pws + 3 * i;

        for(int j = 0; j < 3; j++) {
            abt[3 * j    ] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
            abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
            abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
        }
    }

    cvSVD(&ABt, &ABt_D, &ABt_U, &ABt_V, CV_SVD_MODIFY_A);

    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            R[i][j] = dot(abt_u + 3 * i, abt_v + 3 * j);

        const double det =
                R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] + R[0][2] * R[1][0] * R[2][1] -
                R[0][2] * R[1][1] * R[2][0] - R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

        if (det < 0) {
            R[2][0] = -R[2][0];
            R[2][1] = -R[2][1];
            R[2][2] = -R[2][2];
        }

        t[0] = pc0[0] - dot(R[0], pw0);
        t[1] = pc0[1] - dot(R[1], pw0);
        t[2] = pc0[2] - dot(R[2], pw0);
}

void PnPsolver::print_pose(const double R[3][3], const double t[3])
{
    cout << R[0][0] << " " << R[0][1] << " " << R[0][2] << " " << t[0] << endl;
    cout << R[1][0] << " " << R[1][1] << " " << R[1][2] << " " << t[1] << endl;
    cout << R[2][0] << " " << R[2][1] << " " << R[2][2] << " " << t[2] << endl;
}

void PnPsolver::solve_for_sign(void)
{
    if (pcs[2] < 0.0) {
        for(auto & cc : ccs)
            for(double & j : cc)
                j = -j;

            for(int i = 0; i < num_of_corr; i++) {
                pcs[3 * i    ] = -pcs[3 * i];
                pcs[3 * i + 1] = -pcs[3 * i + 1];
                pcs[3 * i + 2] = -pcs[3 * i + 2];
            }
    }
}

double PnPsolver::compute_R_and_t(const double * ut, const double * betas,
                                  double R[3][3], double t[3]){
    compute_ccs(betas, ut);
    compute_pcs();

    solve_for_sign();

    estimate_R_and_t(R, t);

    return reprojection_error(R, t);
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]

void PnPsolver::find_betas_approx_1(const CvMat * L_6x10, const CvMat * Rho,
                                    double * betas){
    double l_6x4[6 * 4], b4[4];
    CvMat L_6x4 = cvMat(6, 4, CV_64F, l_6x4);
    CvMat B4    = cvMat(4, 1, CV_64F, b4);

    for(int i = 0; i < 6; i++) {
        cvmSet(&L_6x4, i, 0, cvmGet(L_6x10, i, 0));
        cvmSet(&L_6x4, i, 1, cvmGet(L_6x10, i, 1));
        cvmSet(&L_6x4, i, 2, cvmGet(L_6x10, i, 3));
        cvmSet(&L_6x4, i, 3, cvmGet(L_6x10, i, 6));
    }

    cvSolve(&L_6x4, Rho, &B4, CV_SVD);

    if (b4[0] < 0) {
        betas[0] = sqrt(-b4[0]);
        betas[1] = -b4[1] / betas[0];
        betas[2] = -b4[2] / betas[0];
        betas[3] = -b4[3] / betas[0];
    } else {
        betas[0] = sqrt(b4[0]);
        betas[1] = b4[1] / betas[0];
        betas[2] = b4[2] / betas[0];
        betas[3] = b4[3] / betas[0];
    }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

void PnPsolver::find_betas_approx_2(const CvMat * L_6x10, const CvMat * Rho,
                                    double * betas){
    double l_6x3[6 * 3], b3[3];
    CvMat L_6x3  = cvMat(6, 3, CV_64F, l_6x3);
    CvMat B3     = cvMat(3, 1, CV_64F, b3);

    for(int i = 0; i < 6; i++) {
        cvmSet(&L_6x3, i, 0, cvmGet(L_6x10, i, 0));
        cvmSet(&L_6x3, i, 1, cvmGet(L_6x10, i, 1));
        cvmSet(&L_6x3, i, 2, cvmGet(L_6x10, i, 2));
    }

    cvSolve(&L_6x3, Rho, &B3, CV_SVD);

    if (b3[0] < 0) {
        betas[0] = sqrt(-b3[0]);
        betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
    } else {
        betas[0] = sqrt(b3[0]);
        betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
    }

    if (b3[1] < 0) betas[0] = -betas[0];

    betas[2] = 0.0;
    betas[3] = 0.0;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

void PnPsolver::find_betas_approx_3(const CvMat * L_6x10, const CvMat * Rho,
                                    double * betas){
    double l_6x5[6 * 5], b5[5];
    CvMat L_6x5 = cvMat(6, 5, CV_64F, l_6x5);
    CvMat B5    = cvMat(5, 1, CV_64F, b5);

    for(int i = 0; i < 6; i++) {
        cvmSet(&L_6x5, i, 0, cvmGet(L_6x10, i, 0));
        cvmSet(&L_6x5, i, 1, cvmGet(L_6x10, i, 1));
        cvmSet(&L_6x5, i, 2, cvmGet(L_6x10, i, 2));
        cvmSet(&L_6x5, i, 3, cvmGet(L_6x10, i, 3));
        cvmSet(&L_6x5, i, 4, cvmGet(L_6x10, i, 4));
    }

    cvSolve(&L_6x5, Rho, &B5, CV_SVD);

    if (b5[0] < 0) {
        betas[0] = sqrt(-b5[0]);
        betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
    } else {
        betas[0] = sqrt(b5[0]);
        betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
    }
    if (b5[1] < 0) betas[0] = -betas[0];
    betas[2] = b5[3] / betas[0];
    betas[3] = 0.0;
}


void PnPsolver::compute_rho(double * rho)
{
    rho[0] = dist2(cws[0], cws[1]);
    rho[1] = dist2(cws[0], cws[2]);
    rho[2] = dist2(cws[0], cws[3]);
    rho[3] = dist2(cws[1], cws[2]);
    rho[4] = dist2(cws[1], cws[3]);
    rho[5] = dist2(cws[2], cws[3]);
}

void PnPsolver::compute_A_and_b_gauss_newton(const double * l_6x10, const double * rho,
                                             double betas[4], CvMat * A, CvMat * b){
    for(int i = 0; i < 6; i++) {
        const double * rowL = l_6x10 + i * 10;
        double * rowA = A->data.db + i * 4;

        rowA[0] = 2 * rowL[0] * betas[0] +     rowL[1] * betas[1] +     rowL[3] * betas[2] +     rowL[6] * betas[3];
        rowA[1] =     rowL[1] * betas[0] + 2 * rowL[2] * betas[1] +     rowL[4] * betas[2] +     rowL[7] * betas[3];
        rowA[2] =     rowL[3] * betas[0] +     rowL[4] * betas[1] + 2 * rowL[5] * betas[2] +     rowL[8] * betas[3];
        rowA[3] =     rowL[6] * betas[0] +     rowL[7] * betas[1] +     rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

        cvmSet(b, i, 0, rho[i] -
        (
                rowL[0] * betas[0] * betas[0] +
                rowL[1] * betas[0] * betas[1] +
                rowL[2] * betas[1] * betas[1] +
                rowL[3] * betas[0] * betas[2] +
                rowL[4] * betas[1] * betas[2] +
                rowL[5] * betas[2] * betas[2] +
                rowL[6] * betas[0] * betas[3] +
                rowL[7] * betas[1] * betas[3] +
                rowL[8] * betas[2] * betas[3] +
                rowL[9] * betas[3] * betas[3]
                ));
    }
}

/**
 * 使用GN算法优化EPnP的betas参数
 * @param L_6x10
 * @param Rho
 * @param betas
 */
void PnPsolver::gauss_newton(const CvMat * L_6x10, const CvMat * Rho,
                             double betas[4]){
    const int iterations_number = 5;

    double a[6*4], b[6], x[4];
    CvMat A = cvMat(6, 4, CV_64F, a);
    CvMat B = cvMat(6, 1, CV_64F, b);
    CvMat X = cvMat(4, 1, CV_64F, x);

    for(int k = 0; k < iterations_number; k++) {
        compute_A_and_b_gauss_newton(L_6x10->data.db, Rho->data.db,betas, &A, &B);
        qr_solve(&A, &B, &X);

        for(int i = 0; i < 4; i++)
            betas[i] += x[i];
    }
}

void PnPsolver::qr_solve(CvMat * A, CvMat * b, CvMat * X)
{
    static int max_nr = 0;
    static double * A1, * A2;

    const int nr = A->rows;
    const int nc = A->cols;

    if (max_nr != 0 && max_nr < nr) {
        delete [] A1;
        delete [] A2;
    }
    if (max_nr < nr) {
        max_nr = nr;
        A1 = new double[nr];
        A2 = new double[nr];
    }

    double * pA = A->data.db, * ppAkk = pA;
    for(int k = 0; k < nc; k++) {
        double * ppAik = ppAkk, eta = fabs(*ppAik);
        for(int i = k + 1; i < nr; i++) {
            double elt = fabs(*ppAik);
            if (eta < elt) eta = elt;
            ppAik += nc;
        }

        if (eta == 0) {
            A1[k] = A2[k] = 0.0;
            cerr << "God damnit, A is singular, this shouldn't happen." << endl;
            return;
        } else {
            double * ppAik = ppAkk, sum = 0.0, inv_eta = 1. / eta;
            for(int i = k; i < nr; i++) {
                *ppAik *= inv_eta;
                sum += *ppAik * *ppAik;
                ppAik += nc;
            }
            double sigma = sqrt(sum);
            if (*ppAkk < 0)
                sigma = -sigma;
            *ppAkk += sigma;
            A1[k] = sigma * *ppAkk;
            A2[k] = -eta * sigma;
            for(int j = k + 1; j < nc; j++) {
                double * ppAik = ppAkk, sum = 0;
                for(int i = k; i < nr; i++) {
                    sum += *ppAik * ppAik[j - k];
                    ppAik += nc;
                }
                double tau = sum / A1[k];
                ppAik = ppAkk;
                for(int i = k; i < nr; i++) {
                    ppAik[j - k] -= tau * *ppAik;
                    ppAik += nc;
                }
            }
        }
        ppAkk += nc + 1;
    }

    // b <- Qt b
    double * ppAjj = pA, * pb = b->data.db;
    for(int j = 0; j < nc; j++) {
        double * ppAij = ppAjj, tau = 0;
        for(int i = j; i < nr; i++)	{
            tau += *ppAij * pb[i];
            ppAij += nc;
        }
        tau /= A1[j];
        ppAij = ppAjj;
        for(int i = j; i < nr; i++) {
            pb[i] -= tau * *ppAij;
            ppAij += nc;
        }
        ppAjj += nc + 1;
    }

    // X = R-1 b
    double * pX = X->data.db;
    pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
    for(int i = nc - 2; i >= 0; i--) {
        double * ppAij = pA + i * nc + (i + 1), sum = 0;

        for(int j = i + 1; j < nc; j++) {
            sum += *ppAij * pX[j];
            ppAij++;
        }
        pX[i] = (pb[i] - sum) / A2[i];
    }
}



void PnPsolver::relative_error(double & rot_err, double & transl_err,
                               const double Rtrue[3][3], const double ttrue[3],
                               const double Rest[3][3],  const double test[3]){
    double qtrue[4], qest[4];

    mat_to_quat(Rtrue, qtrue);
    mat_to_quat(Rest, qest);

    double rot_err1 = sqrt((qtrue[0] - qest[0]) * (qtrue[0] - qest[0]) +
            (qtrue[1] - qest[1]) * (qtrue[1] - qest[1]) +
            (qtrue[2] - qest[2]) * (qtrue[2] - qest[2]) +
            (qtrue[3] - qest[3]) * (qtrue[3] - qest[3]) ) /
                    sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

    double rot_err2 = sqrt((qtrue[0] + qest[0]) * (qtrue[0] + qest[0]) +
            (qtrue[1] + qest[1]) * (qtrue[1] + qest[1]) +
            (qtrue[2] + qest[2]) * (qtrue[2] + qest[2]) +
            (qtrue[3] + qest[3]) * (qtrue[3] + qest[3]) ) /
                    sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

    rot_err = min(rot_err1, rot_err2);

    transl_err =
            sqrt((ttrue[0] - test[0]) * (ttrue[0] - test[0]) +
            (ttrue[1] - test[1]) * (ttrue[1] - test[1]) +
            (ttrue[2] - test[2]) * (ttrue[2] - test[2])) /
            sqrt(ttrue[0] * ttrue[0] + ttrue[1] * ttrue[1] + ttrue[2] * ttrue[2]);
}

void PnPsolver::mat_to_quat(const double R[3][3], double q[4])
{
    double tr = R[0][0] + R[1][1] + R[2][2];
    double n4;

    if (tr > 0.0f) {
        q[0] = R[1][2] - R[2][1];
        q[1] = R[2][0] - R[0][2];
        q[2] = R[0][1] - R[1][0];
        q[3] = tr + 1.0f;
        n4 = q[3];
    } else if ( (R[0][0] > R[1][1]) && (R[0][0] > R[2][2]) ) {
        q[0] = 1.0f + R[0][0] - R[1][1] - R[2][2];
        q[1] = R[1][0] + R[0][1];
        q[2] = R[2][0] + R[0][2];
        q[3] = R[1][2] - R[2][1];
        n4 = q[0];
    } else if (R[1][1] > R[2][2]) {
        q[0] = R[1][0] + R[0][1];
        q[1] = 1.0f + R[1][1] - R[0][0] - R[2][2];
        q[2] = R[2][1] + R[1][2];
        q[3] = R[2][0] - R[0][2];
        n4 = q[1];
    } else {
        q[0] = R[2][0] + R[0][2];
        q[1] = R[2][1] + R[1][2];
        q[2] = 1.0f + R[2][2] - R[0][0] - R[1][1];
        q[3] = R[0][1] - R[1][0];
        n4 = q[2];
    }
    double scale = 0.5f / double(sqrt(n4));

    q[0] *= scale;
    q[1] *= scale;
    q[2] *= scale;
    q[3] *= scale;
}






/**
* RANSAC求解PnP
* @param nIterations
* @param bNoMore
* @param vbInliers
* @param nInliers
* @return
*/
cv::Mat PnPsolver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers.clear();
    nInliers=0;
    //设置pws,us,alphas,pcs
    set_maximum_number_of_correspondences(mRansacMinSet);
    //
    if(N<mRansacMinInliers){
        bNoMore = true;
        return {};
    }

    vector<size_t> vAvailableIndices;
    int nCurrentIterations = 0;
    while(mnIterations<mRansacMaxIts || nCurrentIterations<nIterations){
        nCurrentIterations++;
        mnIterations++;
        reset_correspondences();
        // 每次迭代将所有的特征匹配复制一份mvAllIndices--->vAvailableIndices，然后选取mRansacMinSet个进行求解
        vAvailableIndices = mvAllIndices;
        // Get min set of points
        //Step1:随机获取四个点，并删之，下次不能再使用,mRansacMinSet默认为4
        for(int i = 0; i < mRansacMinSet; ++i){
            int randi = DUtils::Random::RandomInt(0, (int)vAvailableIndices.size()-1);
            int idx = (int)vAvailableIndices[randi];
            add_correspondence(mvP3Dw[idx].x,mvP3Dw[idx].y,mvP3Dw[idx].z,mvP2D[idx].x,mvP2D[idx].y);
            vAvailableIndices[randi] = vAvailableIndices.back();//将randi处的索引删除：首先将最后一个特征的索引放到randi处，然后删除最后一个
            vAvailableIndices.pop_back();
        }
        // Compute camera pose
        //Step2: 根据这4个点使用EPnP计算mRi,mti-
        compute_pose(mRi, mti);

        //Step3: 根据最初的mRi, mti,计算内点mvbInliers
        CheckInliers();

        //Step4:根据局内点再次使用EPnP计算位姿
        //计算成功则返回mRefinedTcw,否则返回mBestTcw
        if(mnInliersi>=mRansacMinInliers) //如果小于，则返回循环，到Step1
        {
            //Step4.1:计算mBestTcw， If it is the best solution so far, save it
            if(mnInliersi>mnBestInliers){
                mvbBestInliers = mvbInliersi;
                mnBestInliers = mnInliersi;
                cv::Mat Rcw(3,3,CV_64F,mRi);
                cv::Mat tcw(3,1,CV_64F,mti);
                Rcw.convertTo(Rcw,CV_32F);
                tcw.convertTo(tcw,CV_32F);
                mBestTcw = cv::Mat::eye(4,4,CV_32F);
                Rcw.copyTo(mBestTcw.rowRange(0,3).colRange(0,3));
                tcw.copyTo(mBestTcw.rowRange(0,3).col(3));
            }
            //Step4.2: 计算mRefinedTcw,mRefineTcw计算成功则返回
            if(Refine()){
                nInliers = mnRefinedInliers;
                vbInliers = vector<bool>(mvpMapPointMatches.size(),false);
                for(int i=0; i<N; i++){
                    if(mvbRefinedInliers[i])
                        vbInliers[mvKeyPointIndices[i]] = true;
                }
                return mRefinedTcw.clone();
            }

        }
    }

    //Step4.3: Refine()计算失败，则返回最初的mBestTcw
    if(mnIterations>=mRansacMaxIts){
        bNoMore=true;
        if(mnBestInliers>=mRansacMinInliers){
            nInliers=mnBestInliers;
            vbInliers = vector<bool>(mvpMapPointMatches.size(),false);
            for(int i=0; i<N; i++){
                if(mvbBestInliers[i])
                    vbInliers[mvKeyPointIndices[i]] = true;
            }
            return mBestTcw.clone();
        }
    }

    return {};
}

} //namespace ORB_SLAM


