/**
* This file is part of ORB-SLAM2.
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

#include "Initializer.h"
#include "Thirdparty/DBoW2/DUtils/Random.h"
#include<thread>

namespace ORB_SLAM2{ \

/**
 * 构造函数
 * @param ReferenceFrame 参考帧
 * @param sigma 重投影误差的阈值
 * @param iterations RANSAC最大迭代次数
 */
Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    mK = ReferenceFrame.mK.clone();
    mvKeys1 = ReferenceFrame.mvKeysUn;
    mSigma = sigma;//重投影误差的阈值
    mSigma2 = sigma*sigma;
    mMaxIterations = iterations;//最大迭代次数
}

/**
 * @brief 计算基础矩阵和单应性矩阵，选取最佳的来恢复出最开始两帧之间的相对姿态，并进行三角化得到初始地图点
 * Step 1 重新记录特征点对的匹配关系
 * Step 2 在所有匹配特征点对中随机选择8对匹配特征点为一组，用于估计H矩阵和F矩阵
 * Step 3 计算fundamental 矩阵 和homography 矩阵，为了加速分别开了线程计算
 * Step 4 计算得分比例来判断选取哪个模型来求位姿R,t
 *
 * @param[in] CurrentFrame          当前帧，也就是SLAM意义上的第二帧
 * @param[in] vMatches12            当前帧（2）和参考帧（1）图像中特征点的匹配关系
 *                                  vMatches12[i]解释：i表示帧1中关键点的索引值，vMatches12[i]的值为帧2的关键点索引值
 *                                  没有匹配关系的话，vMatches12[i]值为 -1
 * @param[in & out] R21                   相机从参考帧到当前帧的旋转
 * @param[in & out] t21                   相机从参考帧到当前帧的平移
 * @param[in & out] vP3D                  三角化测量之后的三维地图点
 * @param[in & out] vbTriangulated        标记三角化点是否有效，有效为true
 * @return true                     该帧可以成功初始化，返回true
 * @return false                    该帧不满足初始化条件，返回false
 * @return
 */
bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    //Initializer对象创建时就已指定mvKeys1,调用本函数只需指定mvKeys2即可
    mvKeys2 = CurrentFrame.mvKeysUn;
    // mvMatches12记录匹配上的特征点对，记录的是帧2在帧1的匹配索引
    mvMatches12.clear();
    mvMatches12.reserve(mvKeys2.size());
    mvbMatched1.resize(mvKeys1.size());
    /// Step 1 重新记录特征点对的匹配关系存储在mvMatches12，是否有匹配存储在mvbMatched1
    // 将vMatches12（有冗余） 转化为 mvMatches12（只记录了匹配关系）
    for(size_t i=0, iend=vMatches12.size();i<iend; i++){
        if(vMatches12[i]>=0){ //没有匹配关系的话，vMatches12[i]值为 -1
            mvMatches12.emplace_back(i,vMatches12[i]);
            mvbMatched1[i]=true;
        }
        else{
            mvbMatched1[i]=false;
        }
    }

    const int N = (int)mvMatches12.size();
    //设置0~N-1的索引. Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    for(int i=0; i<N; i++){
        vAllIndices.push_back(i);
    }
    //随机选择mMaxIterations * 8个点. Generate sets of 8 points for each RANSAC iteration
    /// Step 2 在所有匹配特征点对中随机选择8对匹配特征点为一组，用于估计H矩阵和F矩阵
    // 共选择 mMaxIterations (默认200) 组
    mvSets = vector<vector<size_t>>(mMaxIterations,vector<size_t>(8,0)); //mvSets用于保存每次迭代时所使用的向量
    DUtils::Random::SeedRandOnce(0);//用于进行随机数据样本采样，设置随机数种子
    vector<size_t> vAvailableIndices;
    for(int it=0; it<mMaxIterations; it++){
        vAvailableIndices = vAllIndices; //迭代开始的时候，所有的点都是可用的
        for(size_t j=0; j<8; j++){ //选择最小的数据样本集，使用八点法求，所以这里就循环了八次
            int randi = DUtils::Random::RandomInt(0,(int)vAvailableIndices.size()-1);// 随机产生一对点的id,范围从0到N-1
            int idx =(int) vAvailableIndices[randi];
            mvSets[it][j] = idx;
            // 由于这对点在本次迭代中已经被使用了,所以我们为了避免再次抽到这个点,就在"点的可选列表"中,
            // 将这个点原来所在的位置用vector最后一个元素的信息覆盖,并且删除尾部的元素
            // 这样就相当于将这个点的信息从"点的可用列表"中直接删除了
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }
    /// step3. 计算F矩阵和H矩阵及其置信程度,启动两个线程
    vector<bool> vbMatchesInliersH, vbMatchesInliersF; //这两个变量用于标记在H和F的计算中哪些特征点对被认为是Inlier
    float SH, SF;//计算出来的单应矩阵和基础矩阵的RANSAC评分，这里其实是采用重投影误差来计算的
    cv::Mat H, F;//这两个是经过RANSAC算法后计算出来的单应矩阵和基础矩阵
    // thread方法比较特殊，在传递引用的时候，外层需要用ref来进行引用传递，否则就是浅拷贝
    thread threadH(&Initializer::FindHomography,this,ref(vbMatchesInliersH), ref(SH), ref(H));
    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));
    threadH.join();
    threadF.join();
    ///step4. 根据比分计算使用哪个矩阵恢复运动
    float RH = SH/(SH+SF);
    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    if(RH>0.40)
        //更偏向于平面，此时从单应矩阵恢复
        return ReconstructH(vbMatchesInliersH,H,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
    else //if(pF_HF>0.6)
        // 更偏向于非平面，从基础矩阵恢复
        return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
}

/**
 * @brief 计算单应矩阵，假设场景为平面情况下通过前两帧求取Homography矩阵，并得到该模型的评分
 * 原理参考Multiple view geometry in computer vision  P109 算法4.4
 * Step 1 将当前帧和参考帧中的特征点坐标进行归一化
 * Step 2 选择8个归一化之后的点对进行迭代
 * Step 3 八点法计算单应矩阵矩阵
 * Step 4 利用重投影误差为当次RANSAC的结果评分
 * Step 5 更新具有最优评分的单应矩阵计算结果,并且保存所对应的特征点对的内点标记
 *
 * @param[in & out] vbMatchesInliers          标记是否是外点
 * @param[in & out] score                     计算单应矩阵的得分
 * @param[in & out] H21                       单应矩阵结果
 */
void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    const int N = (int)mvMatches12.size();//匹配的特征点对总数
    /// Step 1 将当前帧和参考帧中的特征点坐标进行归一化，主要是平移和尺度变换
    // 具体来说,就是将mvKeys1和mvKey2归一化到均值为0，一阶绝对矩为1，归一化矩阵分别为T1、T2
    // 这里所谓的一阶绝对矩其实就是随机变量到取值的中心的绝对值的平均值
    // 归一化矩阵就是把上述归一化的操作用矩阵来表示。这样特征点坐标乘归一化矩阵可以得到归一化后的坐标
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2inv = T2.inv();

    score = 0.0;// 记录最佳评分
    vbMatchesInliers = vector<bool>(N,false); // 取得历史最佳评分时,特征点对的inliers标记
    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat H21i, H12i;
    // 每次RANSAC记录Inliers与得分
    vector<bool> vbCurrentInliers(N,false);

    //RANSAC循环
    for(int it=0; it<mMaxIterations; it++){
        /// Step 2 选择8个归一化之后的点对进行迭代
        for(size_t j=0; j<8; j++){
            int idx = (int)mvSets[it][j];//从mvSets中获取当前次迭代的某个特征点对的索引信息
            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }
        /// Step 3 八点法计算单应矩阵
        // 关于为什么计算之前要对特征点进行归一化，后面又恢复这个矩阵的尺度？
        // 可以在《计算机视觉中的多视图几何》这本书中P193页中找到答案
        // 书中这里说,8点算法成功的关键是在构造解的方称之前应对输入的数据认真进行适当的归一化
        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
        //恢复H的尺度. 单应矩阵原理：X2=H21*X1，其中X1,X2 为归一化后的特征点
        // 特征点归一化：vPn1 = T1 * mvKeys1, vPn2 = T2 * mvKeys2  得到:T2 * mvKeys2 =  Hn * T1 * mvKeys1
        // 进一步得到:mvKeys2  = T2.inv * Hn * T1 * mvKeys1
        H21i = T2inv*Hn*T1;
        //单应矩阵的逆
        H12i = H21i.inv();
        ///Step 4 卡方检验,利用重投影误差为当次RANSAC的结果评分
        float currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);
        /// Step 5 更新具有最优评分的单应矩阵计算结果,并且保存所对应的特征点对的内点标记
        if(currentScore>score){
            H21 = H21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

/**
 * @brief 计算基础矩阵，假设场景为非平面情况下通过前两帧求取Fundamental矩阵，得到该模型的评分
 * Step 1 将当前帧和参考帧中的特征点坐标进行归一化
 * Step 2 选择8个归一化之后的点对进行迭代
 * Step 3 八点法计算基础矩阵矩阵
 * Step 4 利用重投影误差为当次RANSAC的结果评分
 * Step 5 更新具有最优评分的基础矩阵计算结果,并且保存所对应的特征点对的内点标记
 *
 * @param[in & out] vbMatchesInliers          标记是否是外点
 * @param[in & out] score                     计算基础矩阵得分
 * @param[in & out] F21                       从特征点1到2的基础矩阵
 */
void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    const int N = (int)vbMatchesInliers.size();
    /// step1. 特征点归一化,归一化可以增强计算稳定性.
    cv::Mat T1, T2;
    vector<cv::Point2f> vPn1, vPn2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2t = T2.t();//用于恢复尺度
    /** T2的形式为:
     T =[   sX  0   -mean_x * sX
            0   sY  -mean_y * sY
            0   0   1           ]
     其中mean_x是坐标x的均值, sX= 1/dev_x表示x的方差的倒数
     */
    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);
    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    /// step2. RANSAC循环
    for(int it=0; it<mMaxIterations; it++){
        // Select a minimum set
        for(int j=0; j<8; j++){
            int idx =(int) mvSets[it][j];
            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }
        /// step3. 八点法计算基础矩阵F
        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);
        /// step4. 恢复原始尺度
        F21i = T2t*Fn*T1;
        /// step5. 根据重投影误差进行卡方检验
        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);
        /// step6. 记录最优解
        if(currentScore>score){
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

/**
 * @brief 用DLT方法求解单应矩阵H
 * 这里最少用4对点就能够求出来，不过这里为了统一还是使用了8对点求最小二乘解
 *
 * @param[in] vP1               参考帧中归一化后的特征点
 * @param[in] vP2               当前帧中归一化后的特征点
 * @return cv::Mat              计算的单应矩阵H
 */
cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    // 基本原理：见附件推导过程：
    // |x'|     | h1 h2 h3 ||x|
    // |y'| = a | h4 h5 h6 ||y|  简写: x' = a H x, a为一个尺度因子
    // |1 |     | h7 h8 h9 ||1|
    // 使用DLT(direct linear tranform)求解该模型
    // x' = a H x
    // ---> (x') 叉乘 (H x)  = 0  (因为方向相同) (取前两行就可以推导出下面的了)
    // ---> Ah = 0
    // A = | 0  0  0 -x -y -1 xy' yy' y'|  h = | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
    //     |-x -y -1  0  0  0 xx' yx' x'|
    // 通过SVD求解Ah = 0，A^T*A最小特征值对应的特征向量即为解
    // 其实也就是右奇异值矩阵的最后一列

    ///构造A矩阵,求 Ax=b
    const int N = (int)vP1.size();
    cv::Mat A(2*N,9,CV_32F);
    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;
    }
    ///使用SVD分解求最小二乘解
    cv::Mat u,w,vt;
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A //MODIFY_A是指允许计算函数可以修改待分解的矩阵，官方文档上说这样可以加快计算速度、节省内存
                                    | cv::SVD::FULL_UV);//FULL_UV的意思是把U和VT补充成单位正交方阵
    // 返回最小奇异值所对应的右奇异向量
    // 注意前面说的是右奇异值矩阵的最后一列，但是在这里因为是vt，转置后了，所以是行；由于A有9列数据，故最后一列的下标为8
    return vt.row(8).reshape(0, 3);
}

/**
 * @brief 根据特征点匹配求fundamental matrix（normalized 8点法）
 * 注意F矩阵有秩为2的约束，所以需要两次SVD分解
 *
 * @param[in] vP1           参考帧中归一化后的特征点
 * @param[in] vP2           当前帧中归一化后的特征点
 * @return cv::Mat          最后计算得到的基础矩阵F
 */
cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    // x'Fx = 0 整理可得：Af = 0
    // A = | x'x x'y x' y'x y'y y' x y 1 |, f = | f1 f2 f3 f4 f5 f6 f7 f8 f9 |
    // 通过SVD求解Af = 0，A'A最小特征值对应的特征向量即为解
    ///构造Ax=0的A矩阵
    const int N = (int)vP1.size();
    cv::Mat A(N,9,CV_32F);
    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;
        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }
    ///进行SVD分解,取最小二乘解
    cv::Mat u,w,vt;
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    //去V^T矩阵的最后一行作为解
    cv::Mat Fpre = vt.row(8).reshape(0, 3);
    ///对Fpre进行SVD分解,将F投影到F所在的流形上,即使其奇异值变为sigma,sigma,0的形式
    //基础矩阵的秩为2,而我们不敢保证计算得到的这个结果的秩为2,所以需要通过第二次奇异值分解,来强制使其秩为2
    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    w.at<float>(2)=0;// 秩2约束，强制将第3个奇异值设置为0
    // 重新组合好满足秩约束的基础矩阵，作为最终计算结果返回
    return  u*cv::Mat::diag(w)*vt;
}

/**
 * @brief 对给定的homography matrix打分,需要使用到卡方检验的知识
 *
 * @param[in] H21                       从参考帧到当前帧的单应矩阵
 * @param[in] H12                       从当前帧到参考帧的单应矩阵
 * @param[in] vbMatchesInliers          匹配好的特征点对的Inliers标记
 * @param[in] sigma                     方差，默认为1
 * @return float                        返回得分
 */
float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{
    // 说明：在已值n维观测数据误差服从N(0，sigma）的高斯分布时
    // 其误差加权最小二乘结果为  sum_error = SUM(e(i)^T * Q^(-1) * e(i))
    // 其中：e(i) = [e_x,e_y,...]^T, Q维观测数据协方差矩阵，即sigma * sigma组成的协方差矩阵
    // 误差加权最小二次结果越小，说明观测数据精度越高
    // 那么，score = SUM((th - e(i)^T * Q^(-1) * e(i)))的分数就越高
    // 算法目标： 检查单应变换矩阵
    // 检查方式：通过H矩阵，进行参考帧和当前帧之间的双向投影，并计算起加权最小二乘投影误差

    // 算法流程
    // input: 单应性矩阵 H21, H12, 匹配点集 mvKeys1
    //    do:
    //        for p1(i), p2(i) in mvKeys:
    //           error_i1 = ||p2(i) - H21 * p1(i)||2
    //           error_i2 = ||p1(i) - H12 * p2(i)||2
    //
    //           w1 = 1 / sigma / sigma
    //           w2 = 1 / sigma / sigma
    //
    //           if error1 < th
    //              score +=   th - error_i1 * w1
    //           if error2 < th
    //              score +=   th - error_i2 * w2
    //
    //           if error_1i > th or error_2i > th
    //              p1(i), p2(i) are inner points
    //              vbMatchesInliers(i) = true
    //           else
    //              p1(i), p2(i) are outliers
    //              vbMatchesInliers(i) = false
    //           end
    //        end
    //   output: score, inliers

    const int N = (int)mvMatches12.size();
    /// Step 1 获取从参考帧到当前帧的单应矩阵的各个元素
    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);
    // 获取从当前帧到参考帧的单应矩阵的各个元素
    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    vbMatchesInliers.resize(N);
    float score = 0;
    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
    const float th = 5.991;
    const float invSigmaSquare = 1.f/(sigma*sigma); //信息矩阵，方差平方的倒数
    /// Step 2 通过H矩阵，进行参考帧和当前帧之间的双向投影，并计算起加权重投影误差
    // H21 表示从img1 到 img2的变换矩阵
    // H12 表示从img2 到 img1的变换矩阵
    for(int i=0; i<N; i++)
    {
        bool bIn = true;//默认是内点
        /// Step 2.1 提取参考帧和当前帧之间的特征匹配点对
        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];
        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;
        /// Step 2.2 计算 img2 到 img1 的重投影误差
        // x1 = H12*x2
        // 将图像2中的特征点通过单应变换投影到图像1中
        // |u1|   |h11inv h12inv h13inv||u2|   |u2in1|
        // |v1| = |h21inv h22inv h23inv||v2| = |v2in1| * w2in1inv
        // |1 |   |h31inv h32inv h33inv||1 |   |  1  |
        // 计算投影归一化坐标
        const float w2in1inv = 1.f/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;
        // 计算重投影误差 = ||p1(i) - H12 * p2(i)||2
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);
        const float chiSquare1 = squareDist1*invSigmaSquare;

        /// Step 2.3 用阈值标记外点，内点的话累加得分
        if(chiSquare1>th)
            bIn = false;
        else
            score += th - chiSquare1;// 误差越大，得分越低

        // 计算从img1 到 img2 的投影变换误差
        // x1in2 = H21*x1
        // 将图像2中的特征点通过单应变换投影到图像1中
        // |u2|   |h11 h12 h13||u1|   |u1in2|
        // |v2| = |h21 h22 h23||v1| = |v1in2| * w1in2inv
        // |1 |   |h31 h32 h33||1 |   |  1  |
        // 计算投影归一化坐标
        const float w1in2inv = 1.f/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;
        // 计算重投影误差
        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);
        const float chiSquare2 = squareDist2*invSigmaSquare;
        // 用阈值标记离群点，内点的话累加得分
        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;
        /// Step 2.4 如果从img2 到 img1 和 从img1 到img2的重投影误差均满足要求，则说明是内点
        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

/**
 * 卡方检验计算置信度得分,来判断F的质量
 * 卡方检验通过构造检验统计量 χ2来比较期望结果和实际结果之间的差别,从而得出观察频数极值的发生概率.
 * @param F21
 * @param vbMatchesInliers
 * @param sigma
 * @return
 */
float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    // 说明：在已值n维观测数据误差服从N(0，sigma）的高斯分布时
    // 其误差加权最小二乘结果为  sum_error = SUM(e(i)^T * Q^(-1) * e(i))
    // 其中：e(i) = [e_x,e_y,...]^T, Q维观测数据协方差矩阵，即sigma * sigma组成的协方差矩阵
    // 误差加权最小二次结果越小，说明观测数据精度越高
    // 那么，score = SUM((th - e(i)^T * Q^(-1) * e(i)))的分数就越高
    // 算法目标：检查基础矩阵
    // 检查方式：利用对极几何原理 p2^T * F * p1 = 0
    // 假设：三维空间中的点 P 在 img1 和 img2 两图像上的投影分别为 p1 和 p2（两个为同名点）
    //   则：p2 一定存在于极线 l2 上，即 p2*l2 = 0. 而l2 = F*p1 = (a, b, c)^T
    //      所以，这里的误差项 e 为 p2 到 极线 l2 的距离，如果在直线上，则 e = 0
    //      根据点到直线的距离公式：d = (ax + by + c) / sqrt(a * a + b * b)
    //      所以，e =  (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)

    // 算法流程
    // input: 基础矩阵 F 左右视图匹配点集 mvKeys1
    //    do:
    //        for p1(i), p2(i) in mvKeys:
    //           l2 = F * p1(i)
    //           l1 = p2(i) * F
    //           error_i1 = dist_point_to_line(x2,l2)
    //           error_i2 = dist_point_to_line(x1,l1)
    //
    //           w1 = 1 / sigma / sigma
    //           w2 = 1 / sigma / sigma
    //
    //           if error1 < th
    //              score +=   thScore - error_i1 * w1
    //           if error2 < th
    //              score +=   thScore - error_i2 * w2
    //
    //           if error_1i > th or error_2i > th
    //              p1(i), p2(i) are inner points
    //              vbMatchesInliers(i) = true
    //           else
    //              p1(i), p2(i) are outliers
    //              vbMatchesInliers(i) = false
    //           end
    //        end
    //   output: score, inliers
    const int N = (int)mvMatches12.size();
    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;
    // 基于卡方检验计算出的阈值
    // 自由度为1的卡方分布，显著性水平为0.05，对应的临界阈值
    // ?是因为点到直线距离是一个自由度吗？
    const float th = 3.841;
    // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
    const float thScore = 5.991;
    // 信息矩阵，或 协方差矩阵的逆矩阵
    const float invSigmaSquare = 1.f/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;
        // Step 2.1 提取参考帧和当前帧之间的特征匹配点对
        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];
        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        /// Step 2.2 计算 img1 上的点在 img2 上投影得到的极线 l2 = F21 * p1 = (a2,b2,c2)
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;
        /// Step 2.3 计算误差 e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
        const float num2 = a2*u2+b2*v2+c2;
        const float squareDist1 = num2*num2/(a2*a2+b2*b2);
        const float chiSquare1 = squareDist1*invSigmaSquare;
        /// Step 2.4 误差大于阈值就说明这个点是Outlier
        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        /// 计算img2上的点在 img1 上投影得到的极线 l1= p2 * F21 = (a1,b1,c1)
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;
        // 计算误差 e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
        const float num1 = a1*u1+b1*v1+c1;
        const float squareDist2 = num1*num1/(a1*a1+b1*b1);
        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}


/**
 * @brief 从基础矩阵F中求解位姿R，t及三维点
 * F分解出E，E有四组解，选择计算的有效三维点（在摄像头前方、投影误差小于阈值、视差角大于阈值）最多的作为最优的解
 * @param[in] vbMatchesInliers          匹配好的特征点对的Inliers标记
 * @param[in] F21                       从参考帧到当前帧的基础矩阵
 * @param[in] K                         相机的内参数矩阵
 * @param[in & out] R21                 计算好的相机从参考帧到当前帧的旋转
 * @param[in & out] t21                 计算好的相机从参考帧到当前帧的平移
 * @param[in & out] vP3D                三角化测量之后的特征点的空间坐标
 * @param[in & out] vbTriangulated      特征点三角化成功的标志
 * @param[in] minParallax               认为三角化有效的最小视差角
 * @param[in] minTriangulated           最小三角化点数量
 * @return true                         成功初始化
 * @return false                        初始化失败
 */
bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D,
                            vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    /// Step 1 统计有效匹配点个数，并用 N 表示
    int N=0;
    for(auto && inliner : vbMatchesInliers)
        if(inliner)
            N++;
    /// Step 2 根据基础矩阵和相机的内参数矩阵计算本质矩阵
    cv::Mat E21 = K.t()*F21*K;

    // 定义本质矩阵分解结果，形成四组解,分别是：(R1, t) (R1, -t) (R2, t) (R2, -t)
    cv::Mat R1, R2, t;
    // Step 3 从本质矩阵求解两个R解和两个t解，共四组解,不过由于两个t解互为相反数，因此这里先只获取一个
    // 虽然这个函数对t有归一化，但并没有决定单目整个SLAM过程的尺度.因为 CreateInitialMapMonocular 函数对3D点深度会缩放，然后反过来对 t 有改变.
    //注意下文中的符号“'”表示矩阵的转置
    //                          |0 -1  0|
    // E = U Sigma V'   let W = |1  0  0|
    //                          |0  0  1|
    // 得到4个解 E = [R|t]
    // R1 = UWV' R2 = UW'V' t1 = U3 t2 = -U3
    DecomposeE(E21,R1,R2,t);  

    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;

    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0f*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0f*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0f*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0f*mSigma2, vbTriangulated4, parallax4);

    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if(maxGood==nGood1)
    {
        if(parallax1>minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}

bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D,
                      vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(auto && vbMatchesInlier : vbMatchesInliers)
        if(vbMatchesInlier)
            N++;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K;

    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    V=Vt.t();

    float s = float(cv::determinant(U)*cv::determinant(Vt));

    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }


    int bestGood = 0;
    int secondBestGood = 0;    
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    for(size_t i=0; i<8; i++)
    {
        float parallaxi;
        vector<cv::Point3f> vP3Di;
        vector<bool> vbTriangulatedi;
        int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.f*mSigma2, vbTriangulatedi, parallaxi);

        if(nGood>bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;
        }
    }


    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;

        return true;
    }

    return false;
}

void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

/**
 * 标准化特征点
 * @param vKeys
 * @param vNormalizedPoints
 * @param T 包含了标准化的尺度信息
 */
void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = (int)vKeys.size();
    vNormalizedPoints.resize(N);
    ///计算均值点
    for(int i=0; i<N; i++){
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }
    meanX = meanX/(float)N;
    meanY = meanY/(float)N;
    ///使数据均值为0,同时计算方差
    float meanDevX = 0;
    float meanDevY = 0;
    for(int i=0; i<N; i++){
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;//去中心
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;
        meanDevX += std::fabs(vNormalizedPoints[i].x);//方差
        meanDevY += std::fabs(vNormalizedPoints[i].y);
    }
    meanDevX = meanDevX/(float)N;
    meanDevY = meanDevY/(float)N;
    float sX = 1.0f/meanDevX;
    float sY = 1.0f/meanDevY;
    ///使得方差为1
    for(int i=0; i<N; i++){
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}


int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    vbGood = vector<bool>(vKeys1.size(),false);
    vP3D.resize(vKeys1.size());

    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));

    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    cv::Mat O2 = -R.t()*t;

    int nGood=0;

    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        if(!vbMatchesInliers[i])
            continue;

        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1;

        Triangulate(kp1,kp2,P1,P2,p3dC1);

        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = (float)cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = (float)cv::norm(normal2);

        float cosParallax = (float)normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;

        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.f/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.f/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

        if(squareError2>th2)
            continue;

        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        nGood++;

        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }

    if(nGood>0)
    {
        std::sort(vCosParallax.begin(),vCosParallax.end());
        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = std::acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}

/**
 * @brief 分解Essential矩阵得到R,t
 * 分解E矩阵将得到4组解，这4组解分别为[R1,t],[R1,-t],[R2,t],[R2,-t]
 * 参考：Multiple View Geometry in Computer Vision - Result 9.19 p259
 * @param[in] E                 本质矩阵
 * @param[in & out] R1          旋转矩阵1
 * @param[in & out] R2          旋转矩阵2
 * @param[in & out] t           平移向量，另外一个取相反数
 */
void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    //对本质矩阵进行奇异值分解
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);
    // 左奇异值矩阵U的最后一列就是t，对其进行归一化
    u.col(2).copyTo(t);
    t=t/cv::norm(t);
    // 构造一个绕Z轴旋转pi/2的旋转矩阵W，按照下式组合得到旋转矩阵 R1 = u*W*vt
    //计算完成后要检查一下旋转矩阵行列式的数值，使其满足行列式为1的约束
    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0) //旋转矩阵有行列式为+1的约束，所以如果算出来为负值，需要取反
        R1=-R1;
    // 同理将矩阵W取转置来按照相同的公式计算旋转矩阵R2 = u*W.t()*vt
    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}

} //namespace ORB_SLAM
