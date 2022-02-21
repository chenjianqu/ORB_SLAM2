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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include<mutex>

namespace ORB_SLAM2{ \


/**
 * @brief 全局BA： pMap中所有的MapPoints和关键帧做bundle adjustment优化
 * 这个全局BA优化在本程序中有两个地方使用：
 * 1、单目初始化：CreateInitialMapMonocular函数
 * 2、闭环优化：RunGlobalBundleAdjustment函数
 * @param[in] pMap                  地图点
 * @param[in] nIterations           迭代次数
 * @param[in] pbStopFlag            外部控制BA结束标志
 * @param[in] nLoopKF               形成了闭环的当前关键帧的id
 * @param[in] bRobust               是否使用鲁棒核函数
 */
void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    // 获取地图中的所有关键帧
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    // 获取地图中的所有地图点
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    // 调用GBA
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}

/**
 * @brief bundle adjustment 优化过程
 * 1. Vertex: g2o::VertexSE3Expmap()，即当前帧的Tcw
 *            g2o::VertexSBAPointXYZ()，MapPoint的mWorldPos
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + Vertex：待优化MapPoint的mWorldPos
 *         + measurement：MapPoint在当前帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *
 * @param[in] vpKFs                 参与BA的所有关键帧
 * @param[in] vpMP                  参与BA的所有地图点
 * @param[in] nIterations           优化迭代次数
 * @param[in] pbStopFlag            外部控制BA结束标志
 * @param[in] nLoopKF               形成了闭环的当前关键帧的id
 * @param[in] bRobust               是否使用核函数
 */
void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;// 不参与优化的地图点
    vbNotIncludedMP.resize(vpMP.size());
    /// Step 1 初始化g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    auto * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    // 使用LM算法优化
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    // 如果这个时候外部请求终止，那就结束. 注意这句执行之后，外部再请求结束BA，就结束不了了
    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;// 记录添加到优化器中的顶点的最大关键帧id

    // Set KeyFrame vertices
    /// Step 2.1 ：向优化器添加关键帧位姿顶点
    for(auto pKF : vpKFs) {
        if(pKF->isBad())
            continue;
        auto* vSE3 = new g2o::VertexSE3Expmap();// 对于每一个能用的关键帧构造SE3顶点,其实就是当前关键帧的位姿
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId((int)pKF->mnId);
        vSE3->setFixed(pKF->mnId==0); // 只有第0帧关键帧不优化（参考基准）
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }
    // 卡方分布 95% 以上可信度的时候的阈值
    const auto thHuber2D = (float)sqrt(5.99);
    const auto thHuber3D = (float)sqrt(7.815);

    // Set MapPoint vertices
    /// Step 2.2：向优化器添加地图点作为顶点
    for(size_t i=0; i<vpMP.size(); i++){
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        auto* vPoint = new g2o::VertexSBAPointXYZ();
        // 注意由于地图点的位置是使用cv::Mat数据类型表示的,这里需要转换成为Eigen::Vector3d类型
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = int(pMP->mnId+maxKFid+1); // 前面记录maxKFid 是在这里使用的
        vPoint->setId(id);
        // 注意g2o在做BA的优化时必须将其所有地图点全部schur掉，否则会出错。
        // 原因是使用了g2o::LinearSolver<BalBlockSolver::PoseMatrixType>这个类型来指定linearsolver,
        // 其中模板参数当中的位姿矩阵类型在程序中为相机姿态参数的维度，于是BA当中schur消元后解得线性方程组必须是只含有相机姿态变量。
        // Ceres库则没有这样的限制
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        // 取出地图点和关键帧之间观测的关系
       const map<KeyFrame*,size_t> observations = pMP->GetObservations();

       int nEdges = 0;// 边计数
        //SET EDGES
        /// Step 3：向优化器添加投影边（是在遍历地图点、添加地图点的顶点的时候顺便添加的）
        for(auto observation : observations){
            KeyFrame* pKF = observation.first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;
            nEdges++;
            // 取出该地图点对应该关键帧的2D特征点
            const cv::KeyPoint &kpUn = pKF->mvKeysUn[observation.second];
            /// 以下是单目相机模式：
            if(pKF->mvuRight[observation.second]<0)
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;
                // 创建边
                auto* e = new g2o::EdgeSE3ProjectXYZ();
                // 边连接的第0号顶点对应的是第id个地图点
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                // 边连接的第1号顶点对应的是第id个关键帧
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((int)pKF->mnId)));
                // 设置观测值
                e->setMeasurement(obs);
                // 信息矩阵，也是协方差，表明了这个约束的观测在各个维度（x,y）上的可信程度，在我们这里对于具体的一个点，两个坐标的可信程度都是相同的，
                // 其可信程度受到特征点在图像金字塔中的图层有关，图层越高，可信度越差
                // 为了避免出现信息矩阵中元素为负数的情况，这里使用的是sigma^(-2)
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                // 使用鲁棒核函数
                if(bRobust){
                    auto* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    // 这里的重投影误差，自由度为2，所以这里设置为卡方分布中自由度为2的阈值，如果重投影的误差大约大于1个像素的时候，就认为不太靠谱的点了，
                    // 核函数是为了避免其误差的平方项出现数值上过大的增长
                    rk->setDelta(thHuber2D);
                }
                // 设置相机内参
                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                // 添加边
                optimizer.addEdge(e);
            }
            /// 双目或RGBD相机
            else
            {
                // 双目相机的观测数据则是由三个部分组成：投影点的x坐标，投影点的y坐标，以及投影点在右目中的x坐标（默认y方向上已经对齐了）
                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[observation.second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                auto* e = new g2o::EdgeStereoSE3ProjectXYZ();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((int)pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);
                if(bRobust){
                    auto* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }
        // 如果因为一些特殊原因,实际上并没有任何关键帧观测到当前的这个地图点,那么就删除掉这个顶点,并且这个地图点也就不参与优化
        if(nEdges==0){
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else{
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    /// Step 4：开始优化
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data
    /// Step 5：得到优化的结果

    //Keyframes
    /// Step 5.1 遍历所有的关键帧
    for(auto pKF : vpKFs) {
        if(pKF->isBad())
            continue;
        auto* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex((int)pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();// 获取到优化后的位姿
        if(nLoopKF==0){
            // 原则上来讲不会出现"当前闭环关键帧是第0帧"的情况,如果这种情况出现,只能够说明是在创建初始地图点的时候调用的这个全局BA函数.
            // 这个时候,地图中就只有两个关键帧,其中优化后的位姿数据可以直接写入到帧的成员变量中
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else{
            // 正常的操作,先把优化后的位姿写入到帧的一个专门的成员变量mTcwGBA中备用
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    /// Step 5.2 Points
    for(size_t i=0; i<vpMP.size(); i++){
        if(vbNotIncludedMP[i])
            continue;
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        // 获取优化之后的地图点的位置
        auto* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        if(nLoopKF==0){
            // 如果这个GBA是在创建初始地图的时候调用的话,那么地图点的位姿也可以直接写入
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else{
            //如果是正常的闭环过程调用,就先临时保存一下
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

}


/**
 * @brief Pose Only Optimization
 *
 * 3D-2D 最小化重投影误差 e = (u,v) - project(Tcw*Pw) \n
 * 只优化Frame的Tcw，不优化MapPoints的坐标
 *
 * 1. Vertex: g2o::VertexSE3Expmap()，即当前帧的Tcw
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZOnlyPose()，BaseUnaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + measurement：MapPoint在当前帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *     - g2o::EdgeStereoSE3ProjectXYZOnlyPose()，BaseUnaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + measurement：MapPoint在当前帧中的二维位置(ul,v,ur)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *
 * @param   pFrame Frame
 * @return  inliers数量
 */
int Optimizer::PoseOptimization(Frame *pFrame)
{
    // 该优化函数主要用于Tracking线程中：运动跟踪、参考帧跟踪、地图跟踪、重定位
    /// Step 1：构造g2o优化器, BlockSolver_6_3表示：位姿 _PoseDim 为6维，路标点 _LandmarkDim 是3维
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    auto * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);//使用LM算法
    optimizer.setAlgorithm(solver);
    // 输入的帧中,有效的,参与优化过程的2D-3D点对
    int nInitialCorrespondences=0;
    /// Step 2：添加顶点：待优化当前帧的Tcw
    auto * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);// 要优化的变量，所以不能固定
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;//单目观测的边
    vector<size_t> vnIndexEdgeMono;//单目边的索引
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const auto deltaMono = (float)sqrt(5.991);// 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
    const auto deltaStereo = (float)sqrt(7.815);// 自由度为3的卡方分布，显著性水平为0.05，对应的临界阈值7.815

    /// Step 3：添加一元边
    {
    // 锁定地图点。由于需要使用地图点来构造顶点和边,因此不希望在构造的过程中部分地图点被改写造成不一致甚至是段错误
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);
    // 遍历当前地图中的所有地图点
    for(int i=0; i<N; i++){
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP){
            /// 单目情况
            if(pFrame->mvuRight[i]<0){
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;
                // 对这个地图点的观测
                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;
                // 新建单目的边，一元边，误差为观测特征点坐标减去投影点的坐标
                auto* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                // 设置边的顶点
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                //设置边的观测
                e->setMeasurement(obs);
                // 这个点的可信程度和特征点所在的图层有关
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                // 在这里使用了鲁棒核函数
                auto* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);
                // 设置相机内参
                e->fx = Frame::fx;
                e->fy = Frame::fy;
                e->cx = Frame::cx;
                e->cy = Frame::cy;
                // 地图点的空间位置,作为迭代的初始值
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            ///双目
            else {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;
                //SET EDGE
                // 观测多了一项右目的坐标
                Eigen::Matrix<double,3,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                const float &kp_ur = pFrame->mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
                // 新建边，一元边，误差为观测特征点坐标减去投影点的坐标
                auto* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                auto* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = ORB_SLAM2::Frame::fx;
                e->fy = ORB_SLAM2::Frame::fy;
                e->cx = ORB_SLAM2::Frame::cx;
                e->cy = ORB_SLAM2::Frame::cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }

    }
    }

    // 如果没有足够的匹配点,那么就只好放弃了
    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    /// Step 4：开始优化，总共优化四次，每次优化迭代10次,每次优化后，将观测分为outlier和inlier，outlier不参与下次优化
    // 由于每次优化后是对所有的观测进行outlier和inlier判别，因此之前被判别为outlier有可能变成inlier，反之亦然

    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};   // 四次迭代，每次迭代的次数

    int nBad=0;// bad 的地图点个数
    for(size_t it=0; it<4; it++){
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        // 其实就是初始化优化器,这里的参数0就算是不填写,默认也是0,也就是只对level为0的边进行优化
        optimizer.initializeOptimization(0);
        /// 开始优化，优化10次
        optimizer.optimize(its[it]);

        /// 优化结束,开始遍历参与优化的每一条误差边(单目)
        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++){
            auto* e = vpEdgesMono[i];
            const size_t idx = vnIndexEdgeMono[i];
            if(pFrame->mvbOutlier[idx]){ // 如果这条误差边是来自于outlier
                e->computeError();
            }
            const auto chi2 =(float) e->chi2();// 就是error*\Omega*error,表征了这个点的误差大小(考虑置信度以后)
            // 设置为outlier
            if(chi2>chi2Mono[it]){
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);//level 1 对应为外点,上面的过程中我们设置其为不优化
                nBad++;
            }
            else{
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }
            // 除了前两次优化需要RobustKernel以外, 其余的优化都不需要 -- 因为重投影的误差已经有明显的下降了
            if(it==2)
                e->setRobustKernel(nullptr);
        }
        /// 同样的原理遍历双目的误差边
        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++){
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];
            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx]){
                e->computeError();
            }
            const auto chi2 = (float)e->chi2();

            if(chi2>chi2Stereo[it]){
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else{
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(nullptr);
        }

        if(optimizer.edges().size()<10)
            break;
    }    

    // Recover optimized pose and return number of inliers
    /// Step 5 得到优化后的当前帧的位姿
    auto* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);
    // 并且返回内点数目
    return nInitialCorrespondences-nBad;
}


/**
 * @brief Local Bundle Adjustment
 *
 * 1. Vertex:
 *     - g2o::VertexSE3Expmap()，LocalKeyFrames，即当前关键帧的位姿、与当前关键帧相连的关键帧的位姿
 *     - g2o::VertexSE3Expmap()，FixedCameras，即能观测到LocalMapPoints的关键帧（并且不属于LocalKeyFrames）的位姿，在优化中这些关键帧的位姿不变
 *     - g2o::VertexSBAPointXYZ()，LocalMapPoints，即LocalKeyFrames能观测到的所有MapPoints的位置
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *     - g2o::EdgeStereoSE3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(ul,v,ur)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *
 * @param pKF        KeyFrame
 * @param pbStopFlag 是否停止优化的标志
 * @param pMap       在优化后，更新状态时需要用到Map的互斥量mMutexMapUpdate
 * @note 由局部建图线程调用,对局部地图进行优化的函数
 */
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{
    // 该优化函数用于LocalMapping线程的局部BA优化
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;// 局部关键帧
    /// Step 1 将当前关键帧及其共视关键帧加入局部关键帧
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    // 找到关键帧连接的共视关键帧（一级相连），加入局部关键帧中
    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(auto pKFi : vNeighKFs){
        pKFi->mnBALocalForKF = pKF->mnId;// 把参与局部BA的每一个关键帧的 mnBALocalForKF设置为当前关键帧的mnId，防止重复添加
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }
    /// Step 2 遍历局部关键帧中(一级相连)关键帧，将它们观测的地图点加入到局部地图点
    list<MapPoint*> lLocalMapPoints;
    for(auto & lLocalKeyFrame : lLocalKeyFrames){
        vector<MapPoint*> vpMPs = lLocalKeyFrame->GetMapPointMatches();
        for(auto pMP : vpMPs){
            if(pMP && !pMP->isBad() && pMP->mnBALocalForKF!=pKF->mnId){// mnBALocalForKF 是为了防止重复添加
                lLocalMapPoints.push_back(pMP);
                pMP->mnBALocalForKF=pKF->mnId;
            }
        }
    }
    /// Step 3 能观测到局部地图点，但不属于局部关键帧的关键帧(二级相连)，这些二级相连关键帧在局部BA优化时不优化
    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for(auto & lLocalMapPoint : lLocalMapPoints){
        map<KeyFrame*,size_t> observations = lLocalMapPoint->GetObservations();
        for(auto & observation : observations){
            KeyFrame* pKFi = observation.first;
            // pKFi->mnBALocalForKF!=pKF->mnId 表示不属于局部关键帧，pKFi->mnBAFixedForKF!=pKF->mnId 表示还未标记为fixed（固定的）关键帧
            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId){
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    /// Step 4 构造g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver= new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    auto * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)// 外界设置的停止优化标志, 可能在 Tracking::NeedNewKeyFrame() 里置位
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;// 记录参与局部BA的最大关键帧mnId

    /// Step 5 添加待优化的位姿顶点：局部关键帧的位姿
    for(auto pKFi : lLocalKeyFrames){
        auto * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId((int)pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }
    /// Step  6 添加不优化的位姿顶点：固定关键帧的位姿，注意这里调用了vSE3->setFixed(true)
    // 不优化为啥也要添加？回答：为了增加约束信息
    for(auto pKFi : lFixedCameras){
        auto * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId((int)pKFi->mnId);
        vSE3->setFixed(true);// 所有的这些顶点的位姿都不优化，只是为了增加约束项
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // 边的最大数目 = 位姿数目 * 地图点数目
    const int nExpectedSize = int((lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size());

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);
    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);
    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);
    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);
    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);
    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const auto thHuberMono =(float) sqrt(5.991);// 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
    const auto thHuberStereo =(float) sqrt(7.815);// 自由度为3的卡方分布，显著性水平为0.05，对应的临界阈值7.815
    /// Step  7 添加待优化的局部地图点顶点
    // 遍历所有的局部地图点
    for(auto pMP : lLocalMapPoints){
        // 添加顶点：MapPoint
        auto* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = int(pMP->mnId+maxKFid+1);// 前面记录maxKFid的作用在这里体现
        vPoint->setId(id);
        vPoint->setMarginalized(true);// 因为使用了LinearSolverType，所以需要将所有的三维点边缘化掉
        optimizer.addVertex(vPoint);

        /// Step 8 在添加完了一个地图点之后, 对每一对关联的地图点和关键帧构建边
        const map<KeyFrame*,size_t> observations = pMP->GetObservations();
        for(auto observation : observations){
            KeyFrame* pKFi = observation.first;
            if(!pKFi->isBad()){
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[observation.second];
                // Monocular observation
                if(pKFi->mvuRight[observation.second]<0){ /// 单目模式下
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;
                    //创建边
                    auto* e = new g2o::EdgeSE3ProjectXYZ();
                    // 边的第一个顶点是地图点
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    // 边的第一个顶点是观测到该地图点的关键帧
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((int)pKFi->mnId)));
                    //设置观测
                    e->setMeasurement(obs);
                    // 权重为特征点所在图像金字塔的层数的倒数
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                    // 使用鲁棒核函数抑制外点
                    auto* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    // 将边添加到优化器，记录边、边连接的关键帧、边连接的地图点信息
                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else{ /// 双目或RGB-D模式
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[observation.second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    auto* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((int)pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    auto* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }
    // 开始BA前再次确认是否有外部请求停止优化，因为这个变量是引用传递，会随外部变化
    // 可能在 Tracking::NeedNewKeyFrame(), mpLocalMapper->InsertKeyFrame 里置位
    if(pbStopFlag && *pbStopFlag)
        return;
    /// Step 9 分成两个阶段开始优化。
    // 第一阶段优化
    optimizer.initializeOptimization();
    optimizer.optimize(5);
    bool bDoMore= true;
    // 检查是否外部请求停止
    if(pbStopFlag && *pbStopFlag)
        bDoMore = false;
    // 如果有外部请求停止,那么就不在进行第二阶段的优化
    if(bDoMore){
        /// Step 10 检测outlier，并设置下次不优化
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++){
            g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];
            if(pMP->isBad())
                continue;
            // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）,如果 当前边误差超出阈值，或者边链接的地图点深度值为负，说明这个边有问题，不优化了。
            if(e->chi2()>5.991 || !e->isDepthPositive())
                e->setLevel(1);
            e->setRobustKernel(nullptr);// 第二阶段优化的时候就属于精求解了,所以就不使用核函数
        }
        // 对于所有的双目的误差边也都进行类似的操作
        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++){
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];
            if(pMP->isBad())
                continue;
            if(e->chi2()>7.815 || !e->isDepthPositive())
                e->setLevel(1);
            e->setRobustKernel(nullptr);
        }
        /// Step 11：排除误差较大的outlier后再次优化 -- 第二阶段优化
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }
    /// Step 12：在优化后重新计算误差，剔除连接误差比较大的关键帧和地图点
    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++){ // 对于单目误差边
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];
        if(pMP->isBad())
            continue;
        if(e->chi2()>5.991 || !e->isDepthPositive()){
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.emplace_back(pKFi,pMP);
        }
    }
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++){// 双目误差边
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];
        if(pMP->isBad())
            continue;
        if(e->chi2()>7.815 || !e->isDepthPositive()){
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.emplace_back(pKFi,pMP);
        }
    }
    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);
    // 删除点
    if(!vToErase.empty()){
        for(auto & pair : vToErase){
            KeyFrame* pKFi = pair.first;
            MapPoint* pMPi = pair.second;
            pKFi->EraseMapPointMatch(pMPi);// 连接偏差比较大，在关键帧中剔除对该地图点的观测
            pMPi->EraseObservation(pKFi);// 连接偏差比较大，在地图点中剔除对该关键帧的观测
        }
    }
    // Recover optimized data
    /// Step 13：优化后更新关键帧位姿以及地图点的位置、平均观测方向等属性
    //Keyframes
    for(auto localKeyFrame : lLocalKeyFrames){
        auto* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex((int)localKeyFrame->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        localKeyFrame->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for(auto pMP : lLocalMapPoints){
        auto* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(int(pMP->mnId+maxKFid+1)));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}


/**
 * @brief 闭环检测后，EssentialGraph优化，仅优化所有关键帧位姿，不优化地图点
 *
 * 1. Vertex:
 *     - g2o::VertexSim3Expmap，Essential graph中关键帧的位姿
 * 2. Edge:
 *     - g2o::EdgeSim3()，BaseBinaryEdge
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：经过CorrectLoop函数步骤2，Sim3传播校正后的位姿
 *         + InfoMatrix: 单位矩阵
 *
 * @param pMap               全局地图
 * @param pLoopKF            闭环匹配上的关键帧
 * @param pCurKF             当前关键帧
 * @param NonCorrectedSim3   未经过Sim3传播调整过的关键帧位姿
 * @param CorrectedSim3      经过Sim3传播调整过的关键帧位姿
 * @param LoopConnections    因闭环时地图点调整而新生成的边
 */
void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
    /// Step 1：构造优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    auto * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e-16);// 第一次迭代的初始lambda值，如未指定会自动计算一个合适的值
    optimizer.setAlgorithm(solver);
    // 获取当前地图中的所有关键帧 和地图点
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();
    const unsigned int nMaxKFid = pMap->GetMaxKFid();// 最大关键帧id，用于添加顶点时使用
    // 记录所有优化前关键帧的位姿，优先使用在闭环时通过Sim3传播调整过的Sim3位姿
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    // 记录所有关键帧经过本次本质图优化过的位姿
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);
    const int minFeat = 100; // 两个关键帧之间共视关系的权重的最小值

    /// Step 2：将地图中所有关键帧的位姿作为顶点添加到优化器
    // 尽可能使用经过Sim3调整的位姿,遍历全局地图中的所有的关键帧
    for(auto pKF : vpKFs){
        if(pKF->isBad())
            continue;
        auto* VSim3 = new g2o::VertexSim3Expmap();
        const int nIDi = (int)pKF->mnId;
        auto it = CorrectedSim3.find(pKF);
        if(it!=CorrectedSim3.end()){ // 如果该关键帧在闭环时通过Sim3传播调整过，优先用调整后的Sim3位姿
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else{// 如果该关键帧在闭环时没有通过Sim3传播调整过，用跟踪时的位姿，尺度为1
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }
        if(pKF==pLoopKF) // 闭环匹配上的帧不进行位姿优化（认为是准确的，作为基准）,注意这里并没有锁住第0个关键帧，所以初始关键帧位姿也做了优化
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        // 和当前系统的传感器有关，如果是RGBD或者是双目，那么就不需要优化sim3的缩放系数，保持为1即可
        VSim3->_fix_scale = bFixScale;
        // 添加顶点
        optimizer.addVertex(VSim3);
        vpVertices[nIDi]=VSim3;
    }
    // 保存由于闭环后优化sim3而出现的新的关键帧和关键帧之间的连接关系,其中id比较小的关键帧在前,id比较大的关键帧在后
    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;
    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();// 单位矩阵

    /// Step 3：添加第1种边：闭环时因为地图点调整而出现的关键帧间的新连接关系
    for(const auto & LoopConnection : LoopConnections){
        KeyFrame* pKF = LoopConnection.first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = LoopConnection.second; // 和pKF 形成新连接关系的关键帧
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();
        // 对于当前关键帧nIDi而言，遍历每一个新添加的关键帧nIDj链接关系
        for(auto spConnection : spConnections){
            const long unsigned int nIDj = spConnection->mnId;
            // 同时满足下面2个条件的跳过
            // 条件1：至少有一个不是pCurKF或pLoopKF
            // 条件2：共视程度太少(<100),不足以构成约束的边
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(spConnection)<minFeat)
                continue;
            // 通过上面考验的帧有两种情况：
            // 1、恰好是当前帧及其闭环帧 nIDi=pCurKF 并且nIDj=pLoopKF（此时忽略共视程度）
            // 2、任意两对关键帧，共视程度大于100
            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            auto* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((int)nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((int)nIDi)));
            e->setMeasurement(Sji);// Sji内部是经过了Sim调整的观测

            e->information() = matLambda;// 信息矩阵是单位阵,说明这类新增加的边对总误差的贡献也都是一样大的
            optimizer.addEdge(e);
            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj))); // 保证id小的在前,大的在后
        }
    }

    /// Step 4：添加跟踪时形成的边、闭环匹配成功形成的边
    for(auto pKF : vpKFs){
        const int nIDi =(int) pKF->mnId;
        g2o::Sim3 Swi;
        auto iti = NonCorrectedSim3.find(pKF);
        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();//优先使用未经过Sim3传播调整的位姿
        else
            Swi = vScw[nIDi].inverse();//没找到才考虑已经经过Sim3传播调整的位姿

        KeyFrame* pParentKF = pKF->GetParent();
        /// Step 4.1：添加第2种边：生成树的边（有父关键帧）
        // 父关键帧就是和当前帧共视程度最高的关键帧
        if(pParentKF){
            int nIDj = (int)pParentKF->mnId;// 父关键帧id
            g2o::Sim3 Sjw;
            auto itj = NonCorrectedSim3.find(pParentKF);
            if(itj!=NonCorrectedSim3.end())//优先使用未经过Sim3传播调整的位姿
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];
            // 计算父子关键帧之间的相对位姿
            g2o::Sim3 Sji = Sjw * Swi;

            auto* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;// 所有元素的贡献都一样;每个误差边对总误差的贡献也都相同
            optimizer.addEdge(e);
        }

        /// Step 4.2：添加第3种边：当前帧与闭环匹配帧之间的连接关系(这里面也包括了当前遍历到的这个关键帧之前曾经存在过的回环边)
        // 获取和当前关键帧形成闭环关系的关键帧
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(auto pLKF : sLoopEdges){
            if(pLKF->mnId<pKF->mnId){ // 注意要比当前遍历到的这个关键帧的id小,这个是为了避免重复添加
                g2o::Sim3 Slw;
                auto itl = NonCorrectedSim3.find(pLKF);
                if(itl!=NonCorrectedSim3.end()) //优先使用未经过Sim3传播调整的位姿
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                auto* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((int)pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        /// Step 4.3：添加第4种边：共视程度超过100的关键帧也作为边进行优化
        // 取出和当前关键帧共视程度超过100的关键帧
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(auto pKFn : vpConnectedKFs){
            // 避免重复添加. 避免以下情况：最小生成树中的父子关键帧关系,以及和当前遍历到的关键帧构成了回环关系
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn)){
                // 注意要比当前遍历到的这个关键帧的id要小,这个是为了避免重复添加
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId){
                    // 如果这条边已经添加了，跳过
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;
                    g2o::Sim3 Snw;
                    auto itn = NonCorrectedSim3.find(pKFn);
                    if(itn!=NonCorrectedSim3.end())// 优先未经过Sim3传播调整的位姿
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];
                    g2o::Sim3 Sni = Snw * Swi;
                    // 也是同样计算相对位姿
                    auto* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((int)pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    /// Step 5：开始g2o优化，迭代20次
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    /// Step 6：将优化后的位姿更新到关键帧中
    // 遍历地图中的所有关键帧
    for(auto pKFi : vpKFs){
        const int nIDi = (int)pKFi->mnId;
        auto* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]// 转换成尺度为1的变换矩阵的形式

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);
        pKFi->SetPose(Tiw);// 将更新的位姿写入到关键帧中
    }

    /// Step 7：步骤5和步骤6优化得到关键帧的位姿后，地图点根据参考帧优化前后的相对关系调整自己的位置
    // 遍历所有地图点
    for(auto pMP : vpMPs){
        if(pMP->isBad())
            continue;
        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId){ // 该地图点在闭环检测中被当前KF调整过，那么使用调整它的KF id
            nIDr = (int)pMP->mnCorrectedReference;
        }
        else{
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = (int)pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];  // 得到地图点参考关键帧优化前的位姿
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];// 得到地图点参考关键帧优化后的位姿

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);// 这里优化后的位置也是直接写入到地图点之中的

        pMP->UpdateNormalAndDepth();
    }
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    auto * solver_ptr = new g2o::BlockSolverX(linearSolver);

    auto* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    auto * vSim3 = new g2o::VertexSim3Expmap();
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0,2);
    vSim3->_principle_point1[1] = K1.at<float>(1,2);
    vSim3->_focal_length1[0] = K1.at<float>(0,0);
    vSim3->_focal_length1[1] = K1.at<float>(1,1);
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = (int)vpMatches1.size();
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                auto* vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                auto* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        auto* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        auto* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        auto* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        auto* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(nullptr);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(nullptr);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(nullptr);
            nBad++;
        }
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(nullptr);
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    auto* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}


} //namespace ORB_SLAM
