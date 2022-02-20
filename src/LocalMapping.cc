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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2 { \

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

/**
 * 线程主循环
 */
void LocalMapping::Run()
{
    mbFinished = false;// 标记状态，表示当前run函数正在运行，尚未结束
    while(true)
    {
        /// Step 1 告诉Tracking，LocalMapping正处于繁忙状态，请不要给我发送关键帧打扰我
        SetAcceptKeyFrames(false);
        if(CheckNewKeyFrames()){// 等待处理的关键帧列表不为空
            /// Step 2 处理列表中的关键帧，包括计算BoW、更新观测、描述子、共视图，插入到地图等
            ProcessNewKeyFrame();
            /// Step 3 根据地图点的观测情况剔除质量不好的地图点
            MapPointCulling();
            /// Step 4 当前关键帧与相邻关键帧通过三角化产生新的地图点，使得跟踪更稳
            CreateNewMapPoints();
            if(!CheckNewKeyFrames()){// 已经处理完队列中的最后的一个关键帧
                ///  Step 5 检查并融合当前关键帧与相邻关键帧帧（两级相邻）中重复的地图点
                SearchInNeighbors();
            }
            mbAbortBA = false;// 终止BA的标志
            // 已经处理完队列中的最后的一个关键帧，并且闭环检测没有请求停止LocalMapping
            if(!CheckNewKeyFrames() && !stopRequested()){
                /// Step 6 当局部地图中的关键帧大于2个的时候进行局部地图的BA
                if(mpMap->KeyFramesInMap()>2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);
                /// Step 7 检测并剔除当前帧相邻的关键帧中冗余的关键帧
                // 冗余的判定：该关键帧的90%的地图点可以被其它关键帧观测到
                KeyFrameCulling();
            }
            /// Step 8 将当前帧加入到闭环检测队列中
            // 注意这里的关键帧被设置成为了bad的情况,这个需要注意
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop()){
            // Safe area to stop
            while(isStopped() && !CheckFinish())
                std::this_thread::sleep_for(std::chrono::milliseconds(3000));
            if(CheckFinish()) // 然后确定终止了就跳出这个线程的主循环
                break;
        }
        // 查看是否有复位线程的请求
        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())// 如果当前线程已经结束了就跳出主循环
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    }

    SetFinish();// 设置线程已经终止
}

/**
 * @brief 插入关键帧,由外部线程调用
 * @details 将关键帧插入到地图中，以便将来进行局部地图优化 \n
 * NOTICE 这里仅仅是将关键帧插入到列表中进行等待
 * @param pKF KeyFrame
 */
void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}

/**
 * 查看列表中是否有等待被插入的关键帧
 * @return
 */
bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

/**
 * @brief 处理列表中的关键帧，包括计算BoW、更新观测、描述子、共视图，插入到地图等
 *
 */
void LocalMapping::ProcessNewKeyFrame()
{
    /// Step 1：从缓冲队列中取出一帧关键帧
    // 该关键帧队列是Tracking线程向LocalMapping中插入的关键帧组成
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    /// Step 2：计算该关键帧特征点的词袋向量
    mpCurrentKeyFrame->ComputeBoW();
    // Associate MapPoints to the new keyframe and update normal and descriptor
    /// Step 3：当前处理关键帧中有效的地图点，更新normal，描述子等信息
    // TrackLocalMap中和当前帧新匹配上的地图点和当前关键帧进行关联绑定
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    // 对当前处理的这个关键帧中的所有的地图点展开遍历
    for(size_t i=0; i<vpMapPointMatches.size(); i++){
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP){
            if(!pMP->isBad()){
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame)){ // 如果地图点不是来自当前帧的观测（比如来自局部地图点）
                    pMP->AddObservation(mpCurrentKeyFrame, i);//为当前地图点添加观测
                    pMP->UpdateNormalAndDepth();// 获得该点的平均观测方向和观测距离范围
                    pMP->ComputeDistinctiveDescriptors();// 更新地图点的最佳描述子
                }
                else{ // this can only happen for new stereo points inserted by the Tracking
                    // 如果当前帧中已经包含了这个地图点,但是这个地图点中却没有包含这个关键帧的信息
                    // 这些地图点可能来自双目或RGBD跟踪过程中新生成的地图点，或者是CreateNewMapPoints 中通过三角化产生
                    // 将上述地图点放入mlpRecentAddedMapPoints，等待后续MapPointCulling函数的检验
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    // Update links in the Covisibility Graph
    /// Step 4：更新关键帧间的连接关系（共视图）
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    /// Step 5：将该关键帧插入到地图中
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

/**
 * @brief 检查新增地图点，根据地图点的观测情况剔除质量不好的新增的地图点
 * mlpRecentAddedMapPoints：存储新增的地图点，这里是要删除其中不靠谱的
 */
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    auto lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;
    /// Step 1：根据相机类型设置不同的观测阈值
    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;
    /// Step 2：遍历检查新添加的地图点
    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad()){
            /// Step 2.1：已经是坏点的地图点仅从队列中删除
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f ){
            /// Step 2.2：跟踪到该地图点的帧数相比预计可观测到该地图点的帧数的比例小于25%，从地图中删除
            //(mnFound/mnVisible） < 25%, mnFound ：地图点被多少帧（包括普通帧）看到，次数越多越好,mnVisible：地图点应该被看到的次数
            // (mnFound/mnVisible）：对于大FOV镜头这个比例会高，对于窄FOV镜头这个比例会低
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs){
            /// Step 2.3：从该点建立开始，到现在已经过了不小于2个关键帧
            // 但是观测到该点的相机数却不超过阈值cnThObs，从地图中删除
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid- (int)pMP->mnFirstKFid) >=3){
            /// Step 2.4：从建立该点开始，已经过了3个关键帧而没有被剔除，则认为是质量高的点
            // 因此没有SetBadFlag()，仅从队列中删除
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else{
            lit++;
        }
    }
}


/**
 * @brief 用当前关键帧与相邻关键帧通过三角化产生新的地图点，使得跟踪更稳
 *
 */
void LocalMapping::CreateNewMapPoints()
{
    // nn表示搜索最佳共视关键帧的数目
    // 不同传感器下要求不一样,单目的时候需要有更多的具有较好共视关系的关键帧来建立地图
    int nn = 10;
    if(mbMonocular)
        nn=20;
    /// Step 1：在当前关键帧的共视关键帧中找到共视程度最高的nn帧相邻关键帧
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    // 取出当前帧从世界坐标系到相机坐标系的变换矩阵
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();
    // 得到当前关键帧（左目）光心在世界坐标系中的坐标、内参
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;
    // 用于后面的点深度的验证;这里的1.5是经验值, mfScaleFactor = 1.2
    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;
    int nnew=0;// 记录三角化成功的地图点数目
    // Search matches with epipolar restriction and triangulate
    ORBmatcher matcher(0.6,false);// 特征点匹配配置 最佳距离 < 0.6*次佳距离，比较苛刻了。不检查旋转

    /// Step 2：遍历相邻关键帧，搜索匹配并用极线约束剔除误匹配，最终三角化
    for(size_t i=0; i<vpNeighKFs.size(); i++){
        if(i>0 && CheckNewKeyFrames()) //先去处理新的关键帧
            return;
        KeyFrame* pKF2 = vpNeighKFs[i];
        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();// 相邻的关键帧光心在世界坐标系中的坐标
        cv::Mat vBaseline = Ow2-Ow1;// 基线向量，两个关键帧间的相机位移
        const float baseline =(float) cv::norm(vBaseline);// 基线长度
        /// Step 3：判断相机运动的基线是不是足够长
        if(!mbMonocular){
            // 如果是双目相机，关键帧间距小于本身的基线时不生成3D点,因为太短的基线下能够恢复的地图点不稳定
            if(baseline < pKF2->mb)
                continue;
        }
        else{// 单目相机情况
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);// 相邻关键帧的场景深度中值
            const float ratioBaselineDepth = baseline/medianDepthKF2;// 基线与景深的比例
            // 如果比例特别小，基线太短恢复3D点不准，那么跳过当前邻接的关键帧，不生成3D点
            if(ratioBaselineDepth<0.01)
                continue;
        }
        /// Step 4：根据两个关键帧的位姿计算它们之间的基础矩阵
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);
        /// Step 5：通过词袋对两关键帧的未匹配的特征点快速匹配，用极线约束抑制离群点，生成新的匹配点对
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        /// Step 6：对每对匹配通过三角化生成3D点,和 Triangulate函数差不多
        const int nmatches =(int) vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++){
            /// Step 6.1：取出匹配特征点
            const int &idx1 =(int) vMatchedIndices[ikp].first;// 当前匹配对在当前关键帧中的索引
            const int &idx2 =(int) vMatchedIndices[ikp].second;// 当前匹配对在邻接关键帧中的索引
            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];// 当前匹配在当前关键帧中的特征点
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];// mvuRight中存放着双目的深度值，如果不是双目，其值将为-1
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];// 当前匹配在邻接关键帧中的特征点
            const float kp2_ur = pKF2->mvuRight[idx2];// mvuRight中存放着双目的深度值，如果不是双目，其值将为-1
            bool bStereo2 = kp2_ur>=0;

            /// Step 6.2：利用匹配点反投影得到视差角
            // 特征点反投影,其实得到的是在各自相机坐标系下的一个非归一化的方向向量,和这个点的反投影射线重合
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);
            // 由相机坐标系转到世界坐标系(得到的是那条反投影射线的一个同向向量在世界坐标系下的表示,还是只能够表示方向)，得到视差角余弦值
            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays =float( ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2)));// 匹配点射线夹角余弦值

            float cosParallaxStereo = cosParallaxRays+1;// 加1是为了让cosParallaxStereo随便初始化为一个很大的值
            // cosParallaxStereo1、cosParallaxStereo2在后面可能不存在，需要初始化为较大的值
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;
            /// Step 6.3：对于双目，利用双目得到视差角；单目相机没有特殊操作
            if(bStereo1)
                // 传感器是双目相机,并且当前的关键帧的这个点有对应的深度
                // 假设是平行的双目相机，计算出双目相机观察这个点的时候的视差角余弦
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                // 传感器是双目相机,并且邻接的关键帧的这个点有对应的深度，和上面一样操作
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));
            // 得到双目观测的视差角中最小的那个
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            /// Step 6.4：三角化恢复3D点
            cv::Mat x3D;
            // cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998)表明视差角正常,0.9998 对应1°
            // cosParallaxRays < cosParallaxStereo 表明匹配点对夹角大于双目本身观察三维点夹角
            // 匹配点对夹角大，用三角法恢复3D点
            // 参考：https://github.com/raulmur/ORB_SLAM2/issues/345
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998)){
                // 见Initializer.cc的 Triangulate 函数,实现是一样的,顶多就是把投影矩阵换成了变换矩阵
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0) // 归一化之前的检查
                    continue;
                // 归一化成为齐次坐标,然后提取前面三个维度作为欧式坐标
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            // 匹配点对夹角小，用双目恢复3D点
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                // 如果是双目，用视差角更大的那个双目信息来恢复，直接用已知3D点反投影了
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else{
                continue; //No stereo and very low parallax
            }

            cv::Mat x3Dt = x3D.t();// 为方便后续计算，转换成为了行向量

            /// Step 6.5：检测生成的3D点是否在相机前方,不在的话就放弃这个点
            float z1 = float(Rcw1.row(2).dot(x3Dt))+tcw1.at<float>(2);
            if(z1<=0)
                continue;
            float z2 = float(Rcw2.row(2).dot(x3Dt))+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            /// Step 6.6：计算3D点在当前关键帧下的重投影误差
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = float(Rcw1.row(0).dot(x3Dt))+tcw1.at<float>(0);
            const float y1 = float(Rcw1.row(1).dot(x3Dt))+tcw1.at<float>(1);
            const float invz1 = 1.f/z1;
            if(!bStereo1){ // 单目情况下
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1) // 假设测量有一个像素的偏差，2自由度卡方检验阈值是5.991
                    continue;
            }
            else{// 双目情况
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1) // 自由度为3，卡方检验阈值是7.8
                    continue;
            }

            // 计算3D点在另一个关键帧下的重投影误差，操作同上
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = float(Rcw2.row(0).dot(x3Dt))+tcw2.at<float>(0);
            const float y2 = float(Rcw2.row(1).dot(x3Dt))+tcw2.at<float>(1);
            const float invz2 = 1.0f/z2;
            if(!bStereo2){
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else{
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            /// Step 6.7：检查尺度连续性
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = (float)cv::norm(normal1);
            cv::Mat normal2 = x3D-Ow2;
            float dist2 = (float)cv::norm(normal2);
            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;// ratioDist是不考虑金字塔尺度下的距离比例
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];// 金字塔尺度因子的比例

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            // 距离的比例和图像金字塔的比例不应该差太多，否则就跳过
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;
            /// Step 6.8：三角化生成3D点成功，构造成MapPoint
            auto* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);
            /// a.观测到该MapPoint的关键帧
            pMP->AddObservation(mpCurrentKeyFrame,idx1);
            pMP->AddObservation(pKF2,idx2);
            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);
            /// b.该MapPoint的描述子
            pMP->ComputeDistinctiveDescriptors();
            /// c.该MapPoint的平均观测方向和深度范围
            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            /// Step 6.10：将新产生的点放入检测队列
            // 这些MapPoints都会经过MapPointCulling函数的检验
            mlpRecentAddedMapPoints.push_back(pMP);
            nnew++;
        }
    }
}

/**
 * @brief 检查并融合当前关键帧与相邻帧（两级相邻）重复的地图点
 *
 */
void LocalMapping::SearchInNeighbors()
{
    /// Step 1：获得当前关键帧在共视图中权重排名前nn的邻接关键帧
    // 开始之前先定义几个概念
    // 当前关键帧的邻接关键帧，称为一级相邻关键帧，也就是邻居
    // 与一级相邻关键帧相邻的关键帧，称为二级相邻关键帧，也就是邻居的邻居
    int nn = 10;
    if(mbMonocular)
        nn=20;
    // 和当前关键帧相邻的关键帧，也就是一级相邻关键帧
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    /// Step 2：存储一级相邻关键帧及其二级相邻关键帧
    vector<KeyFrame*> vpTargetKFs;
    for(auto pKFi : vpNeighKFs){
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;// 标记已经加入
        // 以一级相邻关键帧的共视关系最好的5个相邻关键帧 作为二级相邻关键帧
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(auto pKFi2 : vpSecondNeighKFs){
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    /// Step 3：将当前帧的地图点分别投影到两级相邻关键帧，寻找匹配点对应的地图点进行融合，称为正向投影融合
    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(auto pKFi : vpTargetKFs){
        ORB_SLAM2::ORBmatcher::Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(auto pKFi : vpTargetKFs)
    {
        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();
        for(auto pMP : vpMapPointsKFi)
        {
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    ORBmatcher::Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(auto pMP : vpMapPointMatches){
        if(pMP){
            if(!pMP->isBad()){
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

/**
 * 根据两关键帧的姿态计算两个关键帧之间的基本矩阵
 * @param  pKF1 关键帧1
 * @param  pKF2 关键帧2
 * @return      基本矩阵
 */
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    // 先构造两帧之间的R12,t12
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;
    // 得到 t12 的反对称矩阵
    cv::Mat t12x = SkewSymmetricMatrix(t12);
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;
    // Essential Matrix: t12叉乘R12
    // Fundamental Matrix: inv(K1)*E*inv(K2)
    return K1.t().inv()*t12x*R12*K2.inv();
}


/**
 * 停止局部BA
 */
void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(auto & nkf : mlNewKeyFrames)
        delete nkf;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

/**
 * 查看当前是否允许接受关键帧
 * @return
 */
bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

/**
 * 设置"允许接受关键帧"的状态标志
 * @param flag
 */
void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(auto observation : observations)
                        {
                            KeyFrame* pKFi = observation.first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[observation.second].octave;

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

/**
 * @brief 计算三维向量v的反对称矩阵
 * @param[in] v     三维向量
 * @return cv::Mat  反对称矩阵
 */
cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<
            0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(true)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        std::this_thread::sleep_for((std::chrono::milliseconds(3000)));
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
