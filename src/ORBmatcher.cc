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

#include "ORBmatcher.h"

#include<climits>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM2
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50; //匹配的描述子的距离
const int ORBmatcher::HISTO_LENGTH = 30;//旋转直方图的长度


ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

/**
 * @brief 通过投影地图点到当前帧，对Local MapPoint进行跟踪
 * 步骤
 * Step 1 遍历有效的局部地图点
 * Step 2 设定搜索搜索窗口的大小。取决于视角, 若当前视角和平均视角夹角较小时, r取一个较小的值
 * Step 3 通过投影点以及搜索窗口和预测的尺度进行搜索, 找出搜索半径内的候选匹配点索引
 * Step 4 寻找候选匹配点中的最佳和次佳匹配点
 * Step 5 筛选最佳匹配点
 * @param[in] F                         当前帧
 * @param[in] vpMapPoints               局部地图点，来自局部关键帧
 * @param[in] th                        搜索范围
 * @return int                          成功匹配的数目
 */
int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th) const
{
    int nmatches=0;
    const bool bFactor = th!=1.0; // 如果 th！=1 (RGBD 相机或者刚刚进行过重定位), 需要扩大范围搜索
    /// Step 1 遍历有效的局部地图点
    for(auto pMP : vpMapPoints) {
        if(!pMP->mbTrackInView) // 判断该点是否要投影
            continue;
        if(pMP->isBad())
            continue;
        const int &nPredictedLevel = pMP->mnTrackScaleLevel;// 通过距离预测的金字塔层数，该层数相对于当前的帧
        /// Step 2 设定搜索搜索窗口的大小。取决于视角, 若当前视角和平均视角夹角较小时, r取一个较小的值
        float r = RadiusByViewingCos(pMP->mTrackViewCos);
        if(bFactor)// 如果需要扩大范围搜索，则乘以阈值th
            r*=th;
        /// Step 3 通过投影点以及搜索窗口和预测的尺度进行搜索, 找出搜索半径内的候选匹配点索引
        const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],
                                    nPredictedLevel-1,nPredictedLevel);
        if(vIndices.empty())// 没找到候选的,就放弃对当前点的匹配
            continue;

        const cv::Mat MPdescriptor = pMP->GetDescriptor();
        // 最优的次优的描述子距离和index
        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        /// Step 4 寻找候选匹配点中的最佳和次佳匹配点
        for(unsigned long idx : vIndices) {
            if(F.mvpMapPoints[idx]){
                if(F.mvpMapPoints[idx]->Observations()>0) // 如果Frame中的该兴趣点已经有对应的MapPoint了,则退出该次循环
                    continue;
            }
            //如果是双目数据
            if(F.mvuRight[idx]>0){
                const float er = std::fabs(pMP->mTrackProjXR-F.mvuRight[idx]); //计算在X轴上的投影误差
                //超过阈值,说明这个点不行,丢掉.
                //这里的阈值定义是以给定的搜索范围r为参考,然后考虑到越近的点(nPredictedLevel越大), 相机运动时对其产生的影响也就越大,
                //因此需要扩大其搜索空间.
                //当给定缩放倍率为1.2的时候, mvScaleFactors 中的数据是: 1 1.2 1.2^2 1.2^3 ...
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

            const cv::Mat &d = F.mDescriptors.row((int)idx);
            /// 计算地图点和候选投影点的描述子距离
            const int dist = DescriptorDistance(MPdescriptor,d);
            // 寻找描述子距离最小和次小的特征点和索引
            if(dist<bestDist){
                bestDist2=bestDist;
                bestDist=dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx=(int)idx;
            }
            else if(dist<bestDist2){
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        /// Step 5 筛选最佳匹配点
        if(bestDist<=TH_HIGH){// 最佳匹配距离还需要满足在设定阈值内
            // 条件1：bestLevel==bestLevel2 表示 最佳和次佳在同一金字塔层级
            // 条件2：bestDist>mfNNratio*bestDist2 表示最佳和次佳距离不满足阈值比例。理论来说 bestDist/bestDist2 越小越好
            if(bestLevel==bestLevel2 && (float)bestDist>mfNNratio*(float)bestDist2)
                continue;
            //保存结果: 为Frame中的特征点增加对应的MapPoint
            F.mvpMapPoints[bestIdx]=pMP;
            nmatches++;
        }
    }

    return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}


bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
}

/**
 * @brief 通过词袋，对关键帧的特征点进行跟踪
 * 步骤
 * Step 1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
 * Step 2：遍历KF中属于该node的特征点
 * Step 3：遍历F中属于该node的特征点，寻找最佳匹配点
 * Step 4：根据阈值 和 角度投票剔除误匹配
 * Step 5：根据方向剔除误匹配的点
 * @param  pKF               关键帧
 * @param  F                 当前普通帧
 * @param  vpMapPointMatches F中地图点对应的匹配，NULL表示未匹配
 * @return                   成功匹配的数量
 */
int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches) const
{
    // 获取该关键帧的地图点
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
    // 和普通帧F特征点的索引一致
    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(nullptr));
    // 取出关键帧的词袋特征向量
    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;
    int nmatches=0;

    vector<int> rotHist[HISTO_LENGTH];// 特征点角度旋转差统计用的直方图
    for(auto & rh : rotHist)
        rh.reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    // 将属于同一节点(特定层)的ORB特征进行匹配
    auto KFit = vFeatVecKF.cbegin();
    auto Fit = F.mFeatVec.cbegin();
    auto KFend = vFeatVecKF.cend();
    auto Fend = F.mFeatVec.cend();
    while(KFit != KFend && Fit != Fend) {
        /// Step 1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
        if(KFit->first == Fit->first) { // first 元素就是node id，遍历
            const vector<unsigned int> vIndicesKF = KFit->second;// second 是该node内存储的feature index
            const vector<unsigned int> vIndicesF = Fit->second;
            /// Step 2：遍历KF中属于该node的特征点
            for(unsigned int realIdxKF : vIndicesKF){
                MapPoint* pMP = vpMapPointsKF[realIdxKF];// 取出KF中该特征对应的地图点
                if(!pMP)
                    continue;
                if(pMP->isBad())
                    continue;                
                const cv::Mat &dKF= pKF->mDescriptors.row((int)realIdxKF);// 取出KF中该特征对应的描述子
                int bestDist1=256;// 最好的距离（最小距离）
                int bestIdxF =-1 ;
                int bestDist2=256;// 次好距离（倒数第二小距离）
                /// Step 3：遍历F中属于该node的特征点，寻找最佳匹配点
                for(unsigned int realIdxF : vIndicesF){
                    if(vpMapPointMatches[realIdxF])// 如果地图点存在，说明这个点已经被匹配过了，不再匹配，加快速度
                        continue;
                    const cv::Mat &dF = F.mDescriptors.row((int)realIdxF);// 取出F中该特征对应的描述子
                    // 计算描述子的距离
                    const int dist =  DescriptorDistance(dKF,dF);
                    // 遍历，记录最佳距离、最佳距离对应的索引、次佳距离等
                    if(dist<bestDist1){
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=(int)realIdxF;
                    }
                    else if(dist<bestDist2){
                        bestDist2=dist;
                    }
                }
                /// Step 4：根据阈值 和 角度投票剔除误匹配
                /// Step 4.1：第一关筛选：匹配距离必须小于设定阈值
                if(bestDist1<=TH_LOW){
                    /// Step 4.2：第二关筛选：最佳匹配比次佳匹配明显要好，那么最佳匹配才真正靠谱
                    if(float(bestDist1) < mfNNratio*float(bestDist2)){
                        /// Step 4.3：记录成功匹配特征点的对应的地图点(来自关键帧)
                        vpMapPointMatches[bestIdxF]=pMP;
                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];// 这里的realIdxKF是当前遍历到的关键帧的特征点id
                        /// Step 4.4：计算匹配点旋转角度差所在的直方图
                        if(mbCheckOrientation){
                            // angle：每个特征点在提取描述子时的旋转主方向角度，如果图像旋转了，这个角度将发生改变
                            // 所有的特征点的角度变化应该是一致的，通过直方图统计得到最准确的角度变化值
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin =(int) round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        nmatches++;
                    }
                }
            }
            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first){// 对齐
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else{// 对齐
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }

    /// Step 5 根据方向剔除误匹配的点
    if(mbCheckOrientation){
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
        // 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin的索引
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);
        for(int i=0; i<HISTO_LENGTH; i++){
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(int j : rotHist[i]){// 剔除掉不在前三的匹配对，因为他们不符合“主流旋转方向”
                vpMapPointMatches[j]=static_cast<MapPoint*>(nullptr);
                nmatches--;
            }
        }
    }

    return nmatches;
}

/**
 * 用于回环检测
 * 使用Sim3变换，将vpPoints和pKF进行匹配，最匹配的MapPoints存入vpMatched中
 * @param pKF
 * @param Scw
 * @param vpPoints
 * @param vpMatched
 * @param th
 * @return
 */
int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw =(float) sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(nullptr));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(auto pMP : vpPoints)
    {
        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = (float)cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(unsigned long idx : vIndices)
        {
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row((int)idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = (int)idx;
            }
        }

        if(bestDist<=TH_LOW)
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }

    return nmatches;
}


/**
 * 用于单目初始化.
 * 匹配F1和F2的特征点，匹配的特征点索引存入vnMatches12
 * @param F1
 * @param F2
 * @param vbPrevMatched
 * @param vnMatches12
 * @param windowSize
 * @return
 */
int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize) const
{
    //旋转的直方图,表示0-360度的角度,每个bin记录一个角度范围的特征点
    vector<int> rotHist[HISTO_LENGTH];
    for(auto & rh : rotHist)
        rh.reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;
    int nmatches=0;
    //-1表示这里面含没有存储匹配点，存储的是F1到F2特征点的索引
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);
    //两个keypoints的匹配距离，是按照F2中的keypoints的数量大小来进行分配的
    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
    //F2特征到F1特征的匹配,用来标志F2中的某个特征是否已经被匹配过了
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

    ///遍历Frame1中所有的特征点
    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++) {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        if(level1>0) //只提取第0层图像中的特征点
            continue;
        ///获取以vbPrevMatched[i1]为中心,距离在windowSize内的所有特征点,作为候选的匹配特征点
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, (float)windowSize,level1,level1);
        if(vIndices2.empty())
            continue;
        cv::Mat d1 = F1.mDescriptors.row((int)i1);//il的描述子
        int bestDist = INT_MAX;//最佳距离
        int bestDist2 = INT_MAX;//次佳距离
        int bestIdx2 = -1;
        ///遍历所有的候选特征点
        for(unsigned long i2 : vIndices2)
        {
            cv::Mat d2 = F2.mDescriptors.row((int)i2);//i2的描述子
            ///计算两个描述子之间的距离
            int dist = DescriptorDistance(d1,d2);
            if(vMatchedDistance[i2] <= dist)
                continue;
            ///记录最佳的匹配特征
            if(dist<bestDist){
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=(int)i2;
            }
            else if(dist<bestDist2){
                bestDist2=dist;
            }
        }
        ///满足距离阈值
        if(bestDist<=TH_LOW){
            if((float)bestDist<(float)bestDist2*mfNNratio){//最小距离和次小距离的比值 满足 阈值
                if(vnMatches21[bestIdx2]>=0){ //如果最佳匹配点的ID在vnMatches21中>0到说明，他已经被匹配过了
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                ///设置匹配索引 距离
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=(int)i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;
                ///将特征点放入直方图中,用来检查方向是否匹配
                if(mbCheckOrientation){
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;//角度差
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = (int)std::round(rot*factor);
                    if(bin==HISTO_LENGTH) bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back((int)i1);//将特征点放入直方图中
                }
            }
        }
    }
    ///删除旋转直方图中的 非主流部分
    if(mbCheckOrientation){
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
        //算出旋转角度差落在bin中最多的前三位的索引
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);
        for(int i=0; i<HISTO_LENGTH; i++){ //删除其它bin的特征匹配
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(int idx1 : rotHist[i]){
                if(vnMatches12[idx1]>=0){
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }
    }

    ///更新匹配
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;
    return nmatches;
}

/**
 * 用于回环检测
 * 根据Dbow,筛选pKF1和pKF2匹配的MapPoints，并将pKF1的索引和pKF2的索引存入vpMatch12中
 * @param pKF1
 * @param pKF2
 * @param vpMatches12
 * @return
 */
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12) const
{
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(nullptr));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    vector<int> rotHist[HISTO_LENGTH];
    for(auto & i : rotHist)
        i.reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;

    auto f1it = vFeatVec1.cbegin();
    auto f2it = vFeatVec2.cbegin();
    auto f1end = vFeatVec1.cend();
    auto f2end = vFeatVec2.cend();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = vpMapPoints1[idx1];
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row((int)idx1);

                int bestDist1=256;
                int bestIdx2 =-1 ;
                int bestDist2=256;

                for(unsigned long idx2 : f2it->second)
                {
                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    if(vbMatched2[idx2] || !pMP2)
                        continue;

                    if(pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row((int)idx2);

                    int dist = DescriptorDistance(d1,d2);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=(int)idx2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation)
                        {
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin =(int) round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back((int)idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(int j : rotHist[i])
            {
                vpMatches12[j]=static_cast<MapPoint*>(nullptr);
                nmatches--;
            }
        }
    }

    return nmatches;
}


/**
 * @brief 利用基础矩阵F12极线约束，用BoW加速匹配两个关键帧的未匹配的特征点，产生新的匹配点对
 * 具体来说，pKF1图像的每个特征点与pKF2图像同一node节点的所有特征点依次匹配，判断是否满足对极几何约束，满足约束就是匹配的特征点
 * @param pKF1          关键帧1
 * @param pKF2          关键帧2
 * @param F12           从2到1的基础矩阵
 * @param vMatchedPairs 存储匹配特征点对，特征点用其在关键帧中的索引表示
 * @param bOnlyStereo   在双目和rgbd情况下，是否要求特征点在右图存在匹配
 * @return              成功匹配的数量
 */
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, const cv::Mat& F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo) const
{    
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    /// Step 1 计算KF1的相机中心在KF2图像平面的二维像素坐标
    cv::Mat Cw = pKF1->GetCameraCenter(); // KF1相机光心在世界坐标系坐标Cw
    cv::Mat R2w = pKF2->GetRotation();// KF2相机位姿R2w,t2w,是世界坐标系到相机坐标系
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w*Cw+t2w;// KF1的相机光心转化到KF2坐标系中的坐标
    const float invz = 1.0f/C2.at<float>(2);
    const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx; // 得到KF1的相机光心在KF2中的坐标，也叫极点，这里是像素坐标
    const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);// 记录匹配是否成功，避免重复匹配
    vector<int> vMatches12(pKF1->N,-1);
    // 用于统计匹配点对旋转差的直方图
    vector<int> rotHist[HISTO_LENGTH];
    for(auto & rh : rotHist)
        rh.reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    auto f1it = vFeatVec1.cbegin();
    auto f2it = vFeatVec2.cbegin();
    auto f1end = vFeatVec1.cend();
    auto f2end = vFeatVec2.cend();
    /// Step 2 利用BoW加速匹配：只对属于同一节点(特定层)的ORB特征进行匹配
    /// Step 2.1：遍历pKF1和pKF2中的node节点
    while(f1it!=f1end && f2it!=f2end){
        if(f1it->first == f2it->first){ // 如果f1it和f2it属于同一个node节点才会进行匹配，这就是BoW加速匹配原理
            /// Step 2.2：遍历属于同一node节点(id：f1it->first)下的所有特征点
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++){
                const size_t idx1 = f1it->second[i1];// 获取pKF1中属于该node节点的所有特征点索引
                /// Step 2.3：通过特征点索引idx1在pKF1中取出对应的MapPoint
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);
                if(pMP1) // 由于寻找的是未匹配的特征点，所以pMP1应该为NULL
                    continue;
                const bool bStereo1 = pKF1->mvuRight[idx1]>=0;// 如果mvuRight中的值大于0，表示是双目，且该特征点有深度值
                if(bOnlyStereo){
                    if(!bStereo1)
                        continue;
                }
                /// Step 2.4：通过特征点索引idx1在pKF1中取出对应的特征点
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
                const cv::Mat &d1 = pKF1->mDescriptors.row((int)idx1); // 通过特征点索引idx1在pKF1中取出对应的特征点的描述子
                
                int bestDist = TH_LOW;
                int bestIdx2 = -1;
                /// Step 2.5：遍历该node节点下(f2it->first)对应KF2中的所有特征点
                for(unsigned long idx2 : f2it->second){
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2); // 通过特征点索引idx2在pKF2中取出对应的MapPoint
                    // If we have already matched or there is a MapPoint skip
                    // 如果pKF2当前特征点索引idx2已经被匹配过或者对应的3d点非空，那么跳过这个索引idx2
                    if(vbMatched2[idx2] || pMP2)
                        continue;
                    const bool bStereo2 = pKF2->mvuRight[idx2]>=0;
                    if(bOnlyStereo){
                        if(!bStereo2)
                            continue;
                    }
                    // 通过特征点索引idx2在pKF2中取出对应的特征点的描述子
                    const cv::Mat &d2 = pKF2->mDescriptors.row((int)idx2);
                    /// Step 2.6 计算idx1与idx2在两个关键帧中对应特征点的描述子距离
                    const int dist = DescriptorDistance(d1,d2);
                    if(dist>TH_LOW || dist>bestDist)
                        continue;
                    // 通过特征点索引idx2在pKF2中取出对应的特征点
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
                    if(!bStereo1 && !bStereo2){
                        const float distex = ex-kp2.pt.x;
                        const float distey = ey-kp2.pt.y;
                        /// Step 2.7 极点e2到kp2的像素距离如果小于阈值th,认为kp2对应的MapPoint距离pKF1相机太近，跳过该匹配点对
                        // 作者根据kp2金字塔尺度因子(scale^n，scale=1.2，n为层数)定义阈值th
                        // 金字塔层数从0到7，对应距离 sqrt(100*pKF2->mvScaleFactors[kp2.octave]) 是10-20个像素
                        //? 对这个阈值的有效性持怀疑态度
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }
                    /// Step 2.8 计算特征点kp2到kp1对应极线的距离是否小于阈值
                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2)){
                        bestIdx2 = (int)idx2;// bestIdx2，bestDist 是 kp1 对应 KF2中的最佳匹配点 index及匹配距离
                        bestDist = dist;
                    }
                }
                
                if(bestIdx2>=0){
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
                    vMatches12[idx1]=bestIdx2;// 记录匹配结果
                    nmatches++;
                    if(mbCheckOrientation){ // 记录旋转差直方图信息
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin =(int) round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back((int)idx1);
                    }
                }
            }
            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    /// Step 3 用旋转差直方图来筛掉错误匹配对
    if(mbCheckOrientation){
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);
        for(int i=0; i<HISTO_LENGTH; i++){
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(int j : rotHist[i]){
                vMatches12[j]=-1;
                nmatches--;
            }
        }

    }
    /// Step 4 存储匹配关系，下标是关键帧1的特征点id，存储的是关键帧2的特征点id
    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);
    for(size_t i=0, iend=vMatches12.size(); i<iend; i++){
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.emplace_back(i,vMatches12[i]);
    }

    return nmatches;
}


/**
 * @brief 将地图点投影到关键帧中进行匹配和融合；融合策略如下
 * 1.如果地图点能匹配关键帧的特征点，并且该点有对应的地图点，那么选择观测数目多的替换两个地图点
 * 2.如果地图点能匹配关键帧的特征点，并且该点没有对应的地图点，那么为该点添加该投影地图点

 * @param[in] pKF           关键帧
 * @param[in] vpMapPoints   待投影的地图点
 * @param[in] th            搜索窗口的阈值，默认为3
 * @return int              更新地图点的数量
 */
int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
{
    // 取出当前帧位姿、内参、光心在世界坐标系下坐标
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;
    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused=0;

    // 遍历所有的待投影地图点
    const int nMPs =(int) vpMapPoints.size();
    for(int i=0; i<nMPs; i++){
        MapPoint* pMP = vpMapPoints[i];
        /// Step 1 判断地图点的有效性
        if(!pMP)
            continue;
        if(pMP->isBad() || pMP->IsInKeyFrame(pKF)) // 地图点无效 或 已经是该帧的地图点（无需融合），跳过
            continue;
        // 将地图点变换到关键帧的相机坐标系下
        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc = Rcw*p3Dw + tcw;
        // 深度值为负，跳过
        if(p3Dc.at<float>(2)<0.0f)
            continue;
        /// Step 2 得到地图点投影到关键帧的图像坐标
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;
        const float u = fx*x+cx;
        const float v = fy*y+cy;
        // 投影点需要在有效范围内
        if(!pKF->IsInImage(u,v))
            continue;

        const float ur = u-bf*invz;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D =(float) cv::norm(PO);

        /// Step 3 地图点到关键帧相机光心距离需满足在有效范围内
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        /// Step 4 地图点到光心的连线与该地图点的平均观测向量之间夹角要小于60°
        cv::Mat Pn = pMP->GetNormal();
        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist3D,pKF);// 根据地图点到相机光心距离预测匹配点所在的金字塔尺度
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];// 确定搜索范围
        /// Step 5 在投影点附近搜索窗口内找到候选匹配点的索引
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);
        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        /// Step 6 遍历寻找最佳匹配点
        const cv::Mat dMP = pMP->GetDescriptor();
        int bestDist = 256;
        int bestIdx = -1;
        for(unsigned long idx : vIndices){
            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];
            const int &kpLevel= kp.octave;
            // 金字塔层级要接近（同一层或小一层），否则跳过
            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;
            // 计算投影点与候选匹配特征点的距离，如果偏差很大，直接跳过
            if(pKF->mvuRight[idx]>=0){  // 双目情况
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;
                //自由度为3, 误差小于1个像素,这种事情95%发生的概率对应卡方检验阈值为7.82
                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;
            }
            else{// 单目情况
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float e2 = ex*ex+ey*ey;
                // 自由度为2的，卡方检验阈值5.99（假设测量有一个像素的偏差）
                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }

            const cv::Mat &dKF = pKF->mDescriptors.row((int)idx);
            ///计算描述子距离
            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx =(int) idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        /// Step 7 找到投影点对应的最佳匹配特征点，根据是否存在地图点来融合或新增
        // 最佳匹配距离要小于阈值
        if(bestDist<=TH_LOW){
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF){
                if(!pMPinKF->isBad()){
                    if(pMPinKF->Observations() > pMP->Observations()) // 如果最佳匹配点有对应有效地图点，选择被观测次数最多的那个替换
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else{ // 如果最佳匹配点没有对应地图点，添加观测信息
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}


/**
 * 回环检测
 * 使用Sim3变换，将vpPoints和pKF进行匹配，若有匹配的MapPoint，则存入vpReplacePoint，若无，则将vpPoints增补到pKF中
 * @param pKF
 * @param Scw
 * @param vpPoints
 * @param th
 * @param vpReplacePoint
 * @return
 */
int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = (float)std::sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;

    const int nPoints = (int)vpPoints.size();

    // For each candidate MapPoint project and match
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // Project into Image
        const float invz = 1.f/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D =(float) cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(unsigned long idx : vIndices)
        {
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row((int)idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = (int)idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}


/**
 * 筛选回环候选关键帧
 * 使用pKF1和pKF2的Sim3变换，增加pKF1和Pkf2间匹配的MapPoints，并存入vpMatches12中
 * @param pKF1
 * @param pKF2
 * @param vpMatches12
 * @param s12
 * @param R12
 * @param t12
 * @param th
 * @return
 */
int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = (int)vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 =(int) vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)
            continue;

        const float invz = 1.f/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = (float)cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(unsigned long idx : vIndices)
        {
            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row((int)idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx =(int) idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.f/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D =(float) cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(unsigned long idx : vIndices)
        {
            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row((int)idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = (int)idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

/**
 * 用于前端跟踪
 * 从LastFrame的MapPoints中找到与CurrentFram匹配的MapPoints，增补CurrentFrame的MapPoints
 * @param CurrentFrame
 * @param LastFrame
 * @param th
 * @param bMono
 * @return
 */
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono) const
{
    int nmatches = 0;

    /// Step 1 建立旋转直方图，用于检测旋转一致性
    vector<int> rotHist[HISTO_LENGTH];
    for(auto &rh : rotHist)
        rh.reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;
    /// Step 2 计算当前帧和前一帧的平移向量
    //当前帧的相机位姿
    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    //当前相机坐标系到世界坐标系的平移向量
    const cv::Mat twc = -Rcw.t()*tcw;
    //上一帧的相机位姿
    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);
    // 当前帧相对于上一帧相机的平移向量
    const cv::Mat tlc = Rlw*twc+tlw;
    // 判断前进还是后退
    const bool bForward = tlc.at<float>(2) > CurrentFrame.mb && !bMono;// 非单目情况，如果Z大于基线，则表示相机明显前进
    const bool bBackward = -tlc.at<float>(2) > CurrentFrame.mb && !bMono;// 非单目情况，如果-Z小于基线，则表示相机明显后退
    ///  Step 3 对于前一帧的每一个地图点，通过相机投影模型，得到投影到当前帧的像素坐标
    for(int i=0; i<LastFrame.N; i++){
        MapPoint* pMP = LastFrame.mvpMapPoints[i];
        if(pMP){
            if(!LastFrame.mvbOutlier[i]){
                // 对上一帧有效的MapPoints投影到当前帧坐标系
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;
                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.f/x3Dc.at<float>(2);
                if(invzc<0)
                    continue;
                // 投影到当前帧中
                float u = Frame::fx*xc*invzc+Frame::cx;
                float v = Frame::fy*yc*invzc+Frame::cy;
                if(u<Frame::mnMinX || u>Frame::mnMaxX)
                    continue;
                if(v<Frame::mnMinY || v>Frame::mnMaxY)
                    continue;
                // 上一帧中地图点对应二维特征点所在的金字塔层级
                int nLastOctave = LastFrame.mvKeys[i].octave;
                // Search in a window. Size depends on scale,尺度越大，搜索范围越大
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];
                vector<size_t> vIndices2;// 记录候选匹配点的id
                /// Step 4 根据相机的前后前进方向来判断搜索尺度范围。
                // 以下可以这么理解，例如一个有一定面积的圆点，在某个尺度n下它是一个特征点
                // 当相机前进时，圆点的面积增大，在某个尺度m下它是一个特征点，由于面积增大，则需要在更高的尺度下才能检测出来
                // 当相机后退时，圆点的面积减小，在某个尺度m下它是一个特征点，由于面积减小，则需要在更低的尺度下才能检测出来
                if(bForward)// 前进,则上一帧兴趣点在所在的尺度nLastOctave<=nCurOctave
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave);
                else if(bBackward) // 后退,则上一帧兴趣点在所在的尺度0<=nCurOctave<=nLastOctave
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave);
                else // 在[nLastOctave-1, nLastOctave+1]中搜索
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();
                int bestDist = 256;
                int bestIdx2 = -1;
                /// Step 5 遍历候选匹配点，寻找距离最小的最佳匹配点
                for(unsigned long i2 : vIndices2){
                    /// 如果该特征点已经有对应的MapPoint了,则退出该次循环
                    if(CurrentFrame.mvpMapPoints[i2]){
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                            continue;
                    }
                    // 双目和rgbd的情况，需要保证右图的点也在搜索半径以内
                    if(CurrentFrame.mvuRight[i2]>0){
                        const float ur = u - CurrentFrame.mbf*invzc;
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)
                            continue;
                    }

                    const cv::Mat &d = CurrentFrame.mDescriptors.row((int)i2);
                    const int dist = DescriptorDistance(dMP,d);//计算距离

                    if(dist<bestDist){
                        bestDist=dist;
                        bestIdx2=(int)i2;
                    }
                }
                // 最佳匹配距离要小于设定阈值
                if(bestDist<=TH_HIGH){
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;
                    /// Step 6 计算匹配点旋转角度差所在的直方图
                    if(mbCheckOrientation){
                        float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = (int)round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }

    ///  Step 7 进行旋转一致检测，剔除不一致的匹配
    if(mbCheckOrientation){
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++){
            if(i!=ind1 && i!=ind2 && i!=ind3){
                for(int j : rotHist[i]){
                    CurrentFrame.mvpMapPoints[j]=static_cast<MapPoint*>(nullptr);
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}



/**
 * 用于重定位
 * 从pKF的MapPoints中找到与CurrentFrame匹配的MapPoints，以增补CurrentFrame的MapPoints
 * @param CurrentFrame
 * @param pKF
 * @param sAlreadyFound
 * @param th
 * @param ORBdist
 * @return
 */
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist) const
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(auto & i : rotHist)
        i.reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP)
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.f/x3Dc.at<float>(2);

                const float u = Frame::fx*xc*invzc+Frame::cx;
                const float v = Frame::fy*yc*invzc+Frame::cy;

                if(u<Frame::mnMinX || u>Frame::mnMaxX)
                    continue;
                if(v<Frame::mnMinY || v>Frame::mnMaxY)
                    continue;

                // Compute predicted scale level
                cv::Mat PO = x3Dw-Ow;
                float dist3D = (float)cv::norm(PO);

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                // Search in a window
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(unsigned long i2 : vIndices2)
                {
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row((int)i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=(int)i2;
                    }
                }

                if(bestDist<=ORBdist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = (int)round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(int j : rotHist[i])
                {
                    CurrentFrame.mvpMapPoints[j]=nullptr;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

/**
 * 获取旋转直方图histo上特征点最多的3个bin的索引
 * @param histo 直方图
 * @param L 直方图的长度
 * @param ind1
 * @param ind2
 * @param ind3
 */
void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++){
        const int s = (int)histo[i].size();
        if(s>max1){
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2){
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3){
            max3=s;
            ind3=i;
        }
    }

    if((float)max2<0.1f*(float)max1){
        ind2=-1;
        ind3=-1;
    }
    else if((float)max3<0.1f*(float)max1){
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
/**
 * 计算两个描述子之间的距离
 * @param a
 * @param b
 * @return
 */
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    //https://blog.csdn.net/w798396217/article/details/93620407
    /*
     * 0x55555555 = 0101 0101 0101 0101 0101 0101 0101 0101 (偶数位为0，奇数位为1）
     * 0x33333333 = 0011 0011 0011 0011 0011 0011 0011 0011 (1和0每隔两位交替出现)
     * 0x0F0F0F0F = 0000 1111 0000 1111 0000 1111 0000 1111 (1和0每隔四位交替出现)
     * 0x01010101 = 0000 0001 0000 0001 0000 0001 0000 0001
     */
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();
    int dist=0;
    for(int i=0; i<8; i++, pa++, pb++){//描述子为256维,计算距离时取32维为一组
        unsigned  int v = *pa ^ *pb;//异或运算,位相同时为0,相异时为1
        v = v - ((v >> 1) & 0x55555555);//将32位分为16组,看每一组中有几个1. 假设某一组为11,则对应输出为10(两个1)
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);//32位分为8组,看的其实还是原来的那个数每8个单位里有几个1.
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;//0x1010101并且>>24的意思是只保留下了原来的后8位
    }
    return dist;
}

} //namespace ORB_SLAM
