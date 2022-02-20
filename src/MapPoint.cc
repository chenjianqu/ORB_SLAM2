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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM2{ \


long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;

/**
 * @brief Construct a new Map Point:: Map Point object
 *
 * @param[in] Pos           MapPoint的坐标（世界坐标系）
 * @param[in] pRefKF        关键帧
 * @param[in] pMap          地图
 */
MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(nullptr)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    ///平均观测方向初始化为0
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);
    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(nullptr)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(nullptr), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow = pFrame->GetCameraCenter();
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector/cv::norm(mNormalVector);

    cv::Mat PC = Pos - Ow;
    const float dist = (float)cv::norm(PC);
    const int level = pFrame->mvKeysUn[idxF].octave;
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

cv::Mat MapPoint::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

/**
 * @brief 获取当前地图点的平均观测方向
 * @return cv::Mat 一个向量
 */
cv::Mat MapPoint::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}

KeyFrame* MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

/**
 * @brief 给地图点添加观测
 *
 * 记录哪些 KeyFrame 的那个特征点能观测到该 地图点
 * 并增加观测的相机数目nObs，单目+1，双目或者rgbd+2
 * 这个函数是建立关键帧共视关系的核心函数，能共同观测到某些地图点的关键帧是共视关键帧
 * @param pKF KeyFrame
 * @param idx MapPoint在KeyFrame中的索引
 */
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return;
    // 如果没有添加过观测，记录下能观测到该MapPoint的KF和该MapPoint在KF中的索引
    mObservations[pKF]=idx;

    if(pKF->mvuRight[idx]>=0) // 双目或者rgbd
        nObs+=2;
    else // 单目
        nObs++;
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            int idx = mObservations[pKF];
            if(pKF->mvuRight[idx]>=0)
                nObs-=2;
            else
                nObs--;

            mObservations.erase(pKF);

            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)
        SetBadFlag();
}

map<KeyFrame*, size_t> MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

void MapPoint::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;
        mObservations.clear();
    }
    for(auto & ob : obs){
        KeyFrame* pKF = ob.first;
        pKF->EraseMapPointMatch(ob.second);
    }

    mpMap->EraseMapPoint(this);
}

/**
 * @brief 获取取代当前地图点的点,
 *
 * @return
 */
MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

/**
 * @brief 替换地图点，更新观测关系
 *
 * @param[in] pMP    用该地图点来替换当前地图点
 */
void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        mbBad=true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    for(auto & ob : obs)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = ob.first;

        if(!pMP->IsInKeyFrame(pKF))
        {
            pKF->ReplaceMapPointMatch(ob.second, pMP);
            pMP->AddObservation(pKF,ob.second);
        }
        else
        {
            pKF->EraseMapPointMatch(ob.second);
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);
}

bool MapPoint::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}

void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

/**
 * 计算被找到的比例
 * @return
 */
float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (float)mnFound/(float)mnVisible;
}

/**
 * @brief 计算地图点最具代表性的描述子
 *
 * 由于一个地图点会被许多相机观测到，因此在插入关键帧后，需要判断是否更新代表当前点的描述子
 * 先获得当前点的所有描述子，然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小的距离中值
 */
void MapPoint::ComputeDistinctiveDescriptors()
{

    /// Step 1 获取该地图点所有有效的观测关键帧信息
    map<KeyFrame*,size_t> observations;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations=mObservations;
    }
    if(observations.empty())
        return;
    /// Step 2 遍历观测到该地图点的所有关键帧，对应的orb描述子，放到向量vDescriptors中
    vector<cv::Mat> vDescriptors;
    vDescriptors.reserve(observations.size());
    for(auto & observation : observations) {
        // mit->first取观测到该地图点的关键帧, mit->second取该地图点在关键帧中的索引
        KeyFrame* pKF = observation.first;
        if(!pKF->isBad())
            vDescriptors.push_back(pKF->mDescriptors.row((int)observation.second));
    }
    if(vDescriptors.empty())
        return;

    /// Step 3 计算这些描述子两两之间的距离
    const size_t N = vDescriptors.size();
    float Distances[N][N];
    for(size_t i=0;i<N;i++) {
        Distances[i][i]=0; // 和自己的距离当然是0
        for(size_t j=i+1;j<N;j++){
            //计算两个描述子之间的距离
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=(float)distij;
            Distances[j][i]=(float)distij;
        }
    }
    /// Step 4 选择最有代表性的描述子，它与其他描述子应该具有最小的距离中值
    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX; // 记录最小的中值
    int BestIdx = 0; // 最小中值对应的索引
    for(int i=0;i<N;i++) {
        vector<int> vDists(Distances[i],Distances[i]+N); // 第i个描述子到其它所有描述子之间的距离
        std::sort(vDists.begin(),vDists.end());//排序
        int median = vDists[int(0.5f*float(N-1))];// 获得中值
        if(median<BestMedian) {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return (int)mObservations[pKF];
    else
        return -1;
}

/**
 * @brief 检查该地图点是否在关键帧中（有对应的二维特征点）
 *
 * @param[in] pKF       关键帧
 * @return true         如果能够观测到，返回true
 * @return false        如果观测不到，返回false
 */
bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

/**
 * @brief 更新地图点的平均观测方向、观测距离范围
 *
 */
void MapPoint::UpdateNormalAndDepth()
{
    /// Step 1 获得观测到该地图点的所有关键帧、坐标等信息
    map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;
        observations=mObservations;// 获得观测到该地图点的所有关键帧
        pRefKF=mpRefKF;// 观测到该点的参考关键帧（第一次创建时的关键帧）
        Pos = mWorldPos.clone();// 地图点在世界坐标系中的位置
    }
    if(observations.empty())
        return;

    /// Step 2 计算该地图点的平均观测方向
    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;
    for(auto & observation : observations)
    {
        KeyFrame* pKF = observation.first;
        cv::Mat Owi = pKF->GetCameraCenter();
        // 获得地图点和观测到它关键帧的向量并归一化
        cv::Mat normali = mWorldPos - Owi;
        normal = normal + normali/cv::norm(normali);
        n++;
    }

    cv::Mat PC = Pos - pRefKF->GetCameraCenter();// 参考关键帧相机指向地图点的向量（在世界坐标系下的表示）
    const float dist = (float)cv::norm(PC);// 该点到参考关键帧相机的距离
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;// 观测到该地图点的当前帧的特征点在金字塔的第几层
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];// 当前金字塔层对应的尺度因子，scale^n，scale=1.2，n为层数
    const int nLevels = pRefKF->mnScaleLevels; // 金字塔总层数，默认为8

    {
        unique_lock<mutex> lock3(mMutexPos);
        mfMaxDistance = dist*levelScaleFactor;// 观测到该点的距离上限
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];// 观测到该点的距离下限
        mNormalVector = normal/n;// 获得地图点平均的观测方向
    }
}

float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}

// 下图中横线的大小表示不同图层图像上的一个像素表示的真实物理空间中的大小
//              ____
// Nearer      /____\     level:n-1 --> dmin
//            /______\                       d/dmin = 1.2^(n-1-m)
//           /________\   level:m   --> d
//          /__________\                     dmax/d = 1.2^m
// Farther /____________\ level:0   --> dmax
//
//           log(dmax/d)
// m = ceil(------------)
//            log(1.2)
// 这个函数的作用:
// 在进行投影匹配的时候会给定特征点的搜索范围,考虑到处于不同尺度(也就是距离相机远近,位于图像金字塔中不同图层)的特征点受到相机旋转的影响不同,
// 因此会希望距离相机近的点的搜索范围更大一点,距离相机更远的点的搜索范围更小一点,所以要在这里,根据点到关键帧/帧的距离来估计它在当前的关键帧/帧中,
// 会大概处于哪个尺度

/**
 * @brief 预测地图点对应特征点所在的图像金字塔尺度层数. 局部地图中,需要将局部地图点投影到当前帧中,所以需要进行一个地图点和特征点的匹配.为了加速
 * 匹配,要根据地图点到当前帧的距离来预测对应特征点可能在哪一层金字塔上被提取出来
 *
 * @param[in] currentDist   相机光心距离地图点距离
 * @param[in] pKF           关键帧
 * @return int              预测的金字塔尺度
 */
int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        // mfMaxDistance = ref_dist*levelScaleFactor 为参考帧考虑上尺度后的距离
        //则ratio = mfMaxDistance/currentDist = ref_dist/cur_dist
        ratio = mfMaxDistance/currentDist;
    }
    // 取对数
    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}



} //namespace ORB_SLAM
