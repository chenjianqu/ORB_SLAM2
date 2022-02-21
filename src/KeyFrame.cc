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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include<mutex>

namespace ORB_SLAM2{ \

long unsigned int KeyFrame::nNextId=0;

/**
 * @brief 构造函数
 * @param[in] F         父类普通帧的对象
 * @param[in] pMap      所属的地图指针
 * @param[in] pKFDB     使用的词袋模型的指针
 */
KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(Frame::mfGridElementWidthInv), mfGridElementHeightInv(Frame::mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(Frame::fx), fy(Frame::fy), cx(Frame::cx), cy(Frame::cy),invfx(Frame::invfx), invfy(Frame::invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX((int)Frame::mnMinX), mnMinY((int)Frame::mnMinY),mnMaxX((int)Frame::mnMaxX),
    mnMaxY((int)Frame::mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints),
    mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary),
    mbFirstConnection(true), mpParent(nullptr), mbNotErase(false),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2),
    mpMap(pMap)
{
    mnId=nNextId++;
    //其实就把每个网格中有的特征点的索引复制过来
    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++) {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }
    // 设置当前关键帧的位姿
    SetPose(F.mTcw);    
}

/**
 * @brief Bag of Words Representation
 * @detials 计算mBowVec，并且将描述子分散在第4层上，即mFeatVec记录了属于第i个node的ni个描述子
 * @see ProcessNewKeyFrame()
 */
void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty()){
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);//将描述子转换为描述子向量
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        //将描述子向量转换为视觉词汇向量和特征向量
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

/**
 * @brief 设置当前关键帧的位姿
 * @param[in] Tcw 位姿
 */
void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc*tcw;
    /// 计算当前位姿的逆
    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
    // center为相机坐标系（左目）下，立体相机中心的坐标
    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
    // 世界坐标系下，左目相机中心到立体相机中心的向量，方向由左目相机指向立体相机中心
    Cw = Twc*center;
}

/**
 * 获取位姿
 * @return
 */
cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}
/**
 * 获取位姿的逆
 * @return
 */
cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}
/**
 * 获取(左目)相机的中心在世界坐标系下的坐标
 * @return
 */
cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}
/**
 * 获取双目相机的中心,这个只有在可视化的时候才会用到
 * @return
 */
cv::Mat KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}

/**
 * 获取姿态
 * @return
 */
cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}
/**
 * 获取位置
 * @return
 */
cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}


/**
 * @brief 为当前关键帧新建或更新和其他关键帧的连接权重
 *
 * @param[in] pKF       和当前关键帧共视的其他关键帧
 * @param[in] weight    当前关键帧和其他关键帧的权重（共视地图点数目）
 */
void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF)) //新建连接权重
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight) //更新连接权重
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }
    // 连接关系变化就要更新最佳共视，主要是重新进行排序
    UpdateBestCovisibles();
}

/**
 * @brief 按照权重从大到小对连接（共视）的关键帧进行排序
 *
 * 更新后的变量存储在mvpOrderedConnectedKeyFrames和mvOrderedWeights中
 */
void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    // 取出所有连接的关键帧，mConnectedKeyFrameWeights的类型为std::map<KeyFrame*,int>，而vPairs变量将共视的地图点数放在前面，利于排序
    for(auto & ckf : mConnectedKeyFrameWeights)
        vPairs.emplace_back(ckf.second,ckf.first);
    // 按照权重进行排序（默认是从小到大）
    std::sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(auto & vPair : vPairs){
        lKFs.push_front(vPair.second);// push_front 后变成从大到小
        lWs.push_front(vPair.first);
    }
    // 权重从大到小排列的连接关键帧
    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    // 从大到小排列的权重，和mvpOrderedConnectedKeyFrames一一对应
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());    
}

set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(auto & mConnectedKeyFrameWeight : mConnectedKeyFrameWeights)
        s.insert(mConnectedKeyFrameWeight.first);
    return s;
}

/**
 * 得到与该关键帧连接的关键帧(已按权值排序)
 * @return
 */
vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

/**
 * @brief 得到与该关键帧连接的前N个最强共视关键帧(已按权值排序)
 *
 * @param[in] N                 设定要取出的关键帧数目
 * @return vector<KeyFrame*>    满足权重条件的关键帧集合
 */
vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return {};

    auto it = std::upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
    if(it==mvOrderedWeights.end())
        return {};
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

/**
 * @brief Add MapPoint to KeyFrame
 * @param pMP MapPoint
 * @param idx MapPoint在KeyFrame中的索引
 */
void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(nullptr);
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapPoints[idx]=static_cast<MapPoint*>(nullptr);
}


void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(auto & mvpMapPoint : mvpMapPoints)
    {
        if(!mvpMapPoint)
            continue;
        MapPoint* pMP = mvpMapPoint;
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

/**
 * @brief 关键帧中，大于等于minObs的MapPoints的数量
 * @details minObs就是一个阈值，大于minObs就表示该MapPoint是一个高质量的MapPoint \n
 * 一个高质量的MapPoint会被多个KeyFrame观测到.
 * @param  minObs 最小观测
 */
int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++){
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP){
            if(!pMP->isBad()){
                if(bCheckObs){
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else{
                    nPoints++;
                }
            }
        }
    }

    return nPoints;
}

/**
 * @brief Get MapPoint Matches 获取该关键帧的MapPoints
 */
vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

/**
 * 更新关键帧之间的连接图
 *
 * 1. 首先获得该关键帧的所有MapPoint点，统计观测到这些3d点的每个关键帧与其它所有关键帧之间的共视程度
 *    对每一个找到的关键帧，建立一条边，边的权重是该关键帧与当前关键帧公共3d点的个数。
 * 2. 并且该权重必须大于一个阈值，如果没有超过该阈值的权重，那么就只保留权重最大的边（与其它关键帧的共视程度比较高）
 * 3. 对这些连接按照权重从大到小进行排序，以方便将来的处理
 *    更新完covisibility图之后，如果没有初始化过，则初始化为连接权重最大的边（与其它关键帧共视程度最高的那个关键帧），类似于最大生成树
 */
void KeyFrame::UpdateConnections()
{
    // 获得该关键帧的所有地图点
    vector<MapPoint*> vpMP;{
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    /// Step 1 通过地图点被关键帧观测来间接统计关键帧之间的共视程度

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    map<KeyFrame*,int> KFcounter;// 关键帧-权重，权重为其它关键帧与当前关键帧共视地图点的个数，也称为共视程度
    // 统计每一个地图点都有多少关键帧与当前关键帧存在共视关系，统计结果放在KFcounter
    for(auto pMP : vpMP)
    {
        if(!pMP) continue;
        if(pMP->isBad())
            continue;
        // 对于每一个地图点，observations记录了可以观测到该地图点的所有关键帧
        map<KeyFrame*,size_t> observations = pMP->GetObservations();
        for(auto & observation : observations) {
            if(observation.first->mnId==mnId) continue;// 除去自身，自己与自己不算共视
            KFcounter[observation.first]++;
        }
    }
    // This should not happen
    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0; // 记录最高的共视程度
    KeyFrame* pKFmax=nullptr;
    int th = 15; // 至少有15个共视地图点才会添加共视关系
    vector<pair<int,KeyFrame*> > vPairs;// vPairs记录与其它关键帧共视帧数大于th的关键帧
    vPairs.reserve(KFcounter.size());
    /// Step 2 找到对应权重最大的关键帧（共视程度最高的关键帧）
    for(auto & mit : KFcounter) {
        if(mit.second>nmax) {
            nmax=mit.second;
            pKFmax=mit.first;
        }
        if(mit.second>=th) {  // 对应权重需要大于阈值，对这些关键帧建立连接
            vPairs.emplace_back(mit.second,mit.first);
            (mit.first)->AddConnection(this,mit.second);// 对方关键帧也要添加这个信息
        }
    }
    ///  Step 3 如果没有超过阈值的权重，则对权重最大的关键帧建立连接
    if(vPairs.empty()) {
        vPairs.emplace_back(nmax,pKFmax);
        pKFmax->AddConnection(this,nmax);
    }
    ///  Step 4 对满足共视程度的关键帧对更新连接关系及权重（从大到小）
    std::sort(vPairs.begin(),vPairs.end());//从小到大排序
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(auto & vPair : vPairs) {
        lKFs.push_front(vPair.second); // push_front 后变成了从大到小顺序
        lWs.push_front(vPair.first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);
        // mspConnectedKeyFrames = spConnectedKeyFrames;
        // 更新当前帧与其它关键帧的连接权重
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
        /// Step 5 更新生成树的连接
        if(mbFirstConnection && mnId!=0)
        {
            mpParent = mvpOrderedConnectedKeyFrames.front(); // 初始化该关键帧的父关键帧为共视程度最高的那个关键帧
            mpParent->AddChild(this);// 建立双向连接关系，将当前关键帧作为其子关键帧
            mbFirstConnection = false;
        }

    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

/**
 * 获取当前关键帧的子关键帧
 * @return
 */
set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}
/**
 * 获取当前关键帧的父关键帧
 * @return
 */
KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

/**
 * @brief 删除当前的这个关键帧,表示不进行回环检测过程;由回环检测线程调用
 *
 */
void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
            mbNotErase = false;
    }
    if(mbToBeErased){
        SetBadFlag();
    }
}

void KeyFrame::SetBadFlag()
{   
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==0)
            return;
        else if(mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for(auto & mConnectedKeyFrameWeight : mConnectedKeyFrameWeights)
        mConnectedKeyFrameWeight.first->EraseConnection(this);

    for(auto & mvpMapPoint : mvpMapPoints)
        if(mvpMapPoint)
            mvpMapPoint->EraseObservation(this);
    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;

            for(auto pKF : mspChildrens)
            {
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(auto & i : vpConnected)
                {
                    for(auto sParentCandidate : sParentCandidates)
                    {
                        if(i->mnId == sParentCandidate->mnId)
                        {
                            int w = pKF->GetWeight(i);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = i;
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if(!mspChildrens.empty())
            for(auto mspChildren : mspChildrens)
            {
                mspChildren->ChangeParent(mpParent);
            }

        mpParent->EraseChild(this);
        mTcp = Tcw*mpParent->GetPoseInverse();
        mbBad = true;
    }


    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}

vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            for(unsigned long j : vCell)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[j];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(j);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

cv::Mat KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return {};
}

/**
 * // Compute Scene Depth (q=2 median). Used in monocular. 评估当前关键帧场景深度，q=2表示中值. 只是在单目情况下才会使用
// 其实过程就是对当前关键帧下所有地图点的深度进行从小到大排序,返回距离头部其中1/q处的深度值作为当前场景的平均深度
 * @param q
 * @return
 */
float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2,3);
    for(int i=0; i<N; i++){
        if(mvpMapPoints[i]){
            MapPoint* pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = (float)Rcw2.dot(x3Dw)+zcw;
            vDepths.push_back(z);
        }
    }

    std::sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}

} //namespace ORB_SLAM
