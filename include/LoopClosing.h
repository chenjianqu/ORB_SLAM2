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

#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "KeyFrame.h"
#include "LocalMapping.h"
#include "Map.h"
#include "ORBVocabulary.h"
#include "Tracking.h"

#include "KeyFrameDatabase.h"

#include <thread>
#include <mutex>
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{

class Tracking;
class LocalMapping;
class KeyFrameDatabase;


class LoopClosing
{
public:
    typedef pair<set<KeyFrame*>,int> ConsistentGroup;    
    //typedef map<KeyFrame*,g2o::Sim3,std::less<KeyFrame*>, Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3>>> KeyFrameAndPose;
    typedef map<KeyFrame*,g2o::Sim3,std::less<KeyFrame*>, Eigen::aligned_allocator<std::pair<KeyFrame *const, g2o::Sim3> > > KeyFrameAndPose;

public:

    LoopClosing(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc,const bool bFixScale);

    void SetTracker(Tracking* pTracker);

    void SetLocalMapper(LocalMapping* pLocalMapper);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame *pKF);

    void RequestReset();

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(unsigned long nLoopKF);

    bool isRunningGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }
    bool isFinishedGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }   

    void RequestFinish();

    bool isFinished();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:

    bool CheckNewKeyFrames();

    bool DetectLoop();

    bool ComputeSim3();

    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

    void CorrectLoop();

    void ResetIfRequested();
    bool mbResetRequested;// 是否有复位当前线程的请求
    std::mutex mMutexReset;// 和复位当前线程相关的互斥量

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;// 是否有终止当前线程的请求
    bool mbFinished;// 当前线程是否已经停止工作
    std::mutex mMutexFinish;// 和当前线程终止状态操作有关的互斥量

    Map* mpMap;// (全局)地图的指针
    Tracking* mpTracker{};// 追踪线程句柄

    KeyFrameDatabase* mpKeyFrameDB;// 关键帧数据库
    ORBVocabulary* mpORBVocabulary;// 词袋模型中的大字典

    LocalMapping *mpLocalMapper{};// 局部建图线程句柄
    // 一个队列, 其中存储了参与到回环检测的关键帧 (当然这些关键帧也有可能因为各种原因被设置成为bad,
    // 这样虽然这个关键帧还是存储在这里但是实际上已经不再实质性地参与到回环检测的过程中去了)
    std::list<KeyFrame*> mlpLoopKeyFrameQueue;
    std::mutex mMutexLoopQueue;// 操作参与到回环检测队列中的关键帧时,使用的互斥量

    // Loop detector parameters
    float mnCovisibilityConsistencyTh;// 连续性阈值,构造函数中将其设置成为了3

    // Loop detector variables
    KeyFrame* mpCurrentKF{};// 当前关键帧,其实称之为"当前正在处理的关键帧"更加合适
    KeyFrame* mpMatchedKF;// 最终检测出来的,和当前关键帧形成闭环的闭环关键帧
    std::vector<ConsistentGroup> mvConsistentGroups;// 上一次执行的时候产生的连续组s
    // 从上面的关键帧中进行筛选之后得到的具有足够的"连续性"的关键帧 -- 这个其实也是相当于更高层级的、更加优质的闭环候选帧
    std::vector<KeyFrame*> mvpEnoughConsistentCandidates;
    std::vector<KeyFrame*> mvpCurrentConnectedKFs;// 和当前关键帧相连的关键帧形成的"当前关键帧组"
    std::vector<MapPoint*> mvpCurrentMatchedPoints;// 下面的变量中存储的地图点在"当前关键帧"中成功地找到了匹配点的地图点的集合
    std::vector<MapPoint*> mvpLoopMapPoints;// 闭环关键帧上的所有相连关键帧的地图点
    cv::Mat mScw;
    g2o::Sim3 mg2oScw;// 当得到了当前关键帧的闭环关键帧以后,计算出来的从世界坐标系到当前帧的sim3变换

    long unsigned int mLastLoopKFid;// 上一次闭环帧的id

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA;// 全局BA线程是否在进行
    bool mbFinishedGBA;// 全局BA线程在收到停止请求之后是否停止的比标志
    bool mbStopGBA;// 由当前线程调用,请求停止当前正在进行的全局BA
    std::mutex mMutexGBA;// 在对和全局线程标志量有关的操作的时候使用的互斥量
    std::thread* mpThreadGBA;// 全局BA线程句柄

    // Fix scale in the stereo/RGB-D case
    bool mbFixScale;// 如果是在双目或者是RGBD输入的情况下,就要固定尺度,这个变量就是是否要固定尺度的标志

    bool mnFullBAIdx;// 已经进行了的全局BA次数(包含中途被打断的)
};

} //namespace ORB_SLAM

#endif // LOOPCLOSING_H
