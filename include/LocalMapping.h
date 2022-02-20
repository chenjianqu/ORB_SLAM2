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

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Tracking;
class LoopClosing;
class Map;

class LocalMapping
{
public:
    LocalMapping(Map* pMap, const float bMonocular);

    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

protected:

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    static cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

    static cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    bool mbMonocular;// 当前系统输入数单目还是双目RGB-D的标志

    void ResetIfRequested();

    bool mbResetRequested;// 当前系统是否收到了请求复位的信号
    std::mutex mMutexReset;// 和复位信号有关的互斥量

    bool CheckFinish();
    void SetFinish();

    bool mbFinishRequested;// 当前线程是否收到了请求终止的信号
    bool mbFinished;// 当前线程的主函数是否已经终止
    std::mutex mMutexFinish;// 和"线程真正结束"有关的互斥锁

    Map* mpMap;// 指向局部地图的句柄

    LoopClosing* mpLoopCloser{};// 回环检测线程句柄
    Tracking* mpTracker{}; // 追踪线程句柄

    std::list<KeyFrame*> mlNewKeyFrames;//< 等待处理的关键帧列表
    KeyFrame* mpCurrentKeyFrame{};// 当前正在处理的关键帧
    std::list<MapPoint*> mlpRecentAddedMapPoints;// 存储当前关键帧生成的地图点,也是等待检查的地图点列表
    std::mutex mMutexNewKFs;// 操作关键帧列表时使用的互斥量

    bool mbAbortBA;// 终止BA的标志
    bool mbStopped;// 当前线程是否已经真正地终止了
    bool mbStopRequested;// 终止当前线程的请求
    bool mbNotStop;// 标志这当前线程还不能够停止工作,优先级比那个"mbStopRequested"要高.只有这个和mbStopRequested都满足要求的时候,线程才会进行一系列的终止操作
    std::mutex mMutexStop;// 和终止线程相关的互斥锁

    bool mbAcceptKeyFrames;// 当前局部建图线程是否允许关键帧输入
    std::mutex mMutexAccept;// 和操作上面这个变量有关的互斥量
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
