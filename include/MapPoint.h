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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"

#include<opencv2/core/core.hpp>
#include<mutex>

namespace ORB_SLAM2
{

class KeyFrame;
class Map;
class Frame;


class MapPoint
{
public:
    MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
    MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);

    void SetWorldPos(const cv::Mat &Pos);
    cv::Mat GetWorldPos();

    cv::Mat GetNormal();
    KeyFrame* GetReferenceKeyFrame();

    std::map<KeyFrame*,size_t> GetObservations();
    int Observations();

    void AddObservation(KeyFrame* pKF,size_t idx);
    void EraseObservation(KeyFrame* pKF);

    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapPoint* pMP);    
    MapPoint* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, KeyFrame*pKF);
    int PredictScale(const float &currentDist, Frame* pF);

public:
    long unsigned int mnId;// Global ID for MapPoint
    static long unsigned int nNextId;//用于创建地图点的全局ID
    long int mnFirstKFid;// 创建该MapPoint的关键帧ID
    long int mnFirstFrame;// 创建该MapPoint的帧ID（即每一关键帧有一个帧ID）,如果是从帧中创建的话,会将普通帧的id存放于这里
    int nObs;// 观测到该地图点的相机数目，单目+1，双目或RGB-D则+2

    // Variables used by the tracking
    float mTrackProjX{};//< 当前地图点投影到某帧上后的坐标
    float mTrackProjY{};//< 当前地图点投影到某帧上后的坐标
    float mTrackProjXR{};//< 当前地图点投影到某帧上后的坐标(右目)
    // TrackLocalMap - SearchByProjection 中决定是否对该点进行投影的变量
    //NOTICE mbTrackInView==false的点有几种：
    // a 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
    // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
    // c 不在当前相机视野中的点（即未通过isInFrustum判断）
    bool mbTrackInView{};
    int mnTrackScaleLevel{};// 根据地图点到光心距离，预测的该地图点的尺度层级
    float mTrackViewCos{};//< 被追踪到时,那帧相机看到当前地图点的视角
    // TrackLocalMap - UpdateLocalPoints 中防止将MapPoints重复添加至mvpLocalMapPoints的标记
    long unsigned int mnTrackReferenceForFrame;
    // TrackLocalMap - SearchLocalPoints 中决定是否进行isInFrustum判断的变量
    // NOTICE mnLastFrameSeen==mCurrentFrame.mnId的点有几种：
    // a 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
    // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
    long unsigned int mnLastFrameSeen;

    // Variables used by local mapping
    // local mapping中记录地图点对应当前局部BA的关键帧的mnId。mnBALocalForKF 在map point.h里面也有同名的变量。
    long unsigned int mnBALocalForKF;
    //< 在局部建图线程中使用,表示被用来进行地图点融合的关键帧(存储的是这个关键帧的id)
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    // 标记当前地图点是作为哪个"当前关键帧"的回环地图点(即回环关键帧上的地图点),在回环检测线程中被调用
    long unsigned int mnLoopPointForKF;
    // 如果这个地图点对应的关键帧参与到了回环检测的过程中,那么在回环检测过程中已经使用了这个关键帧修正只有的位姿来修正了这个地图点,那么这个标志位置位
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    // 全局BA优化后(如果当前地图点参加了的话),这里记录优化后的位姿
    cv::Mat mPosGBA;
    // 如果当前点的位姿参与到了全局BA优化,那么这个变量记录了那个引起全局BA的"当前关键帧"的id
    long unsigned int mnBAGlobalForKF;
    //全局BA中对当前点进行操作的时候使用的互斥量
    static std::mutex mGlobalMutex;

protected:    
     // Position in absolute coordinates
     cv::Mat mWorldPos;//< MapPoint在世界坐标系下的坐标
     // Keyframes observing the point and associated index in keyframe
     std::map<KeyFrame*,size_t> mObservations;// 观测到该MapPoint的KF和该MapPoint在KF中的索引
     // Mean viewing direction
     cv::Mat mNormalVector;// 该MapPoint平均观测方向,用于判断点是否在可视范围内

     // Best descriptor to fast matching
     // 每个3D点也有一个描述子，但是这个3D点可以观测多个二维特征点，从中选择一个最有代表性的
     //通过 ComputeDistinctiveDescriptors() 得到的最有代表性描述子,距离其它描述子的平均距离最小
     cv::Mat mDescriptor;

     // Reference KeyFrame
     // 通常情况下MapPoint的参考关键帧就是创建该MapPoint的那个关键帧
     KeyFrame* mpRefKF;

     // Tracking counters
     int mnVisible;//在帧中的可视次数
     int mnFound;//被找到的次数 和上面的相比要求能够匹配上

     // Bad flag (we do not currently erase MapPoint from memory)
     bool mbBad;
     MapPoint* mpReplaced;

     // Scale invariance distances
     float mfMinDistance;//当前地图点在某帧下,可信赖的被找到时其到关键帧光心距离的下界
     float mfMaxDistance;//当前地图点在某帧下,可信赖的被找到时其到关键帧光心距离的上界

     Map* mpMap;//所属的地图

     std::mutex mMutexPos;//对当前地图点位姿进行操作的时候的互斥量
     std::mutex mMutexFeatures;//对当前地图点的特征信息进行操作的时候的互斥量
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
