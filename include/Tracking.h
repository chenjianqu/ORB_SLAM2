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


#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"Viewer.h"
#include"FrameDrawer.h"
#include"Map.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include"Frame.h"
#include "ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"

#include <mutex>

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

class Tracking
{  

public:
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp);
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);


public:

    ///跟踪状态类型
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,        ///<系统没有准备好的状态,一般就是在启动后加载配置文件和词典文件时候的状态
        NO_IMAGES_YET=0,            ///<当前无图像
        NOT_INITIALIZED=1,          ///<有图像但是没有完成初始化
        OK=2,                       ///<正常时候的工作状态
        LOST=3                      ///<系统已经跟丢了的状态
    };

    eTrackingState mState;//跟踪状态
    eTrackingState mLastProcessedState;//上一帧的跟踪状态.这个变量在绘制当前帧的时候会被使用到

    //传感器类型
    int mSensor;
    Frame mCurrentFrame;//追踪线程中有一个当前帧
    cv::Mat mImGray;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;//之前的匹配
    std::vector<int> mvIniMatches;//初始化阶段中,当前帧中的特征点和参考帧中的特征点的匹配关系
    std::vector<cv::Point2f> mvbPrevMatched;//在初始化的过程中,保存参考帧中的特征点
    std::vector<cv::Point3f> mvIniP3D;//初始化过程中匹配后进行三角化得到的空间点
    Frame mInitialFrame;//初始化过程中的参考帧

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses; //所有的参考关键帧的位姿;看上面注释的意思,这里存储的也是相对位姿
    list<KeyFrame*> mlpReferences;//参考关键帧
    list<double> mlFrameTimes;//所有帧的时间戳
    list<bool> mlbLost;//是否跟丢的标志

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;//标记当前系统是处于SLAM状态还是纯定位状态

    void Reset();

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;//当进行纯定位时才会有的一个变量,为false表示该帧匹配了很多的地图点,跟踪是正常的;如果少于10个则为true,表示快要完蛋了

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;//用于单目SLAM的初始化,需要检测更多的特征点

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular),单目初始化器
    Initializer* mpInitializer;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    
    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    //Map
    Map* mpMap;

    //Calibration matrix
    cv::Mat mK;//相机内参
    cv::Mat mDistCoef;//矫正系数
    float mbf;//基线长度

    //New KeyFrame rules (according to fps)
    int mMinFrames;// 新建关键帧和重定位中用来判断最小最大时间间隔，和帧率有关
    int mMaxFrames;// 新建关键帧和重定位中用来判断最小最大时间间隔，和帧率有关

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;//用于区分远点和近点的阈值. 近点认为可信度比较高;远点则要求在两个关键帧中得到匹配

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;//深度缩放因子,链接深度值和具体深度值的参数.只对RGBD输入有效

    //Current matches in frame
    int mnMatchesInliers;//当前帧中的进行匹配的内点,将会被不同的函数反复使用

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame; // 上一关键帧
    Frame mLastFrame;// 上一帧
    unsigned int mnLastKeyFrameId;// 上一个关键帧的ID
    unsigned int mnLastRelocFrameId;// 上一次重定位的那一帧的ID

    //Motion Model
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;//RGB图像的颜色通道顺序

    list<MapPoint*> mlpTemporalPoints;//临时的地图点,用于提高双目和RGBD摄像头的帧间效果,用完之后就扔了
};

} //namespace ORB_SLAM

#endif // TRACKING_H
