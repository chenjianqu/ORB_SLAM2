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

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2{ \

class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;


/**
 * @brief 关键帧类
 * @detials 关键帧，和普通的Frame不一样，但是可以由Frame来构造; 许多数据会被三个线程同时访问，所以用锁的地方很普遍
 *
 */
class KeyFrame
{
public:
    KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);

    // Pose functions
    void SetPose(const cv::Mat &Tcw);
    cv::Mat GetPose();
    cv::Mat GetPoseInverse();
    cv::Mat GetCameraCenter();
    cv::Mat GetStereoCenter();
    cv::Mat GetRotation();
    cv::Mat GetTranslation();

    // Bag of Words Representation
    void ComputeBoW();

    // Covisibility graph functions
    void AddConnection(KeyFrame* pKF, const int &weight);
    void EraseConnection(KeyFrame* pKF);
    void UpdateConnections();
    void UpdateBestCovisibles();
    std::set<KeyFrame *> GetConnectedKeyFrames();
    std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);
    int GetWeight(KeyFrame* pKF);

    // Spanning tree functions
    void AddChild(KeyFrame* pKF);
    void EraseChild(KeyFrame* pKF);
    void ChangeParent(KeyFrame* pKF);
    std::set<KeyFrame*> GetChilds();
    KeyFrame* GetParent();
    bool hasChild(KeyFrame* pKF);

    // Loop Edges
    void AddLoopEdge(KeyFrame* pKF);
    std::set<KeyFrame*> GetLoopEdges();

    // MapPoint observation functions
    void AddMapPoint(MapPoint* pMP, const size_t &idx);
    void EraseMapPointMatch(const size_t &idx);
    void EraseMapPointMatch(MapPoint* pMP);
    void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);
    std::set<MapPoint*> GetMapPoints();
    std::vector<MapPoint*> GetMapPointMatches();
    int TrackedMapPoints(const int &minObs);
    MapPoint* GetMapPoint(const size_t &idx);

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
    cv::Mat UnprojectStereo(int i);

    // Image
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    // Compute Scene Depth (q=2 median). Used in monocular.
    float ComputeSceneMedianDepth(const int q);

    static bool weightComp( int a, int b){
        return a>b;
    }

    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2){
        return pKF1->mnId<pKF2->mnId;
    }


    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:
    static long unsigned int nNextId;//表示上一个KeyFrame的ID号
    long unsigned int mnId;//在nNextID的基础上加1就得到了mnID，为当前KeyFrame的ID号
    //每个KeyFrame基本属性是它是一个Frame，KeyFrame初始化的时候需要Frame，mnFrameId记录了该KeyFrame是由哪个Frame初始化的
    const long unsigned int mnFrameId;
    const double mTimeStamp;// 时间戳
    // Grid (to speed up feature matching)
    const int mnGridCols;//与Frame类中的定义一致,网格的列数, 用于加速匹配
    const int mnGridRows;//与Frame类中的定义一致,网格的行数, 用于加速匹配
    const float mfGridElementWidthInv;//与Frame类中的定义一致,网格宽度的逆
    const float mfGridElementHeightInv;//与Frame类中的定义一致,网格高度的逆

    // Variables used by the tracking
    // 用该关键帧的成员变量mnTrackReferenceForFrame 记录当前帧的id,表示它已经是当前帧的局部关键帧了，可以防止重复添加局部关键帧
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnFuseTargetForKF;//< 标记在局部建图线程中,和哪个关键帧进行融合的操作

    // Variables used by the local mapping
    // local mapping中记录当前处理的关键帧的mnId，表示当前局部BA的关键帧id。mnBALocalForKF 在map point.h里面也有同名的变量。
    long unsigned int mnBALocalForKF;
    // local mapping中记录当前处理的关键帧的mnId, 只是提供约束信息但是却不会去优化这个关键帧
    long unsigned int mnBAFixedForKF;

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;// 标记了当前关键帧是id为mnLoopQuery的回环检测的候选关键帧
    int mnLoopWords;// 当前关键帧和这个形成回环的候选关键帧中,具有相同word的个数
    float mLoopScore{};// 和那个形成回环的关键帧的词袋匹配程度的评分
    long unsigned int mnRelocQuery;// 用来存储在辅助进行重定位的时候，要进行重定位的那个帧的id
    int mnRelocWords;// 和那个要进行重定位的帧,所具有相同的单词的个数
    float mRelocScore{};// 还有和那个帧的词袋的相似程度的评分

    // Variables used by loop closing
    cv::Mat mTcwGBA;// 经过全局BA优化后的相机的位姿
    // 进行全局BA优化之前的当前关键帧的位姿. 之所以要记录这个是因为在全局优化之后还要根据该关键帧在优化之前的位姿来更新地图点,which地图点的参考关键帧就是该关键帧
    cv::Mat mTcwBefGBA;
    // 记录是由于哪个"当前关键帧"触发的全局BA,用来防止重复写入的事情发生(浪费时间)
    long unsigned int mnBAGlobalForKF;

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;//相机内参

    // Number of KeyPoints
    const int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    const std::vector<cv::KeyPoint> mvKeys;// 和Frame类中的定义相同
    const std::vector<cv::KeyPoint> mvKeysUn;// 和Frame类中的定义相同
    const std::vector<float> mvuRight; // negative value for monocular points
    const std::vector<float> mvDepth; // negative value for monocular points
    const cv::Mat mDescriptors;

    // BowVector 是描述一张图像的一系列视觉词汇的ID,和它的权重值
    // 内部实际存储的是std::map<WordId, WordValue>, 其中WordId 和 WordValue 表示Word在叶子中的id 和权重
    DBoW2::BowVector mBowVec;
    // FeatureVector是节点的ID和节点拥有的特征在图像中的索引
    // 内部实际存储 std::map<NodeId, std::vector<unsigned int>>, 其中NodeId 表示节点id，std::vector<unsigned int> 是该节点id下所有特征点在图像中的索引
    DBoW2::FeatureVector mFeatVec;

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp;

    // Scale
    const int mnScaleLevels;//<图像金字塔的层数
    const float mfScaleFactor;//<图像金字塔的尺度因子
    const float mfLogScaleFactor;//<图像金字塔的尺度因子的对数值，用于仿照特征点尺度预测地图点的尺度
    const std::vector<float> mvScaleFactors;//<图像金字塔每一层的缩放因子 尺度因子，scale^n，scale=1.2，n为层数
    const std::vector<float> mvLevelSigma2;// 尺度因子的平方
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    const int mnMinX;//与Frame类中的一致
    const int mnMinY;//与Frame类中的一致
    const int mnMaxX;//与Frame类中的一致
    const int mnMaxY;//与Frame类中的一致
    const cv::Mat mK;//与Frame类中的一致


    // The following variables need to be accessed trough a mutex to be thread safe.
protected:

    // SE3 Pose and camera center
    cv::Mat Tcw;// 当前相机的位姿，世界坐标系到相机坐标系
    cv::Mat Twc;// 当前相机位姿的逆
    cv::Mat Ow;// 相机光心(左目)在世界坐标系下的坐标,这里和普通帧中的定义是一样的,用于计算向量
    cv::Mat Cw; // Stereo middel point. Only for visualization

    // MapPoints associated to keypoints
    std::vector<MapPoint*> mvpMapPoints;

    // BoW
    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBvocabulary;/// 词袋对象

    // Grid over the image to speed up feature matching
    //其实应该说是二维的,第三维的 vector中保存的是这个网格内的特征点的索引
    std::vector< std::vector <std::vector<size_t> > > mGrid;

    std::map<KeyFrame*,int> mConnectedKeyFrameWeights;     // 与该关键帧连接（至少15个共视地图点）的关键帧与权重
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;    // 共视关键帧中权重从大到小排序后的关键帧
    std::vector<int> mvOrderedWeights;    // 共视关键帧中从大到小排序后的权重，和上面对应

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;// 是否是第一次生成树
    KeyFrame* mpParent;// 当前关键帧的父关键帧 （共视程度最高的）
    std::set<KeyFrame*> mspChildrens;// 存储当前关键帧的子关键帧
    std::set<KeyFrame*> mspLoopEdges;// 和当前关键帧形成回环关系的关键帧

    // Bad flags
    bool mbNotErase;//当前关键帧已经和其他的关键帧形成了回环关系，因此在各种优化的过程中不应该被删除
    bool mbToBeErased;
    bool mbBad;    

    float mHalfBaseline; ///< 对于双目相机来说,双目相机基线长度的一半. Only for visualization

    Map* mpMap;

    std::mutex mMutexPose;    /// 在对位姿进行操作时相关的互斥锁
    std::mutex mMutexConnections;    /// 在操作当前关键帧和其他关键帧的公式关系的时候使用到的互斥锁
    std::mutex mMutexFeatures;    /// 在操作和特征点有关的变量的时候的互斥锁
};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
