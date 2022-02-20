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

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include <set>

#include <mutex>



namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;

class Map
{
public:
    Map();

    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    void EraseMapPoint(MapPoint* pMP);
    void EraseKeyFrame(KeyFrame* pKF);
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();

    long unsigned int MapPointsInMap();
    long unsigned  KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void clear();

    // 保存了最初始的关键帧
    vector<KeyFrame*> mvpKeyFrameOrigins;
    //当更新地图时的互斥量.回环检测中和局部BA后更新全局地图的时候会用到这个
    std::mutex mMutexMapUpdate;
    // This avoid that two points are created simultaneously in separate threads (id conflict)
    //为了避免地图点id冲突设计的互斥量
    std::mutex mMutexPointCreation;

protected:
    // 存储所有的地图点
    std::set<MapPoint*> mspMapPoints;
    // 存储所有的关键帧
    std::set<KeyFrame*> mspKeyFrames;
    //参考地图点
    std::vector<MapPoint*> mvpReferenceMapPoints;
    //当前地图中具有最大ID的关键帧
    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;
    //类的成员函数在对类成员变量进行操作的时候,防止冲突的互斥量
    std::mutex mMutexMap;
};

} //namespace ORB_SLAM

#endif // MAP_H
