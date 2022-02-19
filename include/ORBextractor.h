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

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>


namespace ORB_SLAM2
{

class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;//该节点存放的关键点
    cv::Point2i UL, UR, BL, BR;//节点的左上角、右上角、左下角、右下角的节点。
    std::list<ExtractorNode>::iterator lit;//指向处于链表中的自己
    bool bNoMore;//表示该节点不再分裂
};

class ORBextractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~ORBextractor()= default;

    /**
     *Compute the ORB features and descriptors on an image.
        ORB are dispersed on the image using an octree.
        Mask is ignored in the current implementation.
        ORB特征检测的主函数
     * @param image
     * @param mask
     * @param keypoints
     * @param descriptors
     */
    void operator()( cv::InputArray image, cv::InputArray mask,
      std::vector<cv::KeyPoint>& keypoints,
      cv::OutputArray descriptors);

    int inline GetLevels() const{
        return nlevels;
    }

    float inline GetScaleFactor() const{
        return scaleFactor;
    }

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

protected:

    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level) const;

    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    std::vector<cv::Point> pattern;

    int nfeatures;//要检测的特征数量
    double scaleFactor;//图像金字塔的尺度系数，或者说放大系数，比如scaleFactor=1.2
    int nlevels;//图像金字塔的层数
    int iniThFAST;//检测FAST关键点时，使用的阈值
    int minThFAST;//再次检测FAST关键点时，使用的阈值；

    std::vector<int> mnFeaturesPerLevel;//金字塔每层要检测的特征数量

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;//每层的放大系数
    std::vector<float> mvInvScaleFactor;//每一层的缩小系数
    std::vector<float> mvLevelSigma2;//每层放大系数的平方
    std::vector<float> mvInvLevelSigma2;//每层缩小系数的平方
};

} //namespace ORB_SLAM

#endif

