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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM2
{

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(nullptr), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(nullptr), mbFixScale(bFixScale), mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

/**
 * 回环线程主函数
 */
void LoopClosing::Run()
{
    mbFinished =false;
    while(true){
        // Loopclosing中的关键帧是LocalMapping发送过来的，LocalMapping是Tracking中发过来的
        // 在LocalMapping中通过 InsertKeyFrame 将关键帧插入闭环检测队列mlpLoopKeyFrameQueue
        /// Step 1 查看闭环检测队列mlpLoopKeyFrameQueue中有没有关键帧进来
        if(CheckNewKeyFrames()){
            // Detect loop candidates and check covisibility consistency
            if(DetectLoop()) {
               // Compute similarity transformation [sR|t]
               // In the stereo/RGBD case s=1
               if(ComputeSim3())
               {
                   // Perform loop fusion and pose graph optimization
                   CorrectLoop();
               }
            }
        }       

        ResetIfRequested();

        if(CheckFinish())
            break;

        //usleep(5000);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    SetFinish();
}

/**
 * 将某个关键帧加入到回环检测的过程中,由局部建图线程调用
 * @param pKF
 */
void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}


/**
 * @brief 闭环检测
 *
 * @return true             成功检测到闭环
 * @return false            未检测到闭环
 */
bool LoopClosing::DetectLoop()
{
    /// Step 1 从队列中取出一个关键帧,作为当前检测闭环关键帧
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();// 设置当前关键帧不要在优化的过程中被删除
    }
    /// Step 2：如果距离上次闭环没多久（小于10帧），或者map中关键帧总共还没有10帧，则不进行闭环检测
    // 后者的体现是当mLastLoopKFid为0的时候
    if(mpCurrentKF->mnId < mLastLoopKFid+10){
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    /// Step 3：遍历当前回环关键帧所有连接（>15个共视地图点）关键帧，计算当前关键帧与每个共视关键的bow相似度得分，并得到最低得分minScore
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
    for(auto pKF : vpConnectedKeyFrames){
        if(pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;
        auto score =(float) mpORBVocabulary->score(CurrentBowVec, BowVec); // 计算两个关键帧的相似度得分；得分越低,相似度越低
        if(score<minScore)
            minScore = score;
    }
    // Query the database imposing the minimum score
    /// Step 4：在所有关键帧中找出闭环候选帧（注意不和当前帧连接）
    // minScore的作用：认为和当前关键帧具有回环关系的关键帧,不应该低于当前关键帧的相邻关键帧的最低的相似度minScore
    // 得到的这些关键帧,和当前关键帧具有较多的公共单词,并且相似度评分都挺高
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);
    if(vpCandidateKFs.empty()){ // 如果没有闭环候选帧，返回false
        mpKeyFrameDB->add(mpCurrentKF);
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    /// Step 5：在候选帧中检测具有连续性的候选帧
    // 1、每个候选帧将与自己相连的关键帧构成一个“子候选组spCandidateGroup”， vpCandidateKFs-->spCandidateGroup
    // 2、检测“子候选组”中每一个关键帧是否存在于“连续组”，如果存在 nCurrentConsistency++，则将该“子候选组”放入“当前连续组vCurrentConsistentGroups”
    // 3、如果nCurrentConsistency大于等于3，那么该”子候选组“代表的候选帧过关，进入mvpEnoughConsistentCandidates

    // 相关的概念说明:（为方便理解，见视频里的图示）
    // 组(group): 对于某个关键帧, 其和其具有共视关系的关键帧组成了一个"组";
    // 子候选组(CandidateGroup): 对于某个候选的回环关键帧, 其和其具有共视关系的关键帧组成的一个"组";
    // 连续(Consistent):  不同的组之间如果共同拥有一个及以上的关键帧,那么称这两个组之间具有连续关系
    // 连续性(Consistency):称之为连续长度可能更合适,表示累计的连续的链的长度:A--B 为1, A--B--C--D 为3等;具体反映在数据类型 ConsistentGroup.second上
    // 连续组(Consistent group): mvConsistentGroups存储了上次执行回环检测时, 新的被检测出来的具有连续性的多个组的集合.由于组之间的连续关系是个网状结构,因此可能存在
    //                          一个组因为和不同的连续组链都具有连续关系,而被添加两次的情况(当然连续性度量是不相同的)
    // 连续组链:自造的称呼,类似于菊花链A--B--C--D这样形成了一条连续组链.对于这个例子中,由于可能E,F都和D有连续关系,因此连续组链会产生分叉;为了简化计算,连续组中将只会保存
    //         最后形成连续关系的连续组们(见下面的连续组的更新)
    // 子连续组: 上面的连续组中的一个组
    // 连续组的初始值: 在遍历某个候选帧的过程中,如果该子候选组没有能够和任何一个上次的子连续组产生连续关系,那么就将添加自己组为连续组,并且连续性为0(相当于新开了一个连续链)
    // 连续组的更新: 当前次回环检测过程中,所有被检测到和之前的连续组链有连续的关系的组,都将在对应的连续组链后面+1,这些子候选组(可能有重复,见上)都将会成为新的连续组;
    //              换而言之连续组mvConsistentGroups中只保存连续组链中末尾的组


    mvpEnoughConsistentCandidates.clear();// 最终筛选后得到的闭环帧
    // ConsistentGroup数据类型为pair<set<KeyFrame*>,int>
    // ConsistentGroup.first对应每个“连续组”中的关键帧，ConsistentGroup.second为每个“连续组”的已连续几个的序号
    vector<ConsistentGroup> vCurrentConsistentGroups;
    // 这个下标是每个"子连续组"的下标,bool表示当前的候选组中是否有和该组相同的一个关键帧
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
    /// Step 5.1：遍历刚才得到的每一个候选关键帧
    for(auto kf : vpCandidateKFs){
        /// Step 5.2：将自己以及与自己相连的关键帧构成一个“子候选组”
        set<KeyFrame*> spCandidateGroup = kf->GetConnectedKeyFrames();
        spCandidateGroup.insert(kf);
        // 连续性达标的标志
        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;

        /// Step 5.3：遍历前一次闭环检测到的连续组链
        // 上一次闭环的连续组链 mvConsistentGroups,其中ConsistentGroup的定义：typedef pair<set<KeyFrame*>,int> ConsistentGroup
        // 其中 ConsistentGroup.first对应每个“连续组”中的关键帧集合，ConsistentGroup.second为每个“连续组”的连续长度
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++){
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;// 取出之前的一个子连续组中的关键帧集合
            bool bConsistent = false;
            for(auto cg : spCandidateGroup){
                if(sPreviousGroup.count(cg)){// 如果存在，该“子候选组 spCandidateGroup”与该“子连续组sPreviousGroup”相连
                    bConsistent=true;
                    bConsistentForSomeGroup=true;
                    break;// 该“子候选组”至少与一个”子连续组“相连，跳出循环
                }
            }
            /// Step 5.5：如果判定为连续，接下来判断是否达到连续的条件
            if(bConsistent){
                int nPreviousConsistency = mvConsistentGroups[iG].second;// 取出和当前的候选组发生"连续"关系的子连续组的"已连续次数"
                int nCurrentConsistency = nPreviousConsistency + 1;// 将当前候选组连续长度在原子连续组的基础上 +1，
                // 如果上述连续关系还未记录到 vCurrentConsistentGroups，那么记录一下
                if(!vbConsistentGroup[iG]){
                    // 将该“子候选组”的该关键帧打上连续编号加入到“当前连续组”,放入本次闭环检测的连续组vCurrentConsistentGroups里
                    vCurrentConsistentGroups.emplace_back(spCandidateGroup,nCurrentConsistency);
                    vbConsistentGroup[iG]=true;// 标记一下，防止重复添加到同一个索引iG,但是spCandidateGroup可能重复添加到不同的索引iG对应的vbConsistentGroup 中
                }
                // 如果连续长度满足要求，那么当前的这个候选关键帧是足够靠谱的. 连续性阈值 mnCovisibilityConsistencyTh=3
                if((float)nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent){
                    mvpEnoughConsistentCandidates.push_back(kf); // 记录为达到连续条件了的候选帧
                    bEnoughConsistent=true;  // 标记一下，防止重复添加
                }
            }
        }
        /// Step 5.6：如果该“子候选组”的所有关键帧都和上次闭环无关（不连续），vCurrentConsistentGroups 没有新添加连续关系
        // 于是就把“子候选组”全部拷贝到 vCurrentConsistentGroups， 用于更新mvConsistentGroups，连续性计数器设为0
        if(!bConsistentForSomeGroup){
            vCurrentConsistentGroups.emplace_back(spCandidateGroup,0);
        }
    }


    // 更新连续组
    mvConsistentGroups = vCurrentConsistentGroups;
    // 当前闭环检测的关键帧添加到关键帧数据库中
    mpKeyFrameDB->add(mpCurrentKF);

    if(mvpEnoughConsistentCandidates.empty()){// 未检测到闭环，返回false
        mpCurrentKF->SetErase();
        return false;
    }
    else{
        return true;
    }

    mpCurrentKF->SetErase();
    return false;
}


/**
 * @brief 计算当前关键帧和上一步闭环候选帧的Sim3变换
 * 1. 遍历闭环候选帧集，筛选出与当前帧的匹配特征点数大于20的候选帧集合，并为每一个候选帧构造一个Sim3Solver
 * 2. 对每一个候选帧进行 Sim3Solver 迭代匹配，直到有一个候选帧匹配成功，或者全部失败
 * 3. 取出闭环匹配上关键帧的相连关键帧，得到它们的地图点放入 mvpLoopMapPoints
 * 4. 将闭环匹配上关键帧以及相连关键帧的地图点投影到当前关键帧进行投影匹配
 * 5. 判断当前帧与检测出的所有闭环关键帧是否有足够多的地图点匹配
 * 6. 清空mvpEnoughConsistentCandidates
 * @return true         只要有一个候选关键帧通过Sim3的求解与优化，就返回true
 * @return false        所有候选关键帧与当前关键帧都没有有效Sim3变换
 */
bool LoopClosing::ComputeSim3()
{
    // Sim3 计算流程说明：
    // 1. 通过Bow加速描述子的匹配，利用RANSAC粗略地计算出当前帧与闭环帧的Sim3（当前帧---闭环帧）
    // 2. 根据估计的Sim3，对3D点进行投影找到更多匹配，通过优化的方法计算更精确的Sim3（当前帧---闭环帧）
    // 3. 将闭环帧以及闭环帧相连的关键帧的地图点与当前帧的点进行匹配（当前帧---闭环帧+相连关键帧）
    // 注意以上匹配的结果均都存在成员变量mvpCurrentMatchedPoints中，实际的更新步骤见CorrectLoop()步骤3
    // 对于双目或者是RGBD输入的情况,计算得到的尺度=1

    // 对每个（上一步得到的具有足够连续关系的）闭环候选帧都准备算一个Sim3
    const int nInitialCandidates = (int)mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBmatcher matcher(0.75,true);

    vector<Sim3Solver*> vpSim3Solvers;// 存储每一个候选帧的Sim3Solver求解器
    vpSim3Solvers.resize(nInitialCandidates);
    vector<vector<MapPoint*> > vvpMapPointMatches;// 存储每个候选帧的匹配地图点信息
    vvpMapPointMatches.resize(nInitialCandidates);
    vector<bool> vbDiscarded;// 存储每个候选帧应该被放弃(True）或者 保留(False)
    vbDiscarded.resize(nInitialCandidates);
    int nCandidates=0; // 完成 Step 1 的匹配后，被保留的候选帧数量

    /// Step 1. 遍历闭环候选帧集，初步筛选出与当前关键帧的匹配特征点数大于20的候选帧集合，并为每一个候选帧构造一个Sim3Solver
    for(int i=0; i<nInitialCandidates; i++){
        /// Step 1.1 从筛选的闭环候选帧中取出一帧有效关键帧pKF
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];
        pKF->SetNotErase(); // 避免在LocalMapping中KeyFrameCulling函数将此关键帧作为冗余帧剔除
        if(pKF->isBad()){// 如果候选帧质量不高，直接PASS
            vbDiscarded[i] = true;
            continue;
        }
        /// Step 1.2 将当前帧 mpCurrentKF 与闭环候选关键帧pKF匹配
        // 通过bow加速得到 mpCurrentKF 与 pKF 之间的匹配特征点. vvpMapPointMatches 是匹配特征点对应的地图点,本质上来自于候选闭环帧
        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);
        if(nmatches<20){// 粗筛：匹配的特征点数太少，该候选帧剔除
            vbDiscarded[i] = true;
            continue;
        }
        else{
            /// Step 1.3 为保留的候选帧构造Sim3求解器
            auto* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
            pSolver->SetRansacParameters(0.99,20,300);// Sim3Solver Ransac 过程置信度0.99，至少20个inliers 最多300次迭代
            vpSim3Solvers[i] = pSolver;
        }
        nCandidates++;
    }

    bool bMatch = false;// 用于标记是否有一个候选帧通过Sim3Solver的求解与优化

    /// Step 2 对每一个候选帧用Sim3Solver 迭代匹配，直到有一个候选帧匹配成功，或者全部失败
    while(nCandidates>0 && !bMatch){
        for(int i=0; i<nInitialCandidates; i++){// 遍历每一个候选帧
            if(vbDiscarded[i])
                continue;
            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];
            vector<bool> vbInliers;// 内点（Inliers）标志, 即标记经过RANSAC sim3 求解后,vvpMapPointMatches中的哪些作为内点
            int nInliers;// 内点（Inliers）数量
            bool bNoMore;// 是否到达了最优解
            Sim3Solver* pSolver = vpSim3Solvers[i];
            /// Step 2.1 取出从 Step 1.3 中为当前候选帧构建的 Sim3Solver 并开始迭代
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);// 最多迭代5次，返回的Scm是候选帧pKF到当前帧mpCurrentKF的Sim3变换（T12）
            if(bNoMore){// 总迭代次数达到最大限制还没有求出合格的Sim3变换，该候选帧剔除
                vbDiscarded[i]=true;
                nCandidates--;
            }
            // 如果计算出了Sim3变换，继续匹配出更多点并优化。因为之前 SearchByBoW 匹配可能会有遗漏
            if(!Scm.empty()){
                // 取出经过Sim3Solver 后匹配点中的内点集合
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(nullptr));
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++){
                    if(vbInliers[j])
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                }
                /// Step 2.2 通过上面求取的Sim3变换引导关键帧匹配，弥补Step 1中的漏匹配
                cv::Mat R = pSolver->GetEstimatedRotation();// 候选帧pKF到当前帧mpCurrentKF的R（R12），t（t12），变换尺度s（s12）
                cv::Mat t = pSolver->GetEstimatedTranslation();
                const float s = pSolver->GetEstimatedScale();
                // 查找更多的匹配（成功的闭环匹配需要满足足够多的匹配特征点数，之前使用SearchByBoW进行特征点匹配时会有漏匹配）
                // 通过Sim3变换，投影搜索pKF1的特征点在pKF2中的匹配，同理，投影搜索pKF2的特征点在pKF1中的匹配
                // 只有互相都成功匹配的才认为是可靠的匹配
                ORBmatcher::SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);
                /// Step 2.3 用新的匹配来优化 Sim3，只要有一个候选帧通过Sim3的求解与优化，就跳出停止对其它候选帧的判断
                // OpenCV的Mat矩阵转成Eigen的Matrix类型
                // gScm：候选关键帧到当前帧的Sim3变换
                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
                // 如果mbFixScale为true，则是6 自由度优化（双目 RGBD），如果是false，则是7 自由度优化（单目）
                // 优化mpCurrentKF与pKF对应的MapPoints间的Sim3，得到优化后的量gScm
                const int nInliers_local = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);
                if(nInliers_local >= 20)// 如果优化成功，则停止while循环遍历闭环候选
                {
                    bMatch = true;// 为True时将不再进入 while循环
                    mpMatchedKF = pKF;// mpMatchedKF就是最终闭环检测出来与当前帧形成闭环的关键帧
                    // gSmw：从世界坐标系 w 到该候选帧 m 的Sim3变换，都在一个坐标系下，所以尺度 Scale=1
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
                    // 得到g2o优化后从世界坐标系到当前帧的Sim3变换
                    mg2oScw = gScm*gSmw;
                    mScw = Converter::toCvMat(mg2oScw);
                    // 只要有一个候选帧通过Sim3的求解与优化，就跳出停止对其它候选帧的判断
                    mvpCurrentMatchedPoints = vpMapPointMatches;
                    break;
                }
            }
        }
    }

    // 退出上面while循环的原因有两种,一种是求解到了bMatch置位后出的,另外一种是nCandidates耗尽为0
    if(!bMatch){
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

    /// Step 3：取出与当前帧闭环匹配上的关键帧及其共视关键帧，以及这些共视关键帧的地图点
    // 注意是闭环检测出来与当前帧形成闭环的关键帧 mpMatchedKF
    // 将mpMatchedKF共视的关键帧全部取出来放入 vpLoopConnectedKFs
    // 将vpLoopConnectedKFs的地图点取出来放入mvpLoopMapPoints
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
    for(auto pKF : vpLoopConnectedKFs){
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for(auto pMP : vpMapPoints){
            if(pMP){
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId){// mnLoopPointForKF 用于标记，避免重复添加
                    mvpLoopMapPoints.push_back(pMP);
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }
    /// Step 4：将闭环关键帧及其连接关键帧的所有地图点投影到当前关键帧进行投影匹配
    // 根据投影查找更多的匹配（成功的闭环匹配需要满足足够多的匹配特征点数）
    // 根据Sim3变换，将每个mvpLoopMapPoints投影到mpCurrentKF上，搜索新的匹配对
    // mvpCurrentMatchedPoints是前面经过SearchBySim3得到的已经匹配的点对，这里就忽略不再匹配了
    // 搜索范围系数为10
    ORBmatcher::SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

    /// Step 5: 统计当前帧与闭环关键帧的匹配地图点数目，超过40个说明成功闭环，否则失败
    int nTotalMatches = 0;
    for(auto & mvpCurrentMatchedPoint : mvpCurrentMatchedPoints){
        if(mvpCurrentMatchedPoint)
            nTotalMatches++;
    }
    if(nTotalMatches>=40){ // 如果当前回环可靠,保留当前待闭环关键帧，其他闭环候选全部删掉以后不用了
        for(int i=0; i<nInitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else{ // 闭环不可靠，闭环候选及当前待闭环帧全部删除
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

}



/**
 * @brief 闭环矫正
 * 1. 通过求解的Sim3以及相对姿态关系，调整与当前帧相连的关键帧位姿以及这些关键帧观测到的地图点位置（相连关键帧---当前帧）
 * 2. 将闭环帧以及闭环帧相连的关键帧的地图点和与当前帧相连的关键帧的点进行匹配（当前帧+相连关键帧---闭环帧+相连关键帧）
 * 3. 通过MapPoints的匹配关系更新这些帧之间的连接关系，即更新covisibility graph
 * 4. 对Essential Graph（Pose Graph）进行优化，MapPoints的位置则根据优化后的位姿做相对应的调整
 * 5. 创建线程进行全局Bundle Adjustment
 */
void LoopClosing::CorrectLoop()
{
    // Step 0：结束局部地图线程、全局BA，为闭环矫正做准备
    // Step 1：根据共视关系更新当前帧与其它关键帧之间的连接
    // Step 2：通过位姿传播，得到Sim3优化后，与当前帧相连的关键帧的位姿，以及它们的MapPoints
    // Step 3：检查当前帧的MapPoints与闭环匹配帧的MapPoints是否存在冲突，对冲突的MapPoints进行替换或填补
    // Step 4：通过将闭环时相连关键帧的mvpLoopMapPoints投影到这些关键帧中，进行MapPoints检查与替换
    // Step 5：更新当前关键帧之间的共视相连关系，得到因闭环时MapPoints融合而新得到的连接关系
    // Step 6：进行EssentialGraph优化，LoopConnections是形成闭环后新生成的连接关系，不包括步骤7中当前帧与闭环匹配帧之间的连接关系
    // Step 7：添加当前帧与闭环匹配帧之间的边（这个连接关系不优化）
    // Step 8：新建一个线程用于全局BA优化

    // g2oSic： 当前关键帧 mpCurrentKF 到其共视关键帧 pKFi 的Sim3 相对变换
    // mg2oScw: 世界坐标系到当前关键帧的 Sim3 变换
    // g2oCorrectedSiw：世界坐标系到当前关键帧共视关键帧的Sim3 变换

    cout << "Loop detected!" << endl;
    /// Step 0：结束局部地图线程、全局BA，为闭环矫正做准备
    mpLocalMapper->RequestStop();
    // 如果有全局BA在运行，终止掉，迎接新的全局BA
    if(isRunningGBA()){
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;
        mnFullBAIdx++;
        if(mpThreadGBA){// 停止全局BA线程
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }
    // 一直等到局部地图线程结束再继续
    while(!mpLocalMapper->isStopped()){
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }



    /// Step 1：根据共视关系更新当前关键帧与其它关键帧之间的连接关系
    // 因为之前闭环检测、计算Sim3中改变了该关键帧的地图点，所以需要更新
    mpCurrentKF->UpdateConnections();

    /// Step 2：通过位姿传播，得到Sim3优化后，与当前帧相连的关键帧的位姿，以及它们的地图点
    // 当前帧与世界坐标系之间的Sim变换在ComputeSim3函数中已经确定并优化，通过相对位姿关系，可以确定这些相连的关键帧与世界坐标系之间的Sim3变换
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();// 取出当前关键帧及其共视关键帧，称为“当前关键帧组”
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);
    KeyFrameAndPose CorrectedSim3;//存放闭环g2o优化后当前关键帧的共视关键帧的世界坐标系下Sim3 变换
    KeyFrameAndPose NonCorrectedSim3;//存放没有矫正的当前关键帧的共视关键帧的世界坐标系下Sim3 变换
    CorrectedSim3[mpCurrentKF]=mg2oScw;// 先将mpCurrentKF的Sim3变换存入，认为是准的，所以固定不动
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();// 当前关键帧到世界坐标系下的变换矩阵
    // 对地图点操作
    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        /// Step 2.1：通过mg2oScw（认为是准的）来进行位姿传播，得到当前关键帧的共视关键帧的世界坐标系下Sim3 位姿
        for(auto pKFi : mvpCurrentConnectedKFs){
            cv::Mat Tiw = pKFi->GetPose();
            if(pKFi!=mpCurrentKF){
                // 得到当前关键帧 mpCurrentKF 到其共视关键帧 pKFi 的相对变换
                cv::Mat Tic = Tiw*Twc;
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                cv::Mat tic = Tic.rowRange(0,3).col(3);
                // g2oSic：当前关键帧 mpCurrentKF 到其共视关键帧 pKFi 的Sim3 相对变换, 这里是non-correct, 所以scale=1.0
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;// 当前帧的位姿固定不动，其它的关键帧根据相对关系得到Sim3调整的位姿
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw;// 存放闭环g2o优化后当前关键帧的共视关键帧的Sim3 位姿
            }
            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            NonCorrectedSim3[pKFi]=g2oSiw;// 存放没有矫正的当前关键帧的共视关键帧的Sim3变换
        }
        /// Step 2.2：得到矫正的当前关键帧的共视关键帧位姿后，修正这些共视关键帧的地图点
        for(auto & mit : CorrectedSim3){
            KeyFrame* pKFi = mit.first;// 取出当前关键帧连接关键帧
            g2o::Sim3 g2oCorrectedSiw = mit.second;// 取出经过位姿传播后的Sim3变换
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();
            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];// 取出未经过位姿传播的Sim3变换
            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(auto pMPi : vpMPsi){
                if(!pMPi || pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                    continue;
                // 矫正过程本质上也是基于当前关键帧的优化后的位姿展开的
                // 将该未校正的eigP3Dw先从世界坐标系映射到未校正的pKFi相机坐标系，然后再反映射到校正后的世界坐标系下
                cv::Mat P3Dw = pMPi->GetWorldPos();// 地图点世界坐标系下坐标
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw)); // map(P) 内部做了相似变换 s*R*P +t

                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;// 记录矫正该地图点的关键帧id，防止重复
                pMPi->mnCorrectedReference = pKFi->mnId;// 记录该地图点所在的关键帧id
                pMPi->UpdateNormalAndDepth();// 因为地图点更新了，需要更新其平均观测方向以及观测距离范围
            }
            /// Step 2.3：将共视关键帧的Sim3转换为SE3，根据更新的Sim3，更新关键帧的位姿
            // 其实是现在已经有了更新后的关键帧组中关键帧的位姿,但是在上面的操作时只是暂时存储到了 KeyFrameAndPose 类型的变量中,还没有写回到关键帧对象中
            // 调用toRotationMatrix 可以自动归一化旋转矩阵
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();
            eigt *=(1./s); //[R t/s;0 1]  // 平移向量中包含有尺度信息，还需要用尺度归一化
            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);
            pKFi->SetPose(correctedTiw);// 设置矫正后的新的pose
            /// Step 2.4：根据共视关系更新当前帧与其它关键帧之间的连接, 地图点的位置改变了,可能会引起共视关系\权值的改变
            pKFi->UpdateConnections();
        }

        /// Step 3：检查当前帧的地图点与经过闭环匹配后该帧的地图点是否存在冲突，对冲突的进行替换或填补
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++){
            if(mvpCurrentMatchedPoints[i]){
                //取出同一个索引对应的两种地图点，决定是否要替换
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];// 匹配投影得到的地图点
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i); // 原来的地图点
                if(pCurMP){
                    pCurMP->Replace(pLoopMP);// 如果有重复的MapPoint，则用匹配的地图点代替现有的.因为匹配的地图点是经过一系列操作后比较精确的，现有的地图点很可能有累计误差
                }
                else{// 如果当前帧没有该MapPoint，则直接添加
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }

    }

    /// Step 4：将闭环相连关键帧组mvpLoopMapPoints 投影到当前关键帧组中，进行匹配，融合，新增或替换当前关键帧组中KF的地图点
    // 因为 闭环相连关键帧组mvpLoopMapPoints 在地图中时间比较久经历了多次优化，认为是准确的
    // 而当前关键帧组中的关键帧的地图点是最近新计算的，可能有累积误差
    // CorrectedSim3：存放矫正后当前关键帧的共视关键帧，及其世界坐标系下Sim3 变换
    SearchAndFuse(CorrectedSim3);

    /// Step 5：更新当前关键帧组之间的两级共视相连关系，得到因闭环时地图点融合而新得到的连接关系
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;//存储因为闭环地图点调整而新生成的连接关系
    /// Step 5.1：遍历当前帧相连关键帧组（一级相连）
    for(auto pKFi : mvpCurrentConnectedKFs){
        /// Step 5.2：得到与当前帧相连关键帧的相连关键帧（二级相连）
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();
        /// Step 5.3：更新一级相连关键帧的连接关系(会把当前关键帧添加进去,因为地图点已经更新和替换了)
        pKFi->UpdateConnections();
        /// Step 5.4：取出该帧更新后的连接关系
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
        /// Step 5.5：从连接关系中去除闭环之前的二级连接关系，剩下的连接就是由闭环得到的连接关系
        for(auto & vpPreviousNeighbor : vpPreviousNeighbors)
            LoopConnections[pKFi].erase(vpPreviousNeighbor);
        /// Step 5.6：从连接关系中去除闭环之前的一级连接关系，剩下的连接就是由闭环得到的连接关系
        for(auto & mvpCurrentConnectedKF : mvpCurrentConnectedKFs)
            LoopConnections[pKFi].erase(mvpCurrentConnectedKF);
    }

    /// Step 6：进行本质图优化，优化本质图中所有关键帧的位姿和地图点
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);
    mpMap->InformNewBigChange();

    /// Step 7：添加当前帧与闭环匹配帧之间的边（这个连接关系不优化）
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    /// Step 8：新建一个线程用于全局BA优化
    // OptimizeEssentialGraph只是优化了一些主要关键帧的位姿，这里进行全局BA可以全局优化所有位姿和MapPoints
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();    

    mLastLoopKFid = mpCurrentKF->mnId;   
}

/**
 * @brief 将闭环相连关键帧组mvpLoopMapPoints 投影到当前关键帧组中，进行匹配，新增或替换当前关键帧组中KF的地图点
 * 因为 闭环相连关键帧组mvpLoopMapPoints 在地图中时间比较久经历了多次优化，认为是准确的
 * 而当前关键帧组中的关键帧的地图点是最近新计算的，可能有累积误差
 *
 * @param[in] CorrectedPosesMap         矫正的当前KF对应的共视关键帧及Sim3变换
 */
void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    ORBmatcher matcher(0.8);
    /// Step 1 遍历待矫正的当前KF的相连关键帧
    for(const auto & mit : CorrectedPosesMap){
        KeyFrame* pKF = mit.first;
        g2o::Sim3 g2oScw = mit.second;// 矫正过的Sim 变换
        cv::Mat cvScw = Converter::toCvMat(g2oScw);
        /// Step 2 将mvpLoopMapPoints投影到pKF帧匹配，检查地图点冲突并融合
        // mvpLoopMapPoints：与当前关键帧闭环匹配上的关键帧及其共视关键帧组成的地图点
        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(nullptr));
        // vpReplacePoints：存储mvpLoopMapPoints投影到pKF匹配后需要替换掉的新增地图点,索引和mvpLoopMapPoints一致，初始化为空
        // 搜索区域系数为4
        ORB_SLAM2::ORBmatcher::Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP =(int) mvpLoopMapPoints.size();
        /// Step 3 遍历闭环帧组的所有的地图点，替换掉需要替换的地图点
        for(int i=0; i<nLP;i++){
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep){
                pRep->Replace(mvpLoopMapPoints[i]);
            }
        }
    }
}


void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(true)
    {
        {
        unique_lock<mutex> lock2(mMutexReset);
        if(!mbResetRequested)
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
    }
}


/**
 * @brief 全局BA线程,这个是这个线程的主函数
 *
 * @param[in] nLoopKF 看上去是闭环关键帧id,但是在调用的时候给的其实是当前关键帧的id
 */
void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx =  mnFullBAIdx;// 记录GBA已经迭代次数,用来检查全局BA过程是否是因为意外结束的
    // mbStopGBA直接传引用过去了,这样当有外部请求的时候这个优化函数能够及时响应并且结束掉
    // 提问:进行完这个过程后我们能够获得哪些信息?
    // 回答：能够得到全部关键帧优化后的位姿,以及优化后的地图点
    /// Step 1 执行全局BA，优化所有的关键帧位姿和地图中地图点
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    // 更新所有的地图点和关键帧
    // 在global BA过程中local mapping线程仍然在工作，这意味着在global BA时可能有新的关键帧产生，但是并未包括在GBA里，
    // 所以和更新后的地图并不连续。需要通过spanning tree来传播
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)// 如果全局BA过程是因为意外结束的,那么直接退出GBA
            return;
        // 如果当前GBA没有中断请求，更新位姿和地图点
        // 这里和上面那句话的功能还有些不同,因为如果一次全局优化被中断,往往意味又要重新开启一个新的全局BA;为了中断当前正在执行的优化过程mbStopGBA将会被置位,同时会有一定的时间
        // 使得该线程进行响应;而在开启一个新的全局优化进程之前 mbStopGBA 将会被置为False
        // 因此,如果被强行中断的线程退出时已经有新的线程启动了,mbStopGBA=false,为了避免进行后面的程序,所以有了上面的程序;
        // 而如果被强行中断的线程退出时新的线程还没有启动,那么上面的条件就不起作用了(虽然概率很小,前面的程序中mbStopGBA置位后很快mnFullBAIdx就++了,保险起见),所以这里要再判断一次

        if(!mbStopGBA){
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped
            // 等待直到local mapping结束才会继续后续操作
            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished()){
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate); // 后续要更新地图所以要上锁

            // Correct keyframes starting at map first keyframe
            // 从第一个关键帧开始矫正关键帧。刚开始只保存了初始化第一个关键帧
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());
            // 问：GBA里锁住第一个关键帧位姿没有优化，其对应的pKF->mTcwGBA是不变的吧？那后面调整位姿的意义何在？
            // 回答：注意在前面essential graph BA里只锁住了回环帧，没有锁定第1个初始化关键帧位姿。所以第1个初始化关键帧位姿已经更新了
            // 在GBA里锁住第一个关键帧位姿没有优化，其对应的pKF->mTcwGBA应该是essential BA结果，在这里统一更新了
            // Step 2 遍历并更新全局地图中的所有spanning tree中的关键帧
            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                cv::Mat Twc = pKF->GetPoseInverse();
                for(auto pChild : sChilds)
                {
                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        cv::Mat Tchildc = pChild->GetPose()*Twc;
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF=nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

            for(auto pMP : vpMPs)
            {
                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);

                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }            

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
