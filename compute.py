import cv2
import numpy as np
import os
import math
from scipy.spatial.transform import Rotation as R

def compute_RT(pt1, pt2, K) :
    #使用RANSAC方法求取本征矩阵
    E,mask = cv2.findEssentialMat(pt1, pt2,K,cv2.RANSAC,0.1, 0.999)
    points, R_est, t_est, mask_pose=cv2.recoverPose(E,pt1,pt2,K)
    return R_est, t_est

def quaternion2euler(quaternion):
    #四元数转欧拉角
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler

def euler2rotation(euler):
    #欧拉角转旋转矩阵
    r = R.from_euler('xyz', euler, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix

def match(matches):
    new_pt1=[]
    new_pt2=[]
    for i in range(matches.shape[0]):
        new_pt1.append(matches[i,:2])
        new_pt2.append(matches[i,2:4])
    new_pt1=np.array(new_pt1)
    new_pt2=np.array(new_pt2)
    return new_pt1,new_pt2

def rot_error(r_gt,r_est):
    #计算旋转误差
    dis = abs(math.acos((np.trace(np.dot(np.linalg.inv(r_gt),r_est))-1)/2))
    #公式计算结果单位为弧度，转成角度返回
    return dis*180/math.pi


def compute_err(path,image_num,skip,k,interpolate):
    i=0
    matchPath=path
    matchfileList = os.listdir(matchPath)
    matchfileList.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))

    dismatch_num=0

    # if interpolate == False:
    #     truthPath= path + "/mocap_pose.csv"
    #     truth = np.loadtxt(truthPath,delimiter = ",")
    #     truth = truth[::skip]
    # else:
    #     a = np.zeros((30, 6))
    #     b = np.ones((1, 30))
    #     truth = np.insert(a, 6, b, axis=1)
    #     image_num=image_num/2
    # t_err_total=0.0
    # R_err_total=0.0
    # while i < image_num-1:
    #     M=open(os.path.join(matchPath,matchfileList[i]))
    #     matches=np.loadtxt(M)
    #     if(matches.shape[0]<8):
    #         dismatch_num+=1
    #         i+=1
    #         continue
    #
    #     pt1,pt2=match(matches)
    #     R_est, t_est = compute_RT(pt1, pt2, k)
    #     t_est=t_est*0.01*skip
    #
    #     t_tru1=truth[i,0:3].reshape(3,1)
    #     t_tru2=truth[i+1, 0:3].reshape(3,1)
    #     # t_tru1 = np.r_[t_tru1, [[1.]]]
    #     # t_tru2 = np.r_[t_tru2, [[1.]]]
    #     # t_tru1=np.dot(m2c,t_tru1)
    #     # t_tru2=np.dot(m2c,t_tru2)
    #     t_tru=(t_tru2-t_tru1)[:3,0]
    #
    #     q_tru1=truth[i,3:]
    #     q_tru2=truth[i+1,3:]
    #     e_tru1=quaternion2euler(q_tru1)
    #     e_tru2=quaternion2euler(q_tru2)
    #     e_tru=e_tru2-e_tru1
    #     R_tru=euler2rotation(e_tru)
    #
    #     t_err=np.linalg.norm(t_tru-t_est)
    #     R_err=rot_error(R_tru,R_est)
    #     t_err_total=t_err_total+t_err
    #     R_err_total=R_err_total + R_err
    #     i = i + 1
    #
    while i < image_num - 1:
        M=open(os.path.join(matchPath,matchfileList[i]))
        matches=np.loadtxt(M)
        if(matches.shape[0]<8):
            dismatch_num+=1
            i+=1
            continue
        i+=1
    return 0,0,dismatch_num