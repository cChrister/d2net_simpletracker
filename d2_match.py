import os
os.environ['CUDA_VISIBLE_DEVICES']='5'

import argparse
import numpy as np
import imageio.v2 as imageio
import cv2
import torch
import time
import copy
import compute
from tqdm import tqdm
import scipy
import scipy.io
import scipy.misc

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

# set device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# 相机内参 
k = np.array([[2.5493694140950984e+02, 0, 3.3958970225755928e+02],               
              [0, 2.5463452970785917e+02, 2.4282638011463291e+02],               
              [0, 0, 1]]) 
# 相机畸变参数
CDP = np.array([-1.1003412529587565e-01,                 
                -3.2512298569625435e-03,                  
                -6.1055603631990905e-04,                 
                -1.8817657758755631e-03,                 
                2.2227416180894094e-03])


# Argument parsing
parser = argparse.ArgumentParser(description='Feature extraction script')
# waht this script need
parser.add_argument('--data_path', type=str, 
                    default='/home/chenxiang/code/d2net_simpletracker/20230702',
                    help='path to a file containinga lists of images solute path')
parser.add_argument('--pair', type=str, 
                    default='1lux_infrared_RGB_1',
                    help='which pair you want to compare')


# what the model need
parser.add_argument('--preprocessing', type=str, default='caffe',
                    help='image preprocessing (caffe or torch)')
parser.add_argument('--model_file', type=str, default='models/d2_ots.pth',
                    help='path to the full model')
parser.add_argument('--max_edge', type=int, default=1600,
                    help='maximum image size at network input')
parser.add_argument('--max_sum_edges', type=int, default=2800,
                    help='maximum sum of image sizes at network input')
parser.add_argument('--output_extension', type=str, default='.d2-net',
                    help='extension for the output')
parser.add_argument('--output_type', type=str, default='npz',
                    help='output file type (npz or mat)')
parser.add_argument('--multiscale', dest='multiscale', action='store_true',
                    help='extract multiscale features')
parser.set_defaults(multiscale=False)
parser.add_argument('--no-relu', dest='use_relu', action='store_false',
                    help='remove ReLU after the dense feature extraction module')
parser.set_defaults(use_relu=True)
args = parser.parse_args()

print(args)

# we use this class to match
class Tracker1(object):
    def __init__(self):
        self.pts_prev = None
        self.desc_prev = None

    def update(self, img, pts, desc,id):
        N_matches = 0 
        # 前一帧和后一帧之间做匹配，如果这是第一帧则保存用于和第二帧匹配
        if self.pts_prev is None:
            self.pts_prev = pts
            self.desc_prev = desc

            out = copy.deepcopy(img)
            for pt1 in pts:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 1, (0, 0, 255), -1, lineType=16)
            return out, N_matches
        
        else:
            matches = self.mnn_mather(self.desc_prev, desc)

            # here we use RANSAC to 
            if matches.shape[0] > 8:
                    pt1 = []
                    pt2 = []
                    for i in range(matches.shape[0]):
                        pt1.append(self.pts_prev[int(matches[i][0]),:])
                        pt2.append(pts[int(matches[i][1]),:])
                    pt1 = np.array(pt1)[:,:-1]
                    pt2 = np.array(pt2)[:,:-1]
                    _, mask = cv2.findEssentialMat(pt1, pt2, k, cv2.RANSAC)
                    matches_new = []
                    for i in range(mask.shape[0]):
                        if mask[i][0] == 0:
                            continue
                        else:
                            matches_new.append(matches[i,:])
                    matches_new = np.array(matches_new)
                    matches = matches_new

            mpts1, mpts2 = self.pts_prev[matches[:, 0]], pts[matches[:, 1]]
            N_matches = len(matches)

            out = copy.deepcopy(img)
            output_result = args.data_path + "/matches_" + args.pair
            os.makedirs(output_result, exist_ok=True)
            with open(output_result + "/match_{}.txt".format((id+1)/2), 'w') as f:
                for pt1, pt2 in zip(mpts1, mpts2):
                    p1 = (int(round(pt1[0])), int(round(pt1[1])))
                    p2 = (int(round(pt2[0])), int(round(pt2[1])))
                    output_result = args.data_path + "/matches"
                    # os.makedirs(output_result, exist_ok=True)

                    f.write("{} {} {} {}\n".format(pt1[0], pt1[1], pt2[0], pt2[1]))

                    cv2.line(out, p1, p2, (0, 255, 0), lineType=16)
                    cv2.circle(out, p2, 1, (0, 0, 255), -1, lineType=16)

                self.pts_prev = pts
                self.desc_prev = desc
            return out, N_matches

    def mnn_mather(self, desc1, desc2):
        sim = desc1 @ desc2.transpose()
        sim[sim < 0.9] = 0
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)
        ids1 = np.arange(0, sim.shape[0])
        mask = (ids1 == nn21[nn12])
        matches = np.stack([ids1[mask], nn12[mask]])
        return matches.transpose()

# Creating CNN model
model = D2Net(
    model_file=args.model_file,
    use_relu=args.use_relu,
    use_cuda=use_cuda
)
_0, _1, _2 = process_multiscale(torch.tensor(np.random.rand(3,480,640)[np.newaxis, :, :, :].astype(np.float32),
                                             device=device),model,scales=[1])
tracker = Tracker1()

# we use these mark
frame_id = 0
d2net_point = 0
d2net_descriptor = 0
match_num_total = 0
extra_time=0

# Process the file, next we get descriptors and keypoints
with open(args.data_path +'/'+ args.pair +'.txt', 'r') as f:
    lines = f.readlines()
start = time.time()
for line in tqdm(lines, total=len(lines)):
    # process image to feed it in network
    path = line.strip('\n')
    image = imageio.imread(path)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)
    resized_image = image
    if max(resized_image.shape) > args.max_edge:
        resized_image = scipy.misc.imresize(
            resized_image,
            args.max_edge / max(resized_image.shape)
        ).astype('float')
    if sum(resized_image.shape[: 2]) > args.max_sum_edges:
        resized_image = scipy.misc.imresize(
            resized_image,
            args.max_sum_edges / sum(resized_image.shape[: 2])
        ).astype('float')
    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]
    input_image = preprocess_image(
        resized_image,
        preprocessing=args.preprocessing
    )

    # next we get descriptors and keypoints
    # torch.cuda.synchronize()
    start2=time.time()
    with torch.no_grad():
        if args.multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(input_image[np.newaxis, :, :, :].astype(np.float32),device=device),model)
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(input_image[np.newaxis, :, :, :].astype(np.float32),device=device),model,scales=[1])
    # torch.cuda.synchronize()
    end2=time.time()
    extra_time+=(end2-start2)
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    keypoints = keypoints[:, [1, 0, 2]]

    # here we match the image
    out, N_matches = tracker.update(image, keypoints, descriptors, frame_id)
    frame_id+=1
    d2net_point += int(keypoints.shape[0])
    match_num_total += N_matches

    write_dir = args.data_path + "/output_" + args.pair
    if not os.path.exists(write_dir):
            os.makedirs(write_dir)
    out_file = os.path.join(write_dir, "frame_%05d.png" % frame_id)
    cv2.imwrite(out_file, out)
end=time.time()

# "frame_id" equals to "total image nums"
# next we compute these marks
output_result = args.data_path + "/matches_" + args.pair
t_err_average, R_err_average, d = compute.compute_err(output_result, frame_id, skip=1, k=k, interpolate=True)
with open(str(args.data_path)+"/" + args.pair + "_result.txt", 'w') as f:
        f.write(args.data_path + ";skip=" + str(1) + ";LDC=" + '_no' + '\n')
        f.write("图像数量=" + str(frame_id) + "张" + '\n')
        f.write("平均特征点数 %d 个" % (d2net_point / frame_id) + '\n')
        f.write("平均配对特征点数 %d 对" % (match_num_total / (frame_id - 1)) + '\n')
        f.write("配对失败图片 %d 对" % (d) + '\n')
        f.write("每次提取特征点和描述子时间 %.6f ms" % (extra_time / frame_id * 1000) + '\n')
        f.write("平均位移误差 %.6f cm" % (t_err_average * 100) + '\n')
        f.write("平均旋转误差 %.1f °" % (R_err_average) + '\n')
