import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

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
k=np.array([[259.09196890763241,0,3.0389934394165209e+02],
            [0,258.91454340850947,244.70094767551925],
            [0, 0, 1]])
# 相机畸变参数
CDP=np.array([5.6250819388809450e-02,-1.2304030974433684e-01,1.5401797739192118e-05,
              -6.2692468784740669e-05,3.4887302271479820e-02])


# Argument parsing
parser = argparse.ArgumentParser(description='Feature extraction script')
# waht this script need
parser.add_argument('--data_path', type=str, 
                    default='/home/chenxiang/code/d2-net/test',
                    help='path to a file containinga lists of images solute path')
parser.add_argument('--pair', type=str, 
                    default='1044_733',
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
        else:
            matches = self.mnn_mather(self.desc_prev, desc)
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
                    os.makedirs(output_result, exist_ok=True)

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
