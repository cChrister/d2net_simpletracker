import os
import glob
import cv2
video_list=['output_1lux_infrared_RGB_0',
            'output_1lux_infrared_RGB_1',
            'output_1lux_infrared_RGB_2',
            'output_1lux_infrared_RGB_3',
            'output_3lux_infrared_RGB_0',
            'output_3lux_infrared_RGB_1',
            'output_3lux_infrared_RGB_2',
            'output_3lux_infrared_RGB_3',
            'output_5lux_corridor_RGB_0',
            'output_5lux_corridor_RGB_1',
            'output_5lux_corridor_RGB_2',
            'output_5lux_corridor_RGB_3',
            'output_300lux_office_RGB_0',
            'output_300lux_office_RGB_1',
            'output_300lux_office_RGB_2',
            'output_300lux_office_RGB_3',
            'output_400lux_office_RGB_0',
            'output_400lux_office_RGB_1',
            'output_400lux_office_RGB_2',
            'output_400lux_office_RGB_3']

def getvideo(file):
    img_paths_list =[]
    with open(file+'.txt',"r") as f:
        lines = f.readlines()
        for line in lines:
            img_paths_list.append(line.strip('\n'))
    video_tag_template = '/home/chenxiang/code/d2net_simpletracker/'+file+'_video.mp4'

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = video_tag_template
    videoWriter = cv2.VideoWriter(video_name, fourcc, fps, (640, 480), True)


    for path in img_paths_list:
        frame = cv2.imread(path)
        videoWriter.write(frame)
    videoWriter.release()

for name in video_list:
    getvideo(name)