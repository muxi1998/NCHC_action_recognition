import json
import glob, os
import numpy as np
import argparse

key_point_no = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]) * 3

parser = argparse.ArgumentParser()
parser.add_argument("--json_folder_dir", help="Folder path from the openpose output(.json)", default="./openpose_mac/openpose/output_json_folder")
parser.add_argument("--output_txt", help="Path of the converted output (.txt) from .json", default="output.txt")
args = parser.parse_args()

if os.path.isdir(args.json_folder_dir):
    os.chdir(args.json_folder_dir)
    kps = []  # 18 key points (x, y)
    for file in sorted(glob.glob("*.json")):
        with open(file) as data_file:
            data = json.load(data_file)  # read json file of one frame
            if len(data["people"]) > 1:
                print("*** Detect more than one person ***")
            frame_kps = []
            print("people num: {}".format(len(data["people"])))  # test
            pose_keypoints = data["people"][0]["pose_keypoints_2d"]
            # loop through 25 pose keypoints (total = 75, 25x3 (x, y and accuracy))
            for kp in key_point_no:
                frame_kps.append(pose_keypoints[kp]) # first is x-axis
                kp += 1
                frame_kps.append(pose_keypoints[kp]) # second is y-axis
                # third is accuracy (ignore)
            kps.append(frame_kps)
else:
    print("### Error: json folder {} is not exist! ###".format(args.json_folder_dir))
        
f = open(args.output_txt, "w")    
for frame_kps in kps:
    for index in range(len(frame_kps)):
        f.write(str(frame_kps[index]))
        if index < len(frame_kps) - 1:
            f.write(",")
    f.write("\n")
f.close()
# print("Total kps: {}".format(kps))