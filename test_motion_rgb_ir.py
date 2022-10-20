"""
baseline for Anti-UAV
https://anti-uav.github.io/
"""
from __future__ import absolute_import
import os
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main(mode='IR', visulization=False):
    assert mode in ['IR', 'RGB'], 'Only Support IR or RGB to evalute'
    # setup tracker
    # setup experiments
    video_paths = glob.glob(os.path.join('/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/anti_UAV/dataset', 'test-dev', '*'))    # data path
    video_num = len(video_paths)
    mode_rgb = 'RGB'
    # run tracking experiments and report performance
    for video_id, video_path in enumerate(video_paths, start=1):
        res_file = os.path.join(video_path, '%s_label.json'%mode)
        with open(res_file, 'r') as f:
            label_res = json.load(f)
        delta_x = 0
        delta_y = 0
        gt = label_res['gt_rect']

        res_file_rgb = os.path.join(video_path, '%s_label.json'%mode_rgb)
        with open(res_file_rgb, 'r') as f:
            label_res_rgb = json.load(f)
        delta_x = 0
        delta_y = 0
        gt_rgb = label_res_rgb['gt_rect']

        for i in range(len(gt)):
            if gt[i] == []:
                gt[i] = [0,0,0,0]
            else:
                continue
        for i in range(len(gt_rgb)):
            if gt_rgb[i] == []:
                gt_rgb[i] = [0,0,0,0]
            else:
                continue
        gt = np.array(gt)
        gt_rgb = np.array(gt_rgb)
        t = np.linspace(0, len(gt), len(gt))
        x_ir = gt[:,0] / 640
        x_rgb = gt_rgb[:, 0] / 1920

        y_ir = gt[:,1] / 512
        y_rgb = gt_rgb[:, 1] / 1080

        plt.figure(figsize=(16, 8))
        plt.plot(t, x_ir, label="$x_ir$", color="red", linewidth=2)
        plt.plot(t, x_rgb, label="$x_rgb$", color="green", linewidth=2)

        plt.xlabel("Frame(s)")
        plt.ylabel("V")
        plt.title(os.path.basename(video_path) + " IR Vs. RGB x coord")
        plt.grid(axis="x", which='major')
        ax = plt.gca()
        ax.set_xlim(0, len(gt))
        miloc = plt.MultipleLocator(10)
        ax.xaxis.set_minor_locator(miloc)
        ax.grid(axis='x', which='minor')
        # plt.show()
        plt.savefig('satistic/' + os.path.basename(video_path) + "_IR_RGB_x" + ".jpg")
        plt.close()

        plt.figure(figsize=(16, 8))
        plt.plot(t, y_ir, label="$y_ir$", color="blue", linewidth=2)
        plt.plot(t, y_rgb, label="$y_rgb$", color="yellow", linewidth=2)
        plt.xlabel("Frame(s)")
        plt.ylabel("V")
        plt.title(os.path.basename(video_path) + " IR Vs. RGB y coord")
        plt.grid(axis="x", which='major')
        ax = plt.gca()
        ax.set_xlim(0, len(gt))
        miloc = plt.MultipleLocator(10)
        ax.xaxis.set_minor_locator(miloc)
        ax.grid(axis='x', which='minor')
        # plt.show()
        plt.savefig('satistic/' + os.path.basename(video_path) + "_IR_RGB_y" + ".jpg")
        plt.close()


if __name__ == '__main__':
    main(mode='IR', visulization=False)
