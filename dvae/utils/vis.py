#!/usr/bin/env python
# encoding: utf-8

# from utils import data_utils
import numpy as np
#import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
#import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import torch
from tqdm import tqdm

##Functions to visualize human poses"""
##from: https://github.com/enriccorona/human-motion-prediction-pytorch/blob/master/src/viz.py
class H36m3DPose(object):
  def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Create a 3d pose visualizer that can be updated with new poses.
    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """
    # Start and endpoints of our representation
    self.I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
    self.J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1
    # Left / right indicator
    self.LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    self.ax = ax

    vals = np.zeros((32, 3))

    # Make connection matrix
    self.plots = []
    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))

    vals_ = np.zeros((13, 3))
    self.plots_ = []
    for i in np.arange( len(self.I) ):
        x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
        y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
        z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
        self.plots_.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))

    self.ax.set_xlabel("x")
    self.ax.set_ylabel("y")
    self.ax.set_zlabel("z")

  def update(self, channels, channels2=None, lcolor="lightcoral", rcolor="cornflowerblue"): #"#3498db", rcolor="#e74c3c"):
    """
    Update the plotted 3d pose.
    Args
      channels: 96-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """
    assert channels.size == 96, "channels should have 96 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (32, -1) )
    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      z = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      y = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      #print(x,y,z)
      self.plots[i][0].set_xdata(x)
      self.plots[i][0].set_ydata(y)
      self.plots[i][0].set_3d_properties(z)
      self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)

    if channels2 is not None:
        vals2 = np.reshape( channels2, (32, -1) )
        for i in np.arange( len(self.I) ):
            x = np.array( [vals2[self.I[i], 0], vals2[self.J[i], 0]] )
            z = np.array( [vals2[self.I[i], 1], vals2[self.J[i], 1]] )
            y = np.array( [vals2[self.I[i], 2], vals2[self.J[i], 2]] )
            self.plots_[i][0].set_xdata(x)
            self.plots_[i][0].set_ydata(y)
            self.plots_[i][0].set_3d_properties(z)
            self.plots_[i][0].set_color('darkred' if self.LR[i] else 'darkblue')


    r = 750;
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    self.ax.set_xlim3d([-r+xroot, r+xroot])
    self.ax.set_zlim3d([-r+zroot, r+zroot])
    self.ax.set_ylim3d([-r+yroot, r+yroot])
    #self.ax.set_aspect('equal')


def vis_h36m(p3d,save_path):
    #input: p3d np array (seq_len, 32, 3)
    num_frames = len(p3d) #2487
    metadata = dict(title='01', artist='Matplotlib',comment='motion')
    writer = FFMpegWriter(fps=10, metadata=metadata)
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ob = H36m3DPose(ax)
    with writer.saving(fig, save_path, 100):
        for i in tqdm(range(num_frames)):
            ob.update(p3d[i])
            print(">>>save")
            writer.grab_frame()
            plt.pause(0.01)
            #plt.clf()

def vis_h36m_compare(p3d_gt,p3d_pred,save_path):
    ## vis input, gt and pred togethor
    #input: p3d_gt:  np array (seq_len=in+out, 32, 3)
           #p3d_pred:np array (seq_len=out, 32, 3)
    num_frames_gt = len(p3d_gt) #75
    num_frames_pred = len(p3d_pred) #25

    metadata = dict(title='01', artist='Matplotlib',comment='motion')
    writer = FFMpegWriter(fps=10, metadata=metadata)
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ob = H36m3DPose(ax)
    with writer.saving(fig, save_path, 100):
        for i in tqdm(range(num_frames_gt - num_frames_pred)):
            ob.update(p3d_gt[i])
            plt.savefig('../tmp/'+str(i)+'.jpg')
            writer.grab_frame()
            plt.pause(0.01)
        for i in tqdm(range(num_frames_gt - num_frames_pred,num_frames_gt)):
            ob.update(p3d_gt[i], p3d_pred[i-num_frames_gt+num_frames_pred])
            writer.grab_frame()
            plt.pause(0.01)

# if __name__ == '__main__':
#     filename = "../datasets/h3.6m/S5/directions_1.txt"
#     the_sequence = data_utils.readCSVasFloat(filename)
#     n, d = the_sequence.shape
#     even_list = range(0, n, 2)
#     #num_frames = len(even_list) #2487
#     the_sequence = np.array(the_sequence[even_list, :]) #(2487,99)
#     the_sequence = torch.from_numpy(the_sequence).float().cuda()
#     the_sequence[:, 0:6] = 0
#     p3d = data_utils.expmap2xyz_torch(the_sequence) #torch.Size([2487, 32, 3])
#     p3d = p3d.detach().cpu().numpy()

#     vis_h36m(p3d,'../tmp/out.mp4')

