#!/usr/bin/env python
# encoding: utf-8
# cp from vis_h36m.py
# vis 13 common kpts of h36m and acro
# & vis 2p with 13 kpts

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

class PI3DPose13kpts(object): #H36m3DPose13kpts):
    def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
        #super(PI3DPose13kpts,self).__init__(ax, lcolor="#3498db", rcolor="#e74c3c")
        #pi
        #self.I   = np.array([0,1,3,0,2,4,0,7,9,0,8,10])
        #self.J   = np.array([1,3,5,2,4,6,7,9,11,8,10,12])
        self.I   = np.array([0,1,3,0,2,4,0,7,9,0,8,10])
        self.J   = np.array([1,3,5,2,4,6,7,9,11,8,10,12])
        self.LR  = np.array([0,0,0, 0,0,0, 0,0,0, 0,0,0], dtype=bool)
        self.ax = ax

        vals = np.zeros((13, 3))
        self.plots = [[],[]] #None]#*2*len(self.I)
        for j in range(2):
            for i in np.arange( len(self.I) ):
                x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
                y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
                z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
                #self.plots[j].append(self.ax.plot(x, y, z, lw=2, c="black" if self.LR[i] else "black"))
                self.plots[j].append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))

        self.plots_ = [[],[]]
        for j in range(2):
            for i in np.arange( len(self.I) ):
                x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
                y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
                z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
                self.plots_[j].append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")


    def update(self, f, channels, channels2=None, lcolor="lightcoral", rcolor="cornflowerblue"): #"#3498db", rcolor="#e74c3c"):
        assert channels.size == 78 or channels.size == 39, "channels should have 39 entries for 1p or 78 for 2p, it has %d instead" % channels.size
        vals_ = np.reshape( channels, (26, -1) )
        vals_l = vals_[:13,:]
        vals_f = vals_[13:,:]
        for j in range(2):
            vals = [vals_l,vals_f][j]
            for i in np.arange( len(self.I) ):
                x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
                y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
                z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
                self.plots[j][i][0].set_xdata(x)
                self.plots[j][i][0].set_ydata(y)
                self.plots[j][i][0].set_3d_properties(z)

                if j ==0: #in, gt
                    self.plots[j][i][0].set_color("mediumaquamarine" if self.LR[i] else "plum")
                else:
                    self.plots[j][i][0].set_color('lightcoral' if self.LR[i] else 'cornflowerblue')
                    #self.plots[j][i][0].set_color(lcolor if self.LR[i] else rcolor)
        if channels2 is not None:
            vals2_ = np.reshape( channels2, (26, -1) )
            vals2_l = vals2_[:13,:]
            vals2_f = vals2_[13:,:]
            for j in range(2): # pred
                vals2 = [vals2_l, vals2_f][j]
                for i in np.arange( len(self.I) ):
                    x = np.array( [vals2[self.I[i], 0], vals2[self.J[i], 0]] )
                    y = np.array( [vals2[self.I[i], 1], vals2[self.J[i], 1]] )
                    z = np.array( [vals2[self.I[i], 2], vals2[self.J[i], 2]] )
                    self.plots_[j][i][0].set_xdata(x)
                    self.plots_[j][i][0].set_ydata(y)
                    self.plots_[j][i][0].set_3d_properties(z)
                    if j == 0:
                        self.plots_[j][i][0].set_color('darkgreen' if self.LR[i] else 'indigo') #darkred' if self.LR[i] else 'darkblue')
                    elif j == 1:
                        self.plots_[j][i][0].set_color('darkred' if self.LR[i] else 'darkblue')
        if f is 0:
            r = 750
            #xroot, yroot, zroot = vals_[0,0], vals_[0,1], vals_[0,2]
            ##self.x, self.y, self.z = [xroot, xroot], [zroot, zroot], [yroot, yroot]
            #self.x, self.y, self.z = [-r+xroot, r+xroot], [-r+zroot, r+zroot], [-r+yroot, r+yroot]
        self.ax.set_xlim3d([-1000,1000])#self.x)
        self.ax.set_zlim3d([-1000,1000])#self.y)
        self.ax.set_ylim3d([-1000,1000])#self.z)


def vis_pi(p3d,save_path):
    #input: p3d np array (seq_len, 13, 3)
    num_frames = len(p3d)
    metadata = dict(title='01', artist='Matplotlib',comment='motion')
    writer = FFMpegWriter(fps=10, metadata=metadata)
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ob = PI3DPose13kpts(ax)
    with writer.saving(fig, save_path, 100):
        f = 0
        for i in tqdm(range(num_frames)):
            ob.update(f, p3d[i])
            plt.savefig('./tmp/'+str(i)+'.jpg')
            exit()
            writer.grab_frame()
            plt.pause(0.01)
            f += 1

def vis_pi_compare(p3d_gt,p3d_pred,save_path):
    num_frames_gt = len(p3d_gt) #75
    num_frames_pred = len(p3d_pred) #25
    p3d_gt = p3d_gt.reshape((num_frames_gt,-1))
    p3d_pred = p3d_pred.reshape((num_frames_pred,-1))
    metadata = dict(title='01', artist='Matplotlib',comment='motion')
    writer = FFMpegWriter(fps=10, metadata=metadata)
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ob = PI3DPose13kpts(ax)
    with writer.saving(fig, save_path, 100):
        f = 0
        for i in tqdm(range(num_frames_gt - num_frames_pred)):
            ob.update(f, p3d_gt[i])
            #plt.savefig('./tmp/'+str(i)+'.jpg')
            writer.grab_frame()
            plt.pause(0.01)
            f += 1
        for i in tqdm(range(num_frames_gt - num_frames_pred,num_frames_gt)):
            #ob.update(p3d_pred[i-num_frames_gt+num_frames_pred])
            #ob.update(f, p3d_gt[i])
            ob.update(f, p3d_gt[i], p3d_pred[i-num_frames_gt+num_frames_pred])
            #plt.savefig('./tmp/'+str(i)+'.jpg')
            writer.grab_frame()
            plt.pause(0.01)
            f += 1


