#!/usr/bin/python

import numpy      as np
import itertools  as itt

from matplotlib.colors      import rgb2hex
from matplotlib.patches     import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.cm          import ScalarMappable

class adaptive_hist2d:

  def __init__(self, X, Y,
               hist_bins = (32, 32),
               hist_lim  = 'auto',
               if_refine = lambda idx: idx.size > 128,
               N_subgrid = (2, 2),
               N_maxlv   = 5):

    self.plv, self.pix, self.piy  = None, None, None
    self.x_ax, self.y_ax          = [], []
    self.clm, self.car, self.cct  = [], [], []
    self.cid, self.clv            = [], []

    N_pts, pt_idx = X.size, np.arange(X.size)
    if hist_lim == 'auto':
      hist_lim = [[X.min(), X.max()], [Y.min(), Y.max()]]
    if N_maxlv == 0: if_refine = lambda idx: False

    x_ax_n = np.linspace(hist_lim[0][0], hist_lim[0][1], hist_bins[0] + 1)
    y_ax_n = np.linspace(hist_lim[1][0], hist_lim[1][1], hist_bins[1] + 1)

    for i_lv in xrange(N_maxlv + 1):

      #print "At level:", i_lv

      # make axes for binning
      x_ax_i, y_ax_i = x_ax_n, y_ax_n

      x_ax_n = np.linspace(hist_lim[0][0], hist_lim[0][1],\
                           hist_bins[0] * N_subgrid[0] ** (i_lv + 1) + 1)
      y_ax_n = np.linspace(hist_lim[1][0], hist_lim[1][1],\
                           hist_bins[1] * N_subgrid[1] ** (i_lv + 1) + 1)

      # make a histogram at the lowest resolution, for the first time
      if i_lv == 0:
        pix_i = np.digitize(X, x_ax_i)
        piy_i = np.digitize(Y, y_ax_i)
        plv_i = np.zeros(shape = N_pts)

      Nxc = hist_bins[0] * N_subgrid[0] ** i_lv + 1
      Nyc = hist_bins[1] * N_subgrid[1] ** i_lv + 1

      # loop over cells and test if they can be refined
      for ic_x, ic_y in itt.product(xrange(1, Nxc), xrange(1, Nyc)):
        id_msk = (plv_i == i_lv) * (pix_i == ic_x) * (piy_i == ic_y)
        id_ins = np.extract(id_msk, pt_idx)

        if id_ins.size != 0:

          # refine them to the next level if possible.
          if if_refine(id_ins) and (i_lv + 1 < N_maxlv):
            pix_i[id_ins] = np.digitize(X[id_ins], x_ax_n)
            piy_i[id_ins] = np.digitize(Y[id_ins], y_ax_n)
            plv_i[id_ins] = i_lv + 1

          else: # fix it to the current level
            clm_i = np.concatenate((x_ax_i[ic_x - 1: ic_x + 1],
                                    y_ax_i[ic_y - 1: ic_y + 1])\
                                  ).tolist() # FIXME
            self.clm.append(clm_i)
            self.car.append( (clm_i[1] - clm_i[0]) * (clm_i[3] - clm_i[2]))
            self.cct.append([(clm_i[1] + clm_i[0]) / 2.,
                             (clm_i[3] + clm_i[2]) / 2.])
            self.cid.append(pt_idx[id_msk])
            self.clv.append(i_lv)

      self.x_ax.append(x_ax_i)
      self.y_ax.append(y_ax_i)

    self.plv = plv_i
    self.pix = pix_i
    self.piy = piy_i
    # done.

  def evaluate(self,
               val_func   = lambda idx, args: float(len(idx)),
               args       = (),
               unit_area  = True):

    Nc, val = len(self.clv), []

    # calculate function value for each bin cell
    for i_c in xrange(Nc):
      val.append(val_func(self.cid[i_c], args))

    # divide by bin area, if necessary
    if unit_area:
      val = [val[i] / self.car[i] for i in xrange(Nc)]

    #print np.sum(val)

    return val, self.cct, self.clm

  def eval_plot(self,
                val_func  = lambda idx, args: float(len(idx)),
                args      = (),
                val_idx   = None,
                unit_area = True,
                cmap      = 'jet',
                cmap_norm = None,
                cmap_vmin = None,
                cmap_vmax = None):

    val, cct, clm = self.evaluate(val_func, args, unit_area)
    if val_idx != None: val = [V[val_idx] for V in val]

    return self.make_plot(val, cct, clm,
                          cmap, cmap_norm, cmap_vmin, cmap_vmax)

  def make_plot(self, val, cct, clm,
                cmap = 'jet', cmap_norm = None,
                cmap_vmin = None, cmap_vmax = None):

    # make color mapping
    smap = ScalarMappable(cmap_norm, cmap)
    smap.set_clim(cmap_vmin, cmap_vmax)
    smap.set_array(val)
    bin_colors = smap.to_rgba(val)

    # make patches
    patches = []
    for i_c, i_clm in enumerate(clm):
      patches.append(Rectangle((i_clm[0],  i_clm[2]),
                                i_clm[1] - i_clm[0],
                                i_clm[3] - i_clm[2]))
    patches_colle = PatchCollection(patches)
    patches_colle.set_edgecolor('face')
    patches_colle.set_facecolor(bin_colors)

    return patches_colle, smap

class adaptive_hist3d:

  plv, pix, piy, piz, x_ax, y_ax, z_ax = None, None, None, None, [], [], []
  clm, cvl, cct, cid, clv = [], [], [], [], []

  def __init__(self, X, Y, Z,
               hist_bins = (32, 32, 32),
               hist_lim  = 'auto',
               if_refine = lambda idx: idx.size > 256,
               N_subgrid = (2, 2, 2),
               N_maxlv   = 5):

    N_pts, pt_idx = X.size, np.arange(X.size)
    if hist_lim == 'auto':
      hist_lim = [[X.min(), X.max()],\
                  [Y.min(), Y.max()],\
                  [Z.min(), Z.max()]]
    if N_maxlv == 0: if_refine = lambda idx: False

    x_ax_n = np.linspace(hist_lim[0][0], hist_lim[0][1], hist_bins[0] + 1)
    y_ax_n = np.linspace(hist_lim[1][0], hist_lim[1][1], hist_bins[1] + 1)
    z_ax_n = np.linspace(hist_lim[2][0], hist_lim[2][1], hist_bins[2] + 1)

    for i_lv in xrange(N_maxlv + 1):

      x_ax_i, y_ax_i, z_ax_i = x_ax_n, y_ax_n, z_ax_n

      # make axes for binning on this level of refinement
      x_ax_n = np.linspace(hist_lim[0][0], hist_lim[0][1],\
                           hist_bins[0] * N_subgrid[0] ** (i_lv + 1) + 1)
      y_ax_n = np.linspace(hist_lim[1][0], hist_lim[1][1],\
                           hist_bins[1] * N_subgrid[1] ** (i_lv + 1) + 1)
      z_ax_n = np.linspace(hist_lim[2][0], hist_lim[2][1],\
                           hist_bins[2] * N_subgrid[2] ** (i_lv + 1) + 1)

      # if this is the lowest level
      if i_lv == 0:
        pix_i = np.digitize(X, x_ax_i)
        piy_i = np.digitize(Y, y_ax_i)
        piz_i = np.digitize(Z, z_ax_i)
        plv_i = np.zeros(shape = N_pts)

      # loop over cells
      Nxc = hist_bins[0] * N_subgrid[0] ** i_lv + 1
      Nyc = hist_bins[1] * N_subgrid[1] ** i_lv + 1
      Nzc = hist_bins[2] * N_subgrid[2] ** i_lv + 1

      for ic_x, ic_y, ic_z in itt.product(xrange(1, Nxc),
                                          xrange(1, Nyc), xrange(1, Nzc)):
        id_msk = (plv_i == i_lv - 1) * (pix_i == ic_x) * \
                 (piy_i == ic_y) * (pix_i == ic_z)
        id_ins = np.extract(id_msk, pt_idx)
        if if_refine(id_ins) and (i_lv < N_maxlv):
          pix_i[id_ins] = np.digitize(X[id_ins], x_ax_i)
          piy_i[id_ins] = np.digitize(Y[id_ins], y_ax_i)
          piz_i[id_ins] = np.digitize(Z[id_ins], z_ax_i)
          plv_i[id_ins] = i_lv
        else:
          clm_i = np.concatenate((self.x_ax[-1][ic_x - 1: ic_x + 1],
                                  self.y_ax[-1][ic_y - 1: ic_y + 1],
                                  self.z_ax[-1][ic_z - 1: ic_z + 1])\
                                ).tolist()
          self.clm.append(clm_i)
          self.cvl.append( (clm_i[1] - clm_i[0]) * (clm_i[3] - clm_i[2]) * \
                           (clm_i[5] - clm_i[4]) )
          self.cct.append([(clm_i[1] + clm_i[0]) / 2.,
                           (clm_i[3] + clm_i[2]) / 2.,
                           (clm_i[5] + clm_i[4]) / 2.])
          self.cid.append(pt_idx[id_msk])
          self.clv.append(i_lv)

      self.x_ax.append(x_ax_i)
      self.y_ax.append(y_ax_i)
      self.z_ax.append(z_ax_i)

  def evaluate(self,
               val_func = lambda idx, args: float(len(idx)),
               args     = (),
               unit_vol = True):

    print "Making function evaluation"
    Nc, val = len(self.clv), []

    # evaluate function for each cell
    for i_c in xrange(Nc):
      val.append(val_func(self.cid[i_c], args))

    # divide by cell volume if required
    if unit_vol:
      val = [val[i] / self.cvl[i] for i in xrange(Nc)]

    print "Done"
    return val, self.cct, self.clm


if __name__ == '__main__':

  if 0:

    from pylab import randn

    x = randn(10000)
    y = randn(10000) * 0.5

    import matplotlib.pyplot as plt
    import matplotlib.colors as clr

    hist = adaptive_hist2d(x, y, hist_bins = (5, 5), N_maxlv = 5,
                     hist_lim = [[-3, 3], [-3, 3]])
    pc   = hist.make_plot(unit_area = True, cmap_norm = clr.LogNorm())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.add_collection(pc)
    plt.show()


  data_path = '/dgalex2/qinyj/dgalex_home_backup/qinyj/workspace/proper_motion_paper_figures/proper_motion_8.5kpc_20deg.dat'


  if 0:

    data_path = '/home/qinyj/workspace/proper_motion_paper_figures/proper_motion_8.5kpc_20deg.dat'
    Lon, Lat, Dist, V_los, mu_l, mu_b = np.loadtxt(data_path, unpack = True)

    hist = adaptive_hist2d(Lon, Lat, hist_bins = (30, 15), N_maxlv = 3,
                     hist_lim = [[-30., 30.], [-15., 15.]],
                     if_refine = lambda idx: idx.size > 256,
                     N_subgrid = (2, 2))

    mean_vlos = lambda idx, args: np.mean((args[0])[idx])
    pc = hist.make_plot(val_func = mean_vlos,
                        args = (V_los,),
                        unit_area = False)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_xlim(-30., 30.)
    ax.set_ylim(-15., 15.)
    ax.add_collection(pc)
    plt.show()

'''
  def __init__(self, X, Y,
               hist_bins = (32, 32),
               hist_lim  = 'auto',
               if_refine = lambda idx: idx.size > 128,
               N_subgrid = (2, 2),
               N_maxlv   = 5):

'''
