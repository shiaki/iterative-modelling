#!/usr/bin/python

import sys
import random as rd
import itertools as itt

import numpy as np
import scipy.stats as stt

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import iterate

cmaps  = ['Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys',
          'Oranges', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 'Purples',
          'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']

'''
  Rotate the particles to the observer's prespective
  From: ../panofit-sandbox/make_mock.py
'''
def Rotate(sn, phase, incli, pitch):

  # Generate rotation matrix
  a, b, c = pitch, incli, phase
  # Note the order...

  print 'Rotate', a, b, c

  ca, cb, cc = np.cos(a), np.cos(b), np.cos(c)
  sa, sb, sc = np.sin(a), np.sin(b), np.sin(c)

  R1 = np.array([[ cc,  sc,  0.],
                 [-sc,  cc,  0.],
                 [ 0.,  0.,  1.]])

  R2 = np.array([[ 1.,  0.,  0.],
                 [ 0.,  cb, -sb],
                 [ 0.,  sb,  cb]])

  R3 = np.array([[ ca,  sa,  0.],
                 [-sa,  ca,  0.],
                 [ 0.,  0.,  1.]])

  R = np.dot(R3, np.dot(R2, R1))

  # fill the rotated matrix.
  x, y, z, vx, vy, vz = sn.T
  rsn = np.empty(shape = (x.size, 6), dtype = sn.dtype)

  rsn[:, 0] = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z
  rsn[:, 1] = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z
  rsn[:, 2] = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z

  rsn[:, 3] = R[0, 0] * vx + R[0, 1] * vy + R[0, 2] * vz
  rsn[:, 4] = R[1, 0] * vx + R[1, 1] * vy + R[1, 2] * vz
  rsn[:, 5] = R[2, 0] * vx + R[2, 1] * vy + R[2, 2] * vz

  return rsn

def multicomponent_losvd_plot(snap, box_size, phase, incli, pitch,
      draw_with = None, color_dict = None, cmap_dict = None, saveto = None,
      losvd_fields = None, losvd_field_uv = None, losvd_field_size = None):

  # how many components?
  N_cps    = len(snap['cps'])
  cps_keys = snap['cps'].keys()

  # gettin ready for plotting
  xt, yt = np.mgrid[-box_size:box_size:129j, -box_size:box_size:129j]
  pos_t  = np.vstack([xt.ravel(), yt.ravel()]) # pos on the celestial sph

  if cmap_dict == None:
    cmap_dict = dict(zip(cps_keys, rd.sample(cmaps, N_cps)))

  if color_dict == None:
    colors_t = [plt.get_cmap(cmap_dict[ik])(0.75) for ik in cps_keys]
    color_dict = dict(zip(cps_keys, colors_t))

  # prepare sub-fields for velocity plotting
  if losvd_fields == None: losvd_fields = 5
  if losvd_field_uv == None:
    losvd_field_uv = [[0., box_size / losvd_fields],
                      [box_size / losvd_fields, 0.]]
  if losvd_field_size == None: losvd_field_size = box_size / 15.

  field_pos = {}
  for i, j in itt.product(range(losvd_fields), range(losvd_fields)):
    field_pos[(i, j)] = {'pos': (losvd_field_uv[0][0] * i \
        + losvd_field_uv[1][0] * i, losvd_field_uv[0][1] * j \
        + losvd_field_uv[1][1] * j), 'vlos':{}}

  fig  = plt.figure(figsize = (16., 8.))
  gs_t = gs.GridSpec(losvd_fields, losvd_fields * 2)
  sigma_plot = plt.subplot(gs_t[:losvd_fields, :losvd_fields],\
                           aspect = 'equal')

  losvd_pltrg = [[0., 0.]]

  # loop over all mass components
  for I_cp, cp_key in enumerate(cps_keys):

    # rotate particles to the position
    print 'Taking component', cp_key, '...'
    pcs_t = Rotate(snap['cps'][cp_key]['pcs'], phase, incli, pitch)

    # generate projected view
    if draw_with in [None, 'hist']:
      sigma = np.histogram2d(pcs_t[:, 0], pcs_t[:, 1], bins = 32,
              range = [[-box_size, box_size], [-box_size, box_size]])[0]

    elif draw_with == 'kde':
      print 'Generating KDE', cp_key, '...'
      kde_t = stt.gaussian_kde((pcs_t.T)[:2])
      sigma = np.reshape(kde_t(pos_t).T, xt.shape)

    sigma = np.log(1. + sigma)

    # put them onto sigma_plot
    sigma_plot.contour(np.rot90(sigma), 9, origin = 'upper',
      extent = [-box_size, box_size, -box_size, box_size],
      cmap = cmap_dict[cp_key])

    # collect particle los velocity in sub-fields
    for i, j in itt.product(range(losvd_fields), range(losvd_fields)):
      f_xt, f_yt = field_pos[(i, j)]['pos']
      f_rt = (pcs_t[:, 0] - f_xt) ** 2 + (pcs_t[:, 1] - f_yt) ** 2
      losvd_t = np.extract(f_rt < losvd_field_size ** 2, pcs_t[:, 5])
      field_pos[(i, j)]['vlos'][cp_key] = losvd_t
      if losvd_t.size != 0: # there are ptcls in this component
        losvd_pltrg.append([losvd_t.min(), losvd_t.max()])

  losvd_pltrg = np.abs(np.array(losvd_pltrg)).max()
  losvd_vmin, losvd_vmax = -losvd_pltrg, losvd_pltrg

  # plot fields onto panel 1
  sigma_plot.autoscale(enable = False)
  for i, j in itt.product(range(losvd_fields), range(losvd_fields)):

    # on the left panel
    f_xt, f_yt = field_pos[(i, j)]['pos']
    circle = plt.Circle((f_xt, f_yt), losvd_field_size,\
                        color = 'k', clip_on = False, fill = False)
    sigma_plot.add_artist(circle)

    # plot losvd in the right panel
    losvd_ax = plt.subplot(gs_t[losvd_fields - 1 - j, i + losvd_fields])
    losvd_ax.get_xaxis().set_visible(False)
    losvd_ax.get_yaxis().set_visible(False)

    for k_cp, v_cp in field_pos[(i, j)]['vlos'].iteritems():
      if v_cp.size == 0: continue
      losvd_ax.hist(v_cp, bins = 16, range = (losvd_vmin, losvd_vmax),
                    color = color_dict[k_cp], histtype = 'step')

  if saveto == None: plt.show()
  else: plt.savefig(saveto, bbox_inches = 'tight')

if __name__ == '__main__':

  import pcs_snap

  snap = pcs_snap.read_snap_galaxy_pcs('./dumps/run999.pcs0')

  multicomponent_losvd_plot(snap, 2.5, 0., 0., .0,
    draw_with = 'kde', cmap_dict = {'C1': 'Blues', 'C2': 'Reds'},
    saveto = 'multicps_losvd-0.png')
