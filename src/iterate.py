#!/usr/bin/python

import numpy as np
import scipy.spatial as spt

import make_ptcls 
import pcs_snap
import vel_plot

import sys

def axisymmetrize(cp_src, with_z = False):

  # find pos & vel in cylindrical coord.
  R = np.linalg.norm(cp_src[:, :2, axis = 1)
  Z, Vz = cp_src[:, 2], cp_src[:, 5]

  cos_phi, sin_phi = cp_src[:, 0] / R, cp_src[:, 1] / R
  Vr  = cp_src[:, 3] * cos_phi + cp_src[:, 4] * sin_phi
  Vaz = cp_src[:, 4] * cos_phi - cp_src[:, 3] * sin_phi

  # generate random angle
  xi = np.random.randn(cp_src.shape[0])
  yi = np.random.randn(cp_src.shape[0])
  ri = np.sqrt(xi ** 2 + yi ** 2)
  cos_psi, sin_psi = xi / ri, yi / ri

  # generate new model
  X, Y = R * cos_psi, R * sin_psi
  Vx = Vr  * cos_psi - Vaz * sin_psi
  Vy = Vaz * cos_psi + Vr  * sin_psi

  if with_z:
    idmsk_inv = np.random.random(R.size) < 0.5
    Z  = np.where(idmsk_inv,  Z,  -Z)
    Vz = np.where(idmsk_inv, Vz, -Vz)

  return np.vstack((X, Y, Z, Vx, Vy, Vz)).T

def pcs_mix(pcs1, pcs2):
  return np.concatenate((pcs1, pcs2), axis = 0)

# for positions in cp_new, sample velocity;
def resample_density_cart(cp_src, cp_new):

  # construct kd-tree
  print "Making tree..."
  kdt = spt.cKDTree(cp_src[:, :3], leafsize = 10)

  # prepare arrays
  print "Making arrays..."
  N_taken = np.ones(shape = cp_src.shape[0], dtype = int)
  V_new   = np.empty(shape = (cp_new.shape[0], 3), dtype = 'f4')

  # loop over ptcls and assign vels
  print "Resampling velocities..."
  for I_pt, W_pt in enumerate(cp_new):

    print '\r', I_pt, '...',

    # search for nearest neighbours
    D_t, Id_t = kdt.query(W_pt[:3], k = 10)

    # How many times are they used
    Ntk_t = N_taken[Id_t]

    # determine which is the best candidate
    Id_min = Id_t[np.argmin(Ntk_t * D_t)]

    # transfer velocity
    V_new[I_pt] = cp_src[Id_min][3:]
    N_taken[Id_min] += 1

  print ''

  pcs_new = np.empty(cp_new.shape, dtype = 'f4')
  pcs_new[:, :3] = cp_new[:, :3]
  pcs_new[:, 3:] = V_new

  return pcs_new

# TEST Vel_transfer
if 0:

  snap = pcs_snap.read_snap_galaxy_pcs('run999.pcs0')
  pcs1 = snap['cps']['C1']['pcs']
  pcs2 = snap['cps']['C2']['pcs']

  pcs1_azmix  = axisymmetrize(pcs1)
  pcs1_vtrans = resample_density_cart(pcs1, pcs1_azmix)

  import matplotlib.pyplot as plt
  from matplotlib.colors import LogNorm

  fig = plt.figure(figsize = (18., 6.))

  ax1 = fig.add_subplot(1, 3, 1)
  ax1.hist2d(pcs1[:, 0], pcs1[:, 1], bins = 64, norm = LogNorm())

  ax2 = fig.add_subplot(1, 3, 2)
  ax2.hist2d(pcs1_azmix[:, 0], pcs1_azmix[:, 1],
             bins = 64, norm = LogNorm())

  ax3 = fig.add_subplot(1, 3, 3)
  ax3.hist2d(pcs1_vtrans[:, 0], pcs1_vtrans[:, 1],
             bins = 64, norm = LogNorm())

  plt.show()

# TEST plotting utils
if 0:

  snap = pcs_snap.read_snap_galaxy_pcs('run999.pcs0')
  pcs1 = snap['cps']['C1']['pcs']
  pcs2 = snap['cps']['C2']['pcs']

  vel_plot.cylindrical_anisotropy_plot(pcs2, R_max = 4., Z_max = 4.,
    base_pixel_size = 0.2, max_lv = 3)
