#!/usr/bin/python

work_dir = ''

import numpy as np
from scipy.io import FortranFile as ufmt

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from matplotlib.colors import LogNorm

# See GALAXY 14.50 Manual, Sec. 9.2, P54
header_dtype = [('n1', '<i4'), ('n2', '<i4'), ('n3', '<i4'),
                ('ncoor', '<i4'), ('np', '<i4'), ('time', '<f4'),
                ('pm', '<f4'), ('pertbn', '<i4')]

def save_snap_galaxy_pcs(filename, snap):

  # unpack snapshot
  cps = snap['cps']

  # Get ptcl number
  n1, n2, n3 = 0, 0, 0
  if(cps.has_key('C1')): n1 = cps['C1']['N_pcs']
  if(cps.has_key('C2')): n2 = cps['C2']['N_pcs']
  if(cps.has_key('C3')): n3 = cps['C3']['N_pcs']
  N_pcs = n1 + n2 + n3

  # Make array
  pcs = np.empty(shape = (N_pcs, 6), dtype = 'f4')
  if n1 != 0: pcs[:n1] = cps['C1']['pcs']
  if n2 != 0: pcs[n1: n1 + n2] = cps['C2']['pcs']
  if n3 != 0: pcs[n1 + n2: n1 + n2 + n3] = cps['C1']['pcs']

  # prepare header,
  header = np.empty(1, dtype = header_dtype)
  header[0]['n1']     = n1
  header[0]['n2']     = n2
  header[0]['n3']     = n3
  header[0]['ncoor']  = 6
  header[0]['np']     = 5000
  header[0]['time']   = snap['time']
  header[0]['pm']     = snap['pm']
  header[0]['pertbn'] = 0

  # open a file, write the header
  pcs_fs = ufmt(filename, 'w')
  pcs_fs.write_record(header)

  # write pcs array in batches of 5k ptcls
  N_put, chunk_size = 0, 5000 * 6
  pcs = pcs.reshape((-1,)) # into 1d array
  while N_put < N_pcs * 6:
    chunk_t = pcs[N_put: N_put + chunk_size]
    pcs_fs.write_record(chunk_t)
    N_put += chunk_t.size

  pcs_fs.close()

  return 0

def read_snap_galaxy_pcs(filename):

  pcs_ds = ufmt(filename, 'r')
  header = pcs_ds.read_record(dtype = header_dtype)[0]

  # read header info / GALAXY 14.50 Manual, 9.2
  n1, n2, n3 = header['n1'], header['n2'], header['n3']
  N_pcs      = n1 + n2 + n3
  chunk_size = header['ncoor'] * header['np']

  # assume 3D problem with equal-mass particles for each component
  assert header['ncoor'] == 6

  # read ptcls in batches
  N_get = 0
  pcs = np.empty(N_pcs * 6, dtype = 'f4')
  while N_get < N_pcs * 6:
    chunk_t = pcs_ds.read_reals(dtype = 'f4')
    pcs[N_get: N_get + chunk_size] = chunk_t
    N_get += chunk_t.size
  pcs = pcs.reshape((-1, 6))

  pcs_ds.close()

  # Make them into components
  snap = {'cps' : {},
          'pm'  : header['pm'],
          'time': header['time']}

  if n1 != 0: # component 1 has mtcls
    snap['cps']['C1'] = {'N_pcs': n1,
                         'pm'   : header['pm'],
                         'pcs'  : pcs[:n1]}

  if n2 != 0: # component 2 has ptcls
    snap['cps']['C2'] = {'N_pcs': n2,
                         'pm'   : header['pm'],
                         'pcs'  : pcs[n1: n1 + n2]}

  if n3 != 0: # component 3 has ptcls
    snap['cps']['C3'] = {'N_pcs': n3,
                         'pm'   : header['pm'],
                         'pcs'  : pcs[n1 + n2: n1 + n2 + n3]}

  return snap


# diff test
if False:

  import os # for diff

  dic = read_snap_galaxy_pcs('run999.pcs0')
  save_snap_galaxy_pcs('test.pcs0', dic)

  df = os.system('diff run999.pcs0 test.pcs0')
  if(df): print "diff test failed."
  else:   print "diff test passed."
