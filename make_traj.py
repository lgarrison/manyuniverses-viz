#!/usr/bin/env python3

from pathlib import Path
import os

import asdf
import numpy as np
import astropy.table
import Corrfunc

NTHREAD = len(os.sched_getaffinity(0))

def read_all(dir, cen=np.array([-960.33453, -176.59247, -903.21063]), width=30):
    dir = Path(dir)
    sims = []
    allpairs = []
    for fn in sorted(dir.glob('*cut40.0.asdf')):
        with asdf.open(fn) as af:
            t = af['particles']
            pairs = af['pairs']

        t['pos'] -= cen  # work in [-w/2,w/2)
        
        # import a few more params from the cosmologies table
        cnum = t.meta['SimName'].split('_')[2][1:]
        root = f'abacus_cosm{cnum}'
        ic = list(cosmologies['root']).index(root)
        t.meta['sigma8_m'] = cosmologies[ic]['sigma8_m']
        t.meta['sigma8_cb'] = cosmologies[ic]['sigma8_cb']
        t.meta['alpha_s'] = cosmologies[ic]['alpha_s']
        
        sims += [t]
        allpairs += [pairs]
    
    return sims, allpairs


def do_slab_RR_and_stack(pairs, nthread=NTHREAD):
    pair = pairs[0]
    domain = np.array(pair.meta['domain_DD'])

    rng = np.random.default_rng()
    nrand = 10**8
    rand = rng.uniform(0, 1, size=(nrand,3))*domain

    bins = np.empty(len(pair)+1)
    bins[:-1] = pair['rmin']
    bins[-1] = pair['rmax'][-1]

    print(f'Using {nthread=}')
    rand_res = Corrfunc.theory.DD(1, nthread, bins, *rand.T,
                periodic=False, verbose=True, bin_type='lin',
            )
    RR = rand_res['npairs'] * (pair.meta['N_DD'] / nrand)**2

    t = astropy.table.Table(pair)  # len(nbin)
    t['npair'] = np.stack([p['npairs'] for p in pairs], axis=1)
    t['RR'] = RR

    return t  # shape (nbin, nstate)


if __name__ == '__main__':
    cosmologies = astropy.table.Table.read('cosmologies.csv')

    sims, pairs = read_all('/mnt/home/lgarrison/ceph/multicosmo-viz/cutouts/')

    pair_traj = do_slab_RR_and_stack(pairs)  # shape (nbin, nstate)

    pid_c000 = sims[0]['pid']
    for sim in sims[1:]:
        isect, comm1, comm2 = np.intersect1d(pid_c000, sim['pid'], assume_unique=True, return_indices=True)
        pid_c000 = pid_c000[comm1]
        print(f'{len(pid_c000)/len(sims[0])*100:.4g}%', end='  ')
    print()

    traj = np.empty((len(pid_c000),len(sims),3), dtype=np.float32)

    for i,sim in enumerate(sims):
        isect, comm1, comm2 = np.intersect1d(pid_c000, sim['pid'], assume_unique=True, return_indices=True)
        assert(len(comm2) == len(traj))
        traj[:,i] = sim['pos'][comm2]

    traj = astropy.table.Table(dict(pos=traj), meta={i:sims[i].meta for i in range(len(sims))})

    with asdf.AsdfFile(dict(particles=traj, pairs=pair_traj)) as af:
        af.write_to('/mnt/home/lgarrison/ceph/multicosmo-viz/traj.asdf')
