#!/usr/bin/env python3
'''
cutout a box from the given sim
'''

import os
import argparse
from pathlib import Path

import numpy as np
import asdf

import multicosmo

CEN = np.array([-960.33453, -176.59247, -903.21063])
WIDTH = 40.
OUT = Path(os.environ['HOME']) / 'ceph/multicosmo-viz/cutouts'
NTHREAD = len(os.sched_getaffinity(0))

def main(sim, out=OUT, cen=CEN, width=WIDTH, nthread=NTHREAD):
    sim = Path(sim)
    arrwidth = np.array([width,width,width])
    p, pairs = multicosmo.cutout_sim_particles(sim, cen, arrwidth, nthread=nthread)
    name = p.meta['SimName']

    af = asdf.AsdfFile(dict(particles=p, pairs=pairs))
    outfn = out / f'{name}-cut{width}.asdf'
    print(f'Writing to {outfn}')
    af.write_to(outfn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sim', help='Simulation dir')
    parser.add_argument('--out', help='Output dir', default=OUT)
    parser.add_argument('--cen', default=CEN)
    parser.add_argument('--width', default=WIDTH, type=float)
    parser.add_argument('--nthread', default=NTHREAD, type=int)

    args = parser.parse_args()
    args = vars(args)

    main(**args)
