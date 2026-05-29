#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_one_seed.py
---------------
Helper invoked once per seed by Cell 2b.

Reads the seed's cached lightcone, rebuilds the matching lightconer,
and calls run_pipeline() from run_lightcone.py.

Config comes from env vars set by the parent (Cell 2b):
    SIMPLEGEN_SEED, SIMPLEGEN_BOX_LEN, SIMPLEGEN_HII_DIM, SIMPLEGEN_MH_CUT
plus CLI args for the sim parameters needed to reconstruct lightconer.
"""

import argparse
import os
import sys
import time

import numpy as np
import py21cmfast as p21c
from astropy.cosmology import FlatLambdaCDM

# script-relative import (no hardcoded paths)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_lightcone import run_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lc-path",               required=True,
                   help="path to the seed's lightcone.h5")
    p.add_argument("--z-min",                 type=float, required=True)
    p.add_argument("--z-max",                 type=float, required=True)
    p.add_argument("--z-step-factor",         type=float, required=True)
    p.add_argument("--hii-dim",               type=int,   required=True)
    p.add_argument("--box-len",               type=float, required=True)
    p.add_argument("--n-threads",             type=int,   default=32)
    p.add_argument("--sampler-min-mass",      type=float, required=True)
    p.add_argument("--sampler-buffer-factor", type=float, required=True)
    p.add_argument("--z-heat-max",            type=float, required=True)
    args = p.parse_args()

    seed = int(os.environ["SIMPLEGEN_SEED"])
    t0   = time.time()

    print("=" * 70)
    print(f"run_one_seed.py — seed {seed}")
    print("=" * 70)
    print(f"  lightcone : {args.lc_path}")
    print(f"  BOX_LEN   : {args.box_len}  HII_DIM: {args.hii_dim}")
    print(f"  z range   : {args.z_min} → {args.z_max} "
          f"(step {args.z_step_factor})")

    # ── reconstruct InputParameters identically to cell 2's worker ─
    node_redshifts_custom = np.array(
        p21c.get_logspaced_redshifts(
            min_redshift  = args.z_min,
            max_redshift  = args.z_max,
            z_step_factor = args.z_step_factor,
        )
    )

    inputs = p21c.InputParameters(
        node_redshifts     = node_redshifts_custom,
        random_seed        = seed,
        simulation_options = p21c.SimulationOptions(
            HII_DIM               = args.hii_dim,
            BOX_LEN               = args.box_len,
            N_THREADS             = args.n_threads,
            Z_HEAT_MAX            = args.z_heat_max,
            SAMPLER_MIN_MASS      = args.sampler_min_mass,
            SAMPLER_BUFFER_FACTOR = args.sampler_buffer_factor,
        ),
        matter_options = p21c.MatterOptions(
            KEEP_3D_VELOCITIES       = True,
            USE_INTERPOLATION_TABLES = 'hmf-interpolation',
        ),
        astro_options = p21c.AstroOptions(
            INHOMO_RECO  = True,
            USE_TS_FLUCT = True,
        ),
    )

    # ── rebuild lightconer (same params as halo_lightcone_worker.py) ─
    lightconer = p21c.RectilinearLightconer.between_redshifts(
        min_redshift = min(inputs.node_redshifts) + 0.1,
        max_redshift = max(inputs.node_redshifts) - 0.1,
        quantities   = (
            "brightness_temp",
            "density",
            "neutral_fraction",
            "kinetic_temperature",
            "velocity_z",
        ),
        resolution   = inputs.simulation_options.cell_size,
    )

    # ── load cached lightcone ──────────────────────────────────────
    lightcone = p21c.LightCone.from_file(args.lc_path, safe=False)
    print(f"  loaded lightcone: {len(lightcone.lightcone_redshifts)} slices")

    # ── cosmology (match cell 1) ───────────────────────────────────
    cosmo = FlatLambdaCDM(H0=67.77, Om0=0.3086,Ob0=0.0489, Tcmb0=2.7255)

    # ── run the SiMPLE-Gen pipeline ────────────────────────────────
    run_pipeline(lightcone, lightconer, inputs, cosmo=cosmo)

    print(f"\n✓ seed {seed} done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
