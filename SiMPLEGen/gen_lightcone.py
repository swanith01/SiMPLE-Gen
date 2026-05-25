#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_lightcone.py
----------------
Replacement for gen.py that reads from a py21cmfast lightcone
instead of a single coeval box.

For each node redshift snapshot, it:
  - loads the slab-filtered halo catalogue
  - extracts LoS sightlines of n_HI, T_k, v_z from the lightcone
    at each halo's (x, y) position
  - rolls each sightline so the halo sits at the centre
  - concatenates all snapshots into single arrays

Output arrays have the same shape and meaning as gen.py:
  n_HI_halo  : (N_halo_total, N_los)  [cm^-3]
  T_halo     : (N_halo_total, N_los)  [K]
  v_pec_halo : (N_halo_total, N_los)  [km/s]
  halomass   : (N_halo_total,)        [Msun]
  halo_coords: (N_halo_total, 3)      [cMpc]
  halo_z     : (N_halo_total,)        redshift of each halo
  x_sim      : (N_los,)               comoving coordinate [cMpc]
  z_grid     : (N_los,)               redshift along LoS
"""

import os
import time
import numpy as np
from astropy.units import pixel
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy.constants import c

from .config import PATHS, HALO_CATALOGUE_DIR, MH_CUT, BOX_LEN, HII_DIM, Z_REDSHIFT


def run_gen_lightcone(lightcone, lightconer, inputs):
    """
    Parameters
    ----------
    lightcone : py21cmfast Lightcone object
        Must contain 'neutral_fraction', 'density',
        'kinetic_temperature', 'velocity_z' fields.
    lightconer : py21cmfast RectilinearLightconer object
        Used to compute slab indices.
    inputs : py21cmfast InputParameters object
        Used for box geometry.
    """
    t0 = time.time()

    # ── geometry ──────────────────────────────────────────────────────────
    cell_size_mpc = BOX_LEN / HII_DIM
    z_lc          = lightcone.lightcone_redshifts
    n_los         = len(z_lc)
    lcpix         = lightconer.get_lc_distances_in_pixels(
                        inputs.simulation_options.cell_size)

    # ── lightcone field arrays ─────────────────────────────────────────────
    # shape: (HII_DIM, HII_DIM, n_los)
    xHI_lc  = lightcone.lightcones['neutral_fraction']   # neutral fraction
    Tk_lc   = lightcone.lightcones['kinetic_temperature'] # K
    vz_lc   = lightcone.lightcones['velocity_z']          # internal units
    dens_lc = lightcone.lightcones['density']              # overdensity delta

    # ── comoving x_sim grid along LoS ─────────────────────────────────────
    # each lightcone slice is one cell_size_mpc step
    x_sim = np.arange(n_los) * cell_size_mpc   # cMpc

    # ── redshift grid along LoS (already z_lc) ────────────────────────────
    z_grid = z_lc.copy()

    # ── load all node catalogues ───────────────────────────────────────────
    halo_files = sorted(
        [f for f in os.listdir(HALO_CATALOGUE_DIR) if f.startswith('masses')],
        key=lambda f: float(f.replace('masses_z', '').replace('.npy', ''))
    )
    node_z_sorted = np.array([
        float(f.replace('masses_z', '').replace('.npy', ''))
        for f in halo_files
    ])

    print(f"[gen_lightcone] {len(node_z_sorted)} snapshots, "
          f"n_los={n_los}, cell={cell_size_mpc:.3f} cMpc")

    # ── accumulators ──────────────────────────────────────────────────────
    all_nHI    = []
    all_Tk     = []
    all_vpec   = []
    all_mass   = []
    all_coords = []
    all_z      = []

    i_center = n_los // 2   # index to roll each halo's sightline to

    for z_node in node_z_sorted:

        # find lightcone slice index for this snapshot
        z_idx  = np.argmin(np.abs(z_lc - z_node))

        # find which z-cell of the raw box this slice samples
        lcidx  = int((lcpix.max() - lcpix[z_idx] + 1 * pixel).to_value(pixel))
        z_cell = (-lcidx + lightconer.index_offset) % HII_DIM
        z_lo   = z_cell * cell_size_mpc
        z_hi   = z_lo + cell_size_mpc

        # load catalogue
        tag    = f"z{z_node:.4f}"
        masses = np.load(os.path.join(HALO_CATALOGUE_DIR, f"masses_{tag}.npy"))
        coords = np.load(os.path.join(HALO_CATALOGUE_DIR, f"coords_{tag}.npy"))

        # mass cut
        mass_mask = masses >= 10.0**MH_CUT
        masses    = masses[mass_mask]
        coords    = coords[mass_mask]

        # slab filter — only halos co-spatial with this lightcone slice
        depth_mask = (coords[:, 2] >= z_lo) & (coords[:, 2] < z_hi)
        masses     = masses[depth_mask]
        coords     = coords[depth_mask]

        if len(masses) == 0:
            continue

        # grid indices of each halo in the transverse plane
        xi = np.clip((coords[:, 0] / cell_size_mpc).astype(int), 0, HII_DIM - 1)
        yi = np.clip((coords[:, 1] / cell_size_mpc).astype(int), 0, HII_DIM - 1)

        # extract LoS sightlines at each halo's (x, y) position
        # shape: (N_halo_this_snap, n_los)
        nHI_snap  = xHI_lc[xi, yi, :]    # neutral fraction (proxy for n_HI)
        Tk_snap   = Tk_lc[xi, yi, :]     # kinetic temperature [K]
        vpec_snap = vz_lc[xi, yi, :]     # LoS velocity

        # roll each sightline so this snapshot's z_idx is at i_center
        shift = i_center - z_idx
        nHI_snap  = np.roll(nHI_snap,  shift, axis=1)
        Tk_snap   = np.roll(Tk_snap,   shift, axis=1)
        vpec_snap = np.roll(vpec_snap, shift, axis=1)

        all_nHI.append(nHI_snap)
        all_Tk.append(Tk_snap)
        all_vpec.append(vpec_snap)
        all_mass.append(masses)
        all_coords.append(coords)
        all_z.append(np.full(len(masses), z_node))

        print(f"  z={z_node:.3f}  z_idx={z_idx}  "
              f"z_cell={z_cell}  halos={len(masses):,}")

    # ── concatenate all snapshots ──────────────────────────────────────────
    n_HI_halo  = np.vstack(all_nHI).astype(np.float32)
    T_halo     = np.vstack(all_Tk).astype(np.float32)
    v_pec_halo = np.vstack(all_vpec).astype(np.float32)
    halomass   = np.concatenate(all_mass).astype(np.float64)
    halo_coords= np.vstack(all_coords).astype(np.float32)
    halo_z     = np.concatenate(all_z).astype(np.float32)

    print(f"\n[gen_lightcone] total halos : {len(halomass):,}")
    print(f"  n_HI_halo  shape : {n_HI_halo.shape}")
    print(f"  T_halo     shape : {T_halo.shape}")
    print(f"  v_pec_halo shape : {v_pec_halo.shape}")

    # ── save ──────────────────────────────────────────────────────────────
    np.save(PATHS["n_HI_halo"],   n_HI_halo)
    np.save(PATHS["T_halo"],      T_halo)
    np.save(PATHS["v_pec_halo"],  v_pec_halo)
    np.save(PATHS["halomass"],    halomass)
    np.save(PATHS["halo_coords"], halo_coords)
    np.save(PATHS["halo_z"],      halo_z)
    np.save(PATHS["x_sim"],       x_sim)
    np.save(PATHS["z_grid"],      z_grid)

    dt = time.time() - t0
    print(f"\n[gen_lightcone] done in {dt:.1f}s")
    print(f"  n_HI_halo  → {PATHS['n_HI_halo']}")
    print(f"  T_halo     → {PATHS['T_halo']}")
    print(f"  v_pec_halo → {PATHS['v_pec_halo']}")
    print(f"  halomass   → {PATHS['halomass']}")
    print(f"  x_sim      → {PATHS['x_sim']}")
    print(f"  z_grid     → {PATHS['z_grid']}")


if __name__ == "__main__":
    print("Run this from your notebook via run_lightcone.py")
