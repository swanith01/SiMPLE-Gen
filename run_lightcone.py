#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_lightcone.py
----------------
Master pipeline for SiMPLE-Gen using py21cmfast lightcone inputs.
Processes one redshift snapshot at a time, with per-snapshot
checkpointing so a crash mid-run loses at most one snapshot.

Configuration via env vars (defaults in config.py):
    SIMPLEGEN_SEED, SIMPLEGEN_BOX_LEN, SIMPLEGEN_HII_DIM,
    SIMPLEGEN_MH_CUT, SIMPLEGEN_HALO_DIR
"""

import os
import sys
import numpy as np
from astropy.units import pixel
from astropy.cosmology import Planck18 as cosmo_default
from astropy import units as u
from astropy.constants import c as c_light

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import SiMPLEGen.config as cfg
from SiMPLEGen.config    import (PATHS, HALO_CATALOGUE_DIR,
                                  MH_CUT, BOX_LEN, HII_DIM)
from SiMPLEGen.spec      import run_spec
from SiMPLEGen.abundance import run_abundance
from SiMPLEGen.assign    import run_assign
from SiMPLEGen.damping   import run_damping

# LoS cells kept around each halo centre
N_CELLS  = 64
# proton mass [g]
M_PROTON = 1.6726e-24


def run_pipeline(lightcone, lightconer, inputs, cosmo=None):
    """
    Run the SiMPLE-Gen LAE pipeline snapshot by snapshot, with
    per-snapshot checkpoints.

    Parameters
    ----------
    lightcone  : py21cmfast Lightcone object
    lightconer : py21cmfast RectilinearLightconer object
    inputs     : py21cmfast InputParameters object
    cosmo      : astropy cosmology, optional (defaults to Planck18).
    """
    if cosmo is None:
        cosmo = cosmo_default

    print("=" * 60)
    print("SiMPLE-Gen Lightcone Pipeline (per-snapshot, checkpointed)")
    print("=" * 60)
    print(f"  SEED      : {cfg.SEED}")
    print(f"  BOX_LEN   : {BOX_LEN} cMpc")
    print(f"  HII_DIM   : {HII_DIM}")
    print(f"  MH_CUT    : {MH_CUT}  (log10 M/Msun)")
    print(f"  HALO_DIR  : {HALO_CATALOGUE_DIR}")
    print(f"  N_CELLS   : {N_CELLS} (LoS window per halo)")

    # ── geometry ──────────────────────────────────────────────────
    cell_size_mpc = BOX_LEN / HII_DIM
    z_lc          = lightcone.lightcone_redshifts
    n_los         = len(z_lc)
    lcpix         = lightconer.get_lc_distances_in_pixels(
                        inputs.simulation_options.cell_size)
    i_center      = n_los // 2

    # ── lightcone field arrays ─────────────────────────────────────
    xHI_lc  = lightcone.lightcones['neutral_fraction']
    Tk_lc   = lightcone.lightcones['kinetic_temperature']
    vz_lc   = lightcone.lightcones['velocity_z']
    dens_lc = lightcone.lightcones['density']

    lo = i_center - N_CELLS // 2
    hi = i_center + N_CELLS // 2

    # ── node catalogue list ────────────────────────────────────────
    halo_files = sorted(
        [f for f in os.listdir(HALO_CATALOGUE_DIR)
         if f.startswith('masses')],
        key=lambda f: float(f.replace('masses_z', '').replace('.npy', ''))
    )
    node_z_sorted = np.array([
        float(f.replace('masses_z', '').replace('.npy', ''))
        for f in halo_files
    ])

    # ──────────────────────────────────────────────────────────────────
    # DENSE LIGHTCONE (Option-A port from halo_lightcone_worker.py)
    #
    # The old loop iterated over the ~63 node catalogues and processed ONE
    # slab per node (z_idx = argmin), leaving every other LC slice empty —
    # hence the striped LAE lightcone.  Here we instead iterate over EVERY
    # LC slice (n_los of them).  Each slice is assigned to its nearest node
    # in comoving distance, and its own one-cell slab is derived from that
    # node's full-box catalogue.  All 32 slabs of each box get LAEfied, one
    # slab at a time, so the LAE lightcone fills continuously.
    #
    # node catalogues are loaded lazily and cached one-at-a-time (LC slices
    # are processed in redshift order, so consecutive slices usually share
    # a node — at most one node catalogue is held in memory).
    # ──────────────────────────────────────────────────────────────────
    node_dc = np.array(
        [cosmo.comoving_distance(zn).to_value('Mpc')
         for zn in node_z_sorted], dtype=np.float64)
    lc_dc   = np.array(lightcone.lightcone_distances.to_value('Mpc'),
                       dtype=np.float64)
    # owner[z_idx] = index into node_z_sorted
    slice_owner = np.argmin(
        np.abs(lc_dc[:, None] - node_dc[None, :]), axis=1)

    # ── output dirs ────────────────────────────────────────────────
    out_dir  = os.path.join(os.path.dirname(PATHS["halomass"]),
                            "lightcone_lae")
    snap_dir = os.path.join(out_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    print(f"\n  {n_los} LC slices to process "
          f"(over {len(node_z_sorted)} node catalogues)")
    print(f"  LoS window: {N_CELLS} cells centred on halo  "
          f"({N_CELLS * cell_size_mpc:.1f} cMpc)")
    print(f"  checkpoints → {snap_dir}\n")

    # one-node-at-a-time catalogue cache
    _cache = {"node": -1, "masses": None, "coords": None}

    def _load_node(node_idx):
        """Load + mass-cut a node's full-box catalogue, caching the last one."""
        if _cache["node"] == node_idx:
            return _cache["masses"], _cache["coords"]
        z_node = node_z_sorted[node_idx]
        tag    = f"z{z_node:.4f}"
        m = np.load(os.path.join(HALO_CATALOGUE_DIR, f"masses_{tag}.npy"))
        c = np.load(os.path.join(HALO_CATALOGUE_DIR, f"coords_{tag}.npy"))
        keep = m >= 10.0**MH_CUT
        _cache.update(node=node_idx, masses=m[keep], coords=c[keep])
        return _cache["masses"], _cache["coords"]

    # ── per-LC-slice loop ──────────────────────────────────────────
    for z_idx in range(n_los):

        node_idx = int(slice_owner[z_idx])
        z_node   = float(node_z_sorted[node_idx])
        # z_slice = this LC slice's TRUE lightcone redshift. The slice's halo
        # SLAB is drawn from the owning node's box (z_node), but the slice
        # itself physically sits at z_slice. All slice physics (z_grid for tau,
        # cfg.Z_REDSHIFT, nH_mean) and the redshifts.npy label use z_slice.
        # z_node survives only as a node property: catalogue + checkpoint names.
        z_slice  = float(z_lc[z_idx])

        ckpt = os.path.join(snap_dir,
                            f"snap_{z_idx:04d}_z{z_node:.4f}.npz")

        # ── skip if this LC slice already checkpointed ─────────────
        if os.path.exists(ckpt):
            print(f"── LC slice {z_idx+1}/{n_los}  "
                  f"z_node={z_node:.4f} ── already done, skipping")
            continue

        print(f"── LC slice {z_idx+1}/{n_los}  "
              f"z_slice={z_slice:.4f}  (node {node_idx}, "
              f"z_node={z_node:.4f}) ──")

        # this slice's one-cell slab within the owning node's box
        lcidx  = int((lcpix.max() - lcpix[z_idx] + 1 * pixel)
                     .to_value(pixel))
        z_cell = (-lcidx + lightconer.index_offset) % HII_DIM
        z_lo   = z_cell * cell_size_mpc
        z_hi   = z_lo + cell_size_mpc

        # load owning node's full-box catalogue (cached) and slab-filter
        masses_box, coords_box = _load_node(node_idx)

        depth_mask = ((coords_box[:, 2] >= z_lo) &
                      (coords_box[:, 2] < z_hi))
        masses     = masses_box[depth_mask]
        coords     = coords_box[depth_mask]

        if len(masses) == 0:
            print(f"  no halos in slab — writing empty checkpoint\n")
            np.savez(ckpt, empty=True, z_node=z_node, z_slice=z_slice)
            continue

        print(f"  halos={len(masses):,}  z_cell={z_cell}  "
              f"slab=[{z_lo:.2f}, {z_hi:.2f}] cMpc")

        # transverse grid indices
        xi = np.clip((coords[:, 0] / cell_size_mpc).astype(int),
                     0, HII_DIM - 1)
        yi = np.clip((coords[:, 1] / cell_size_mpc).astype(int),
                     0, HII_DIM - 1)

        # extract sightlines, roll halo to centre, truncate
        shift     = i_center - z_idx
        xHI_snap  = np.roll(xHI_lc[xi, yi, :], shift, axis=1)[:, lo:hi]
        Tk_snap   = np.roll(Tk_lc[xi,  yi, :], shift, axis=1)[:, lo:hi]
        vpec_snap = np.roll(vz_lc[xi,  yi, :], shift, axis=1)[:, lo:hi]
        dens_snap = np.roll(dens_lc[xi, yi, :], shift, axis=1)[:, lo:hi]

        xHI_snap  = xHI_snap.astype(np.float32)
        Tk_snap   = Tk_snap.astype(np.float32)
        vpec_snap = vpec_snap.astype(np.float32)
        dens_snap = dens_snap.astype(np.float32)

        # unit conversion 1: xHI → n_HI [cm^-3]
        rho_crit = cosmo.critical_density(z_slice).to(u.g/u.cm**3).value
        Ob       = cosmo.Ob0
        nH_mean  = (rho_crit * Ob * (1 + z_slice)**3) / M_PROTON
        nHI_snap = nH_mean * (1 + dens_snap) * xHI_snap

        # unit conversion 2: vz Mpc/s → km/s
        vpec_snap = vpec_snap * 3.086e19

        # x_sim: relative box coords in Mpc/h
        h            = cosmo.h
        BOX_SIZE_mph = BOX_LEN * h
        x_sim_snap   = np.linspace(0, BOX_SIZE_mph, N_CELLS + 1)[:-1]

        # z_grid: analytic, centred on z_slice at i_center
        H_z  = cosmo.H(z_slice).to(u.km/u.s/u.Mpc).value
        dzdx = H_z / c_light.to(u.km/u.s).value

        z_grid_snap = np.zeros(N_CELLS)
        mid         = N_CELLS // 2
        z_grid_snap[mid] = z_slice
        for i in range(mid - 1, -1, -1):
            z_grid_snap[i] = (z_grid_snap[i+1]
                              - (x_sim_snap[i+1] - x_sim_snap[i]) * dzdx)
        for i in range(mid + 1, N_CELLS):
            z_grid_snap[i] = (z_grid_snap[i-1]
                              + (x_sim_snap[i] - x_sim_snap[i-1]) * dzdx)

        # set runtime redshift BEFORE any pipeline step
        cfg.Z_REDSHIFT = float(z_slice)

        # write scratch arrays for the pipeline steps
        np.save(PATHS["n_HI_halo"],  nHI_snap)
        np.save(PATHS["T_halo"],     Tk_snap)
        np.save(PATHS["v_pec_halo"], vpec_snap)
        np.save(PATHS["halomass"],   masses)
        np.save(PATHS["x_sim"],      x_sim_snap)
        np.save(PATHS["z_grid"],     z_grid_snap)

        # Step 2: spec
        print(f"  [spec]...")
        run_spec(cosmo)
        tau_snap = np.load(PATHS["tau_halo"])

        # Step 3: abundance
        print(f"  [abundance]...")
        run_abundance(cosmo)
        Muv_snap = np.load(PATHS["Muv_grid"])

        # Step 4: assign
        print(f"  [assign]...")
        run_assign()
        LLya_snap = np.load(PATHS["LLya_grid"])
        REW_snap  = np.load(PATHS["REW_grid"])

        # Step 5: damping
        print(f"  [damping]...")
        run_damping()
        damp_snap = np.load(PATHS["damping"])

        # ── checkpoint this snapshot ───────────────────────────────
        np.savez(
            ckpt,
            empty     = False,
            z_node    = z_node,
            z_slice   = z_slice,
            tau       = tau_snap,
            Muv       = Muv_snap,
            LLya      = LLya_snap,
            REW       = REW_snap,
            damping   = damp_snap,
            halomass  = masses,
            coords    = coords,
        )
        print(f"  done  τ range: {tau_snap.min():.2e} – "
              f"{tau_snap.max():.2e}  → checkpoint saved\n")

    # ── assemble full catalogue from checkpoints ───────────────────
    print("All LC slices done — assembling full catalogue from "
          "checkpoints...")

    all_tau, all_Muv, all_LLya, all_REW          = [], [], [], []
    all_damping, all_mass, all_coords, all_z     = [], [], [], []

    for z_idx in range(n_los):
        node_idx = int(slice_owner[z_idx])
        z_node   = float(node_z_sorted[node_idx])
        ckpt = os.path.join(snap_dir,
                            f"snap_{z_idx:04d}_z{z_node:.4f}.npz")
        if not os.path.exists(ckpt):
            print(f"  ! missing checkpoint for LC slice {z_idx} "
                  f"z={z_node:.4f} — skipping")
            continue
        d = np.load(ckpt)
        if bool(d["empty"]):
            continue
        all_tau.append(d["tau"])
        all_Muv.append(d["Muv"])
        all_LLya.append(d["LLya"])
        all_REW.append(d["REW"])
        all_damping.append(d["damping"])
        all_mass.append(d["halomass"])
        all_coords.append(d["coords"])
        all_z.append(np.full(len(d["halomass"]), float(d["z_slice"]),
                             dtype=np.float32))

    if len(all_mass) == 0:
        print("  no non-empty snapshots — nothing to save.")
        return

    np.save(os.path.join(out_dir, "tau.npy"),       np.vstack(all_tau))
    np.save(os.path.join(out_dir, "Muv.npy"),       np.concatenate(all_Muv))
    np.save(os.path.join(out_dir, "LLya.npy"),      np.concatenate(all_LLya))
    np.save(os.path.join(out_dir, "REW.npy"),       np.concatenate(all_REW))
    np.save(os.path.join(out_dir, "damping.npy"),   np.concatenate(all_damping))
    np.save(os.path.join(out_dir, "halomass.npy"),  np.concatenate(all_mass))
    np.save(os.path.join(out_dir, "coords.npy"),    np.vstack(all_coords))
    np.save(os.path.join(out_dir, "redshifts.npy"), np.concatenate(all_z))

    total = sum(len(m) for m in all_mass)
    print(f"\n{'='*60}")
    print(f"✓ Pipeline complete.  Total halos: {total:,}")
    print(f"  Output directory: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("Import and call run_pipeline(lightcone, lightconer, inputs, cosmo)")
