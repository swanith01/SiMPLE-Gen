# config.py
import os

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))

# ── RUNTIME-CONFIGURED PARAMS (set by driver via env vars) ────
# All have sensible defaults so config can still be imported standalone.
SEED    = int(os.environ.get("SIMPLEGEN_SEED",     "1"))
BOX_LEN = float(os.environ.get("SIMPLEGEN_BOX_LEN", "400.0"))   # cMpc
HII_DIM = int(os.environ.get("SIMPLEGEN_HII_DIM",   "32"))
MH_CUT  = float(os.environ.get("SIMPLEGEN_MH_CUT",  "8.5"))     # log10(M/Msun)

# ── LIGHTCONE INPUT PATHS ─────────────────────────────────────
HALO_CATALOGUE_DIR = os.environ.get(
    "SIMPLEGEN_HALO_DIR",
    f"/user1/swanith/lightcone_halos/catalogues/seed_{SEED}",
)

# ── OUTPUT FILES (per seed) ───────────────────────────────────
DATA_DIR = os.path.join(PACKAGE_DIR, "data", f"seed_{SEED}")
os.makedirs(DATA_DIR, exist_ok=True)

PATHS = {
    "n_HI_halo":   os.path.join(DATA_DIR, "n_HI_halo.npy"),
    "T_halo":      os.path.join(DATA_DIR, "T_halo.npy"),
    "v_pec_halo":  os.path.join(DATA_DIR, "v_pec_halo.npy"),
    "halomass":    os.path.join(DATA_DIR, "halomass.npy"),
    "x_sim":       os.path.join(DATA_DIR, "x_sim.npy"),
    "z_grid":      os.path.join(DATA_DIR, "z_grid.npy"),
    "tau_halo":    os.path.join(DATA_DIR, "tau_halo.npy"),
    "Muv_grid":    os.path.join(DATA_DIR, "Muv_grid.npy"),
    "LLya_grid":   os.path.join(DATA_DIR, "LLya_grid.npy"),
    "REW_grid":    os.path.join(DATA_DIR, "REW_grid.npy"),
    "damping":     os.path.join(DATA_DIR, "damping.npy"),
    "halo_coords": os.path.join(DATA_DIR, "halo_coords.npy"),
    "halo_z":      os.path.join(DATA_DIR, "halo_z.npy"),
}

# ── PER-SNAPSHOT REDSHIFT ─────────────────────────────────────
# MUST be set by the driver via:
#   import SiMPLEGen.config as cfg
#   cfg.Z_REDSHIFT = float(z_node)
# before each call to run_abundance() / run_assign().
# Left as None so a forgotten override crashes loudly instead of
# silently using a stale value.
Z_REDSHIFT = None
