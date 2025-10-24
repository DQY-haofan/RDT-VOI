from config import load_config
from geometry import build_grid2d_geometry
from spatial_field import build_prior

cfg = load_config()
geom = build_grid2d_geometry(40, 40, h=5.0)
Q_pr, mu_pr = build_prior(geom, cfg.prior)