import numpy as np
import pickle
from scipy.interpolate import griddata
from config import *

# Load data
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)
U = np.asarray(data["U"], dtype=np.float32)

# Same crop region as run_small_domain.py
y_start, y_end = 75, 155
x_start, x_end = 90, 170

# Full-domain boundary mask
obs_mask = np.zeros((168, 200), dtype=bool)
obs_mask[:B_THICK, :] = True; obs_mask[-B_THICK:, :] = True
obs_mask[:, :B_THICK] = True; obs_mask[:, -B_THICK:] = True

bnd_y, bnd_x = np.where(obs_mask)
bnd_pts = np.column_stack([bnd_y, bnd_x])

# Interior points within the crop region only
crop_interior = np.zeros((168, 200), dtype=bool)
crop_interior[y_start:y_end, x_start:x_end] = True
crop_interior[obs_mask] = False
int_y, int_x = np.where(crop_interior)
int_pts = np.column_stack([int_y, int_x])

test_indices = np.arange(TEST_START, TEST_END)
maes = []
for t in test_indices:
    bnd_vals = U[t, bnd_y, bnd_x]
    u_interp = griddata(bnd_pts, bnd_vals, int_pts, method="cubic")
    nan_mask = np.isnan(u_interp)
    if nan_mask.any():
        u_interp[nan_mask] = griddata(bnd_pts, bnd_vals, int_pts[nan_mask], method="nearest")
    mae = float(np.mean(np.abs(u_interp - U[t, int_y, int_x])))
    maes.append(mae)

print(f"Boundary interpolation MAE in crop region: {np.mean(maes):.4f} °C")
print(f"Full-domain SINN MAE in crop region:       1.3353 °C")
print(f"Cropped-domain SINN MAE in crop region:    1.0988 °C")