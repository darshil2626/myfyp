import numpy as np

def load_amira_lattice_float2(path, shape=(512, 512, 1001), dtype=np.dtype('<f4')):
    """
    Loads an AmiraMesh Lattice { float[2] Data } @1
    Returns: data shaped (nz, ny, nx, 2) by default (time as z).
    """
    nx, ny, nz = shape  # from "define Lattice 512 512 1001"

    with open(path, "rb") as f:
        raw = f.read()

    # Find the '@1' marker, then skip to the start of binary data after the newline
    marker = raw.find(b"@1")
    if marker == -1:
        raise ValueError("Could not find '@1' data marker in file.")

    # Data starts after '@1' and the following newline(s)
    data_start = marker + 2
    while data_start < len(raw) and raw[data_start] in (ord('\n'), ord('\r'), ord(' '), ord('\t')):
        data_start += 1

    # Interpret the remaining bytes as little-endian float32
    arr = np.frombuffer(raw, dtype=dtype, offset=data_start)

    expected = nx * ny * nz * 2
    if arr.size < expected:
        raise ValueError(
            f"File truncated. Got {arr.size}, expected at least {expected}."
        )

    arr = arr[:expected]  # Ignore trailing padding


    # Amira Lattice typically stores x fastest, then y, then z
    # So reshape as (nz, ny, nx, 2)
    data = arr.reshape((nz, ny, nx, 2))
    return data

# Example usage:
data = load_amira_lattice_float2("0000.am", shape=(512, 512, 1001))
print(data.shape)  # (1001, 512, 512, 2)

import matplotlib.pyplot as plt

t = 0
u = data[t, :, :, 0]
v = data[t, :, :, 1]
speed = np.sqrt(u*u + v*v)

plt.figure()
plt.imshow(speed, origin="lower")
plt.colorbar(label="|v|")
plt.title(f"Speed magnitude at t-index {t}")
plt.show()
