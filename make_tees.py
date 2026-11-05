"""Free-standing simulator golf tee (sits on the mat, doesn't stick in).

Print orientation: wide base on the bed, cup up — exactly how it sits on the mat.
Hollow cone body with a staggered diamond lattice. Every window's top edges are
56 deg from horizontal, the inner cone leans <20 deg, and the shoulder ceiling is
45 deg, so nothing needs supports or bridging.
H = height from floor to the top of the cup rim.
"""
import math
import numpy as np
import trimesh
from manifold3d import Manifold, CrossSection

# ---- parameters (mm) ----
BASE_R    = 25.0   # 50 mm footprint
CHAMFER   = 0.6    # elephant's-foot chamfer at bed
WALL      = 2.5    # cone wall thickness (horizontal)
SHOULDER_R = 10.0  # cone radius at the shoulder
CUP_OR    = 8.0   # cup tube outer radius -> 16 mm top diameter
CUP_IR    = 6.0   # cup tube inner radius (2 mm wall)
CUP_EXPOSED = 7.0  # tube height above the shoulder
CUP_CHAMF = 0.8    # 45 deg seat chamfer on the inner rim
BAND_BOT  = 5.0    # solid band at the floor
BAND_TOP  = 5.0    # solid band under the shoulder
COLS      = 8      # diamonds around
RIB       = 2.6    # rib thickness between diamonds (perpendicular)
DIAMOND_ANGLE = 56 # window edge angle from horizontal (>45 = self-supporting)
SEG       = 160


def body(H):
    hc = H - CUP_EXPOSED
    lean = (BASE_R - SHOULDER_R) / hc
    ri = lambda z: BASE_R - WALL - lean * z
    z_it = hc - 2.0                          # top of inner cone cavity
    z_trans = z_it + (ri(z_it) - CUP_IR)     # 45 deg ceiling up to tube bore
    prof = [
        (BASE_R - WALL, 0), (BASE_R - CHAMFER, 0), (BASE_R, CHAMFER),
        (SHOULDER_R, hc), (CUP_OR, hc), (CUP_OR, H),
        (CUP_IR + CUP_CHAMF, H), (CUP_IR, H - CUP_CHAMF),
        (CUP_IR, z_trans), (ri(z_it), z_it),
    ]
    return Manifold.revolve(CrossSection([prof]), SEG), hc, lean


def diamond_cutters(H, hc, lean):
    ro = lambda z: BASE_R - lean * z
    t = math.tan(math.radians(DIAMOND_ANGLE))
    cuts, z, row, b = [], BAND_BOT, 0, 6.0
    while True:
        # size the diamond from the radius at its top so ribs never get thin
        for _ in range(5):
            r_top = ro(z + 2 * b) - WALL / 2
            bay = 2 * math.pi * r_top / COLS
            a = (bay - RIB / math.sin(math.radians(DIAMOND_ANGLE))) / 2
            b = a * t
        if z + 2 * b > hc - BAND_TOP or a < 1.5:
            break
        poly = CrossSection([[(a, 0), (0, b), (-a, 0), (0, -b)]])
        prism = Manifold.extrude(poly, 12.0)
        zc = z + b
        r = ro(zc)
        # map: poly x -> tangential(y), poly y -> z, extrude -> radial(x)
        m = prism.transform([[0, 0, 1, r - 6], [1, 0, 0, 0], [0, 1, 0, zc]])
        off = (360 / COLS / 2) * (row % 2)
        for k in range(COLS):
            cuts.append(m.rotate([0, 0, off + k * 360 / COLS]))
        # staggered pitch that keeps the diagonal rib = RIB
        s = math.hypot(a, b)
        gh = bay - 2 * a
        v = (b * (a - gh / 2) + RIB * s) / a   # vertical pitch to next row center
        z = zc + v - b                           # next row bottom
        row += 1
    return cuts


def build(H):
    shell, hc, lean = body(H)
    for c in diamond_cutters(H, hc, lean):
        shell = shell - c
    mesh = shell.to_mesh()
    m = trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts)
    m.fix_normals()
    return m


def overhang_area(m, max_deg=45.0):
    n, z, a = m.face_normals, m.triangles_center[:, 2], m.area_faces
    bad = (n[:, 2] < -math.cos(math.radians(max_deg)) - 1e-3) & (z > 0.05)
    return a[bad].sum()


if __name__ == "__main__":
    for H in (45, 50, 55, 60):
        m = build(H)
        print(f"{H}mm: watertight={m.is_watertight} bodies={len(m.split())} "
              f"height={m.extents[2]:.2f} vol={m.volume/1000:.2f}cm3 "
              f"overhang>45={overhang_area(m):.2f}mm2")
        m.export(f"sim_tee_{H}mm.stl")