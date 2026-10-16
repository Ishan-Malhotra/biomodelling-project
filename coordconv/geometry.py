"""Vector maths and the geometric primitives behind the coordinate conversions.

All lengths are in Angstrom and all angles are in degrees.
"""

import math

EPS = 1e-9


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    n = norm(a)
    if n < EPS:
        raise ValueError("two atoms occupy the same position")
    return scale(a, 1.0 / n)


def distance(a, b):
    """Bond length |a - b|."""
    return norm(sub(a, b))


def bond_angle(a, b, c):
    """Bond angle a-b-c, with b at the vertex."""
    v1 = unit(sub(a, b))
    v2 = unit(sub(c, b))
    cos_t = max(-1.0, min(1.0, dot(v1, v2)))
    return math.degrees(math.acos(cos_t))


def dihedral(a, b, c, d):
    """Dihedral (torsion) angle a-b-c-d in the range (-180, 180].

    Uses the IUPAC sign convention: looking down the b->c bond, the angle is
    positive when a must be rotated clockwise to eclipse d.
    """
    b1 = sub(b, a)
    b2 = sub(c, b)
    b3 = sub(d, c)
    if norm(b2) < EPS:
        raise ValueError("two atoms occupy the same position")
    n1 = cross(b1, b2)
    n2 = cross(b2, b3)
    # atan2(0, 0) is 0, which would report a real angle for an undefined dihedral.
    if norm(n1) < EPS or norm(n2) < EPS:
        raise ValueError("the dihedral is undefined because the atoms are collinear")
    x = dot(n1, n2)
    y = dot(cross(n1, n2), unit(b2))
    angle = math.degrees(math.atan2(y, x))
    # Rounding noise can turn an exact anti-periplanar 180 into -179.99999...
    return 180.0 if angle < -180.0 + 1e-6 else angle


def place_atom(a, b, c, bond, angle_deg, torsion_deg):
    """Return the position of atom d given three already-placed atoms a, b, c.

    The new atom satisfies |c - d| = bond, angle(b, c, d) = angle_deg and
    dihedral(a, b, c, d) = torsion_deg. This is the Natural Extension
    Reference Frame (NeRF) method (Parsons et al., J. Comput. Chem. 2005).
    """
    bc = unit(sub(c, b))
    normal = cross(sub(b, a), bc)
    if norm(normal) < EPS:
        raise ValueError("the three reference atoms are collinear, so the dihedral is undefined")
    n = unit(normal)
    m = cross(n, bc)

    theta = math.radians(angle_deg)
    phi = math.radians(torsion_deg)
    d_local = (
        -bond * math.cos(theta),
        bond * math.sin(theta) * math.cos(phi),
        bond * math.sin(theta) * math.sin(phi),
    )
    offset = add(add(scale(bc, d_local[0]), scale(m, d_local[1])), scale(n, d_local[2]))
    return add(c, offset)


def angle_difference(a, b):
    """Smallest signed difference between two angles, in [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0
