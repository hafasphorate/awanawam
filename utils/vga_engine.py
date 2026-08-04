import math
import numpy as np
import ezdxf
from shapely.geometry import Point, LineString, Polygon, MultiLineString
from shapely.strtree import STRtree


import ezdxf
from ezdxf import recover
from shapely.geometry import LineString

def extract_dxf_walls(dxf_file_path):
    """
    Parses a DXF file from disk and extracts wall line segments.
    Supports LINE, LWPOLYLINE, and POLYLINE entities.
    """
    try:
        # ezdxf.readfile is used for reading file paths on disk
        doc = ezdxf.readfile(dxf_file_path)
    except ezdxf.DXFStructureError:
        # If the DXF has minor errors or non-standard formatting, attempt recovery
        doc, _ = recover.readfile(dxf_file_path)

    msp = doc.modelspace()
    lines = []

    for entity in msp:
        dxftype = entity.dxftype()
        if dxftype == 'LINE':
            start = (entity.dxf.start.x, entity.dxf.start.y)
            end = (entity.dxf.end.x, entity.dxf.end.y)
            if start != end:
                lines.append(LineString([start, end]))
        elif dxftype in ['LWPOLYLINE', 'POLYLINE']:
            # get_points('xy') yields (x, y) tuples
            points = [(p[0], p[1]) for p in entity.get_points('xy')]
            for i in range(len(points) - 1):
                if points[i] != points[i + 1]:
                    lines.append(LineString([points[i], points[i + 1]]))
            if getattr(entity, 'is_closed', False) and len(points) > 1:
                lines.append(LineString([points[-1], points[0]]))

    return lines


def generate_isovist_polygon(origin_pt, wall_lines, strtree, max_dist=50000, num_rays=180):
    """
    Casts rays 360 degrees around origin_pt to generate an Isovist Polygon.
    Uses Shapely STRtree for fast intersection checks.
    """
    ox, oy = origin_pt
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    ray_endpoints = []

    for angle in angles:
        # Define ray tip at max distance
        tx = ox + max_dist * math.cos(angle)
        ty = oy + max_dist * math.sin(angle)
        ray = LineString([(ox, oy), (tx, ty)])

        # Query spatial index for intersecting walls
        candidate_indices = strtree.query(ray)
        closest_dist = max_dist
        closest_hit = (tx, ty)

        for idx in candidate_indices:
            wall = wall_lines[idx]
            intersection = ray.intersection(wall)
            if not intersection.is_empty:
                if intersection.geom_type == 'Point':
                    dist = math.hypot(intersection.x - ox, intersection.y - oy)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_hit = (intersection.x, intersection.y)
                elif intersection.geom_type == 'MultiPoint':
                    for pt in intersection.geoms:
                        dist = math.hypot(pt.x - ox, pt.y - oy)
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_hit = (pt.x, pt.y)

        ray_endpoints.append(closest_hit)

    if len(ray_endpoints) < 3:
        return None

    return Polygon(ray_endpoints)


def compute_isovist_metrics(isovist_poly, origin_pt):
    """
    Calculates key Isovist geometric metrics for a single point.
    """
    if isovist_poly is None or not isovist_poly.is_valid or isovist_poly.is_empty:
        return {}

    ox, oy = origin_pt
    area = isovist_poly.area
    perimeter = isovist_poly.length

    # Compactness (Isoperimetric Quotient: 4 * pi * Area / Perimeter^2)
    compactness = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # Centroid & Drift Magnitude
    centroid = isovist_poly.centroid
    drift_magnitude = math.hypot(centroid.x - ox, centroid.y - oy)

    # Radials (Min & Max distance from origin to perimeter)
    exterior_coords = list(isovist_poly.exterior.coords)
    radials = [math.hypot(cx - ox, cy - oy) for cx, cy in exterior_coords]
    min_radial = min(radials) if radials else 0
    max_radial = max(radials) if radials else 0

    return {
        "isovist_area": round(area, 2),
        "isovist_perimeter": round(perimeter, 2),
        "isovist_compactness": round(compactness, 4),
        "isovist_drift_magnitude": round(drift_magnitude, 2),
        "isovist_min_radial": round(min_radial, 2),
        "isovist_max_radial": round(max_radial, 2),
    }