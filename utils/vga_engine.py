import math
import numpy as np
import ezdxf
from ezdxf import recover
from shapely.geometry import Point, LineString, Polygon
from shapely.strtree import STRtree

import os
import subprocess

def convert_dwg_to_dxf(dwg_path: str) -> str:
    """Converts an uploaded .dwg file to a temporary .dxf file using dwg2dxf."""
    output_dxf_path = dwg_path.rsplit(".", 1)[0] + "_converted.dxf"
    
    try:
        # Call the system CLI tool dwg2dxf
        result = subprocess.run(
            ["dwg2dxf", "-o", output_dxf_path, dwg_path],
            check=True,
            capture_output=True,
            text=True
        )
        if os.path.exists(output_dxf_path):
            return output_dxf_path
        else:
            raise RuntimeError("DWG conversion failed: Output DXF file was not generated.")
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            "DWG conversion tool 'dwg2dxf' (libreDWG) is not found or failed on this system. "
            "Please verify 'libredwg-tools' is installed, or upload a native DXF file."
        ) from e


def process_cad_file(file_path: str):
    """Router function: Accepts .dxf or .dwg, converts if needed, and extracts wall lines."""
    if file_path.lower().endswith(".dwg"):
        dxf_path = convert_dwg_to_dxf(file_path)
    else:
        dxf_path = file_path

    # Uses your existing ezdxf extraction function
    return extract_dxf_walls(dxf_path)

def extract_dxf_walls(dxf_file_path):
    """Parses a DXF file from disk and extracts wall line segments."""
    try:
        doc = ezdxf.readfile(dxf_file_path)
    except ezdxf.DXFStructureError:
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
            points = [(p[0], p[1]) for p in entity.get_points('xy')]
            for i in range(len(points) - 1):
                if points[i] != points[i + 1]:
                    lines.append(LineString([points[i], points[i + 1]]))
            if getattr(entity, 'is_closed', False) and len(points) > 1:
                lines.append(LineString([points[-1], points[0]]))

    return lines


def generate_isovist_polygon(origin_pt, wall_lines, strtree, max_dist=50000, num_rays=180):
    """Casts rays 360 degrees around origin_pt to generate an Isovist Polygon and occlusion rays."""
    ox, oy = origin_pt
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    ray_endpoints = []
    occluded_count = 0

    for angle in angles:
        tx = ox + max_dist * math.cos(angle)
        ty = oy + max_dist * math.sin(angle)
        ray = LineString([(ox, oy), (tx, ty)])

        candidate_indices = strtree.query(ray)
        closest_dist = max_dist
        closest_hit = (tx, ty)
        hit_wall = False

        for idx in candidate_indices:
            wall = wall_lines[idx]
            intersection = ray.intersection(wall)
            if not intersection.is_empty:
                if intersection.geom_type == 'Point':
                    dist = math.hypot(intersection.x - ox, intersection.y - oy)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_hit = (intersection.x, intersection.y)
                        hit_wall = True
                elif intersection.geom_type == 'MultiPoint':
                    for pt in intersection.geoms:
                        dist = math.hypot(pt.x - ox, pt.y - oy)
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_hit = (pt.x, pt.y)
                            hit_wall = True

        if not hit_wall:
            occluded_count += 1

        ray_endpoints.append(closest_hit)

    if len(ray_endpoints) < 3:
        return None, 0

    return Polygon(ray_endpoints), occluded_count


def compute_isovist_metrics(isovist_poly, origin_pt, occluded_count, num_rays):
    """Calculates geometric Isovist properties for a single point."""
    if isovist_poly is None or not isovist_poly.is_valid or isovist_poly.is_empty:
        return {}

    ox, oy = origin_pt
    area = isovist_poly.area
    perimeter = isovist_poly.length

    # Compactness (Isoperimetric Quotient)
    compactness = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # Centroid & Drift
    centroid = isovist_poly.centroid
    drift_magnitude = math.hypot(centroid.x - ox, centroid.y - oy)

    # Exterior Radials
    exterior_coords = list(isovist_poly.exterior.coords)
    radials = [math.hypot(cx - ox, cy - oy) for cx, cy in exterior_coords]
    min_radial = min(radials) if radials else 0
    max_radial = max(radials) if radials else 0

    # Occlusivity (Ratio of unblocked/extended ray perimeter length)
    occlusivity = occluded_count / float(num_rays)

    return {
        "isovist_area": round(area, 2),
        "isovist_perimeter": round(perimeter, 2),
        "isovist_compactness": round(compactness, 4),
        "isovist_drift_magnitude": round(drift_magnitude, 2),
        "isovist_min_radial": round(min_radial, 2),
        "isovist_max_radial": round(max_radial, 2),
        "isovist_drift_minima": round(abs(drift_magnitude - min_radial), 2),
        "isovist_occlusivity": round(occlusivity, 4)
    }


def compute_graph_vga_metrics(vga_list, isovist_polys):
    """
    Computes Space Syntax graph properties:
    Connectivity, Visual Integration, Visual Mean Depth, and Visual Entropy.
    """
    N = len(vga_list)
    if N < 2:
        return vga_list

    # Build Adjacency Matrix (Points are connected if point B lies inside point A's Isovist)
    adjacency = np.zeros((N, N), dtype=int)
    for i in range(N):
        poly_i = isovist_polys[i]
        for j in range(N):
            if i != j:
                pt_j = Point(vga_list[j]["x"], vga_list[j]["y"])
                if poly_i and poly_i.contains(pt_j):
                    adjacency[i, j] = 1

    # Shortest path matrix (Breadth-First Search / Floyd-Warshall approximation)
    dist_matrix = np.full((N, N), fill_value=np.inf)
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix[adjacency == 1] = 1

    for k in range(N):
        dist_matrix = np.minimum(dist_matrix, dist_matrix[:, [k]] + dist_matrix[[k], :])

    # Calculate Graph Metrics
    for i in range(N):
        connectivity = int(np.sum(adjacency[i]))
        valid_depths = dist_matrix[i][np.isfinite(dist_matrix[i])]
        
        # Visual Mean Depth
        mean_depth = float(np.mean(valid_depths)) if len(valid_depths) > 0 else N
        
        # Visual Integration (Relative Asymmetry inversion)
        RA = (2 * (mean_depth - 1)) / (N - 1) if N > 1 else 1.0
        integration = 1.0 / RA if RA > 0 else 0.0

        # Visual Entropy
        depth_counts = np.bincount(valid_depths.astype(int))
        probs = depth_counts[depth_counts > 0] / float(len(valid_depths))
        entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0

        vga_list[i]["connectivity"] = connectivity
        vga_list[i]["visual_mean_depth"] = round(mean_depth, 2)
        vga_list[i]["visual_integration"] = round(integration, 3)
        vga_list[i]["visual_entropy"] = round(entropy, 3)

    return vga_list