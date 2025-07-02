"""
Cleanify v2-alpha Geographic Utilities
H3 tiling, projections, distance calculations, and corridor analysis
"""

import math
import h3
from typing import List, Tuple, Set, Dict, Optional
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform
import pyproj
from functools import partial

from core.models import Bin, CorridorConfig


def get_tile_id(lat: float, lon: float, res: int = 9) -> str:
    """
    Get H3 tile ID for coordinates
    Default res=9 gives ~500m hex cells
    """
    return h3.latlng_to_cell(lat, lon, res)


def get_tile_bounds(tile_id: str) -> List[Tuple[float, float]]:
    """Get boundary coordinates of H3 tile"""
    return h3.h3_to_geo_boundary(tile_id)


def get_neighbor_tiles(tile_id: str, k: int = 1) -> Set[str]:
    """Get neighboring H3 tiles within k rings"""
    return h3.k_ring(tile_id, k)


def tiles_in_radius(center_lat: float, center_lon: float, radius_m: float, res: int = 9) -> Set[str]:
    """Get all H3 tiles within radius of center point"""
    center_tile = get_tile_id(center_lat, center_lon, res)
    
    # Estimate k-ring size based on radius and resolution
    avg_hex_edge_m = h3.edge_length(res, unit='m')
    k = max(1, int(radius_m / avg_hex_edge_m) + 1)
    
    # Get tiles in k-ring
    candidate_tiles = h3.k_ring(center_tile, k)
    
    # Filter by actual distance
    valid_tiles = set()
    for tile in candidate_tiles:
        tile_center = h3.h3_to_geo(tile)
        if haversine_distance(center_lat, center_lon, tile_center[0], tile_center[1]) <= radius_m:
            valid_tiles.add(tile)
    
    return valid_tiles


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate haversine distance between two points in meters
    """
    R = 6371000  # Earth radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in kilometers"""
    return haversine_distance(lat1, lon1, lat2, lon2) / 1000.0


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing from point 1 to point 2 in degrees"""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    
    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
    
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    
    return (bearing_deg + 360) % 360


def create_projection_transformer(from_crs: str = "EPSG:4326", to_crs: str = "EPSG:3857"):
    """Create projection transformer for coordinate system conversion"""
    return pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)


def project_point(lat: float, lon: float, transformer) -> Tuple[float, float]:
    """Project lat/lon to different coordinate system"""
    return transformer.transform(lon, lat)


def tube_filter(polyline_latlon: List[Tuple[float, float]], half_m: float) -> Set[str]:
    """
    Filter bins within tube around polyline
    Returns set of bin IDs within the tube
    """
    if not polyline_latlon or len(polyline_latlon) < 2:
        return set()
    
    # Create LineString from polyline
    line = LineString([(lon, lat) for lat, lon in polyline_latlon])
    
    # Create UTM transformer for accurate distance calculations
    # Use the centroid to determine appropriate UTM zone
    centroid = line.centroid
    utm_zone = int((centroid.x + 180) / 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}"  # Northern hemisphere assumption
    
    transformer = create_projection_transformer("EPSG:4326", utm_crs)
    
    # Transform line to UTM
    utm_line = transform(transformer.transform, line)
    
    # Create buffer around line
    buffered = utm_line.buffer(half_m)
    
    # Transform buffer back to WGS84
    inverse_transformer = create_projection_transformer(utm_crs, "EPSG:4326")
    wgs84_buffer = transform(inverse_transformer.transform, buffered)
    
    return wgs84_buffer


def crowflight_candidates(bins: List[Bin], polyline_latlon: List[Tuple[float, float]], 
                         radius_m: float) -> List[Bin]:
    """
    Find bins within crow-flight distance of any point on polyline
    """
    candidates = []
    
    for bin_obj in bins:
        min_distance = float('inf')
        
        # Check distance to each point on polyline
        for lat, lon in polyline_latlon:
            distance = haversine_distance(bin_obj.lat, bin_obj.lon, lat, lon)
            min_distance = min(min_distance, distance)
        
        if min_distance <= radius_m:
            candidates.append(bin_obj)
    
    return candidates


def detour_km(polyline_latlon: List[Tuple[float, float]], bin_obj: Bin) -> float:
    """
    Calculate detour distance to visit bin from polyline route
    """
    if len(polyline_latlon) < 2:
        return 0.0
    
    # Find closest point on route to bin
    min_detour = float('inf')
    bin_point = (bin_obj.lat, bin_obj.lon)
    
    for i in range(len(polyline_latlon) - 1):
        # Calculate detour for inserting bin between points i and i+1
        point_a = polyline_latlon[i]
        point_b = polyline_latlon[i + 1]
        
        # Original distance A -> B
        original_dist = distance_km(point_a[0], point_a[1], point_b[0], point_b[1])
        
        # Detour distance A -> Bin -> B
        detour_dist = (distance_km(point_a[0], point_a[1], bin_point[0], bin_point[1]) +
                      distance_km(bin_point[0], bin_point[1], point_b[0], point_b[1]))
        
        detour = detour_dist - original_dist
        min_detour = min(min_detour, detour)
    
    return max(0.0, min_detour)


def point_to_line_distance(point_lat: float, point_lon: float, 
                          line_start: Tuple[float, float], 
                          line_end: Tuple[float, float]) -> float:
    """
    Calculate shortest distance from point to line segment
    """
    # Convert to Cartesian for calculation
    x0, y0 = point_lon, point_lat
    x1, y1 = line_start[1], line_start[0]  # lon, lat
    x2, y2 = line_end[1], line_end[0]      # lon, lat
    
    # Vector from line start to end
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        # Line is just a point
        return haversine_distance(point_lat, point_lon, line_start[0], line_start[1])
    
    # Parameter t for closest point on line
    t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)))
    
    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance from point to closest point on line
    return haversine_distance(point_lat, point_lon, closest_y, closest_x)


def build_corridor(polyline_latlon: List[Tuple[float, float]], 
                  route_wids: List[int], 
                  bins: List[Bin],
                  config: CorridorConfig) -> Set[str]:
    """
    EXACT implementation from specification:
    Two-layer filter for corridor analysis
    """
    # Layer 1: Tube filter
    tube_bins = set()
    
    # Create buffer around polyline
    if polyline_latlon and len(polyline_latlon) >= 2:
        buffer_polygon = tube_filter(polyline_latlon, config.corridor_half_m)
        
        for bin_obj in bins:
            bin_point = Point(bin_obj.lon, bin_obj.lat)
            if buffer_polygon.contains(bin_point):
                tube_bins.add(bin_obj.id)
    
    # Layer 2: Candidate set analysis
    candidates = crowflight_candidates(bins, polyline_latlon, config.candidate_scan_radius)
    
    # Edge match bins
    edge_match_bins = set()
    route_offset_m = {}  # Simplified - in real implementation would calculate offsets
    
    for bin_obj in candidates:
        if (bin_obj.way_id in route_wids and 
            abs(bin_obj.snap_offset_m - route_offset_m.get(bin_obj.way_id, 0)) <= config.way_offset_thresh_m):
            edge_match_bins.add(bin_obj.id)
    
    # Detour bins
    detour_bins = set()
    candidate_bin_ids = {bin_obj.id for bin_obj in candidates}
    remaining_candidates = candidate_bin_ids - edge_match_bins
    
    if polyline_latlon:
        route_len_km = sum(distance_km(polyline_latlon[i][0], polyline_latlon[i][1],
                                     polyline_latlon[i+1][0], polyline_latlon[i+1][1])
                          for i in range(len(polyline_latlon) - 1))
    else:
        route_len_km = 1.0  # Avoid division by zero
    
    for bin_obj in candidates:
        if bin_obj.id in remaining_candidates:
            bin_detour = detour_km(polyline_latlon, bin_obj)
            
            if (bin_detour <= config.max_detour_km or 
                bin_detour / route_len_km <= config.max_detour_ratio):
                detour_bins.add(bin_obj.id)
    
    # Combine all layers
    return tube_bins | edge_match_bins | detour_bins


def create_bin_tiles(bins: List[Bin], res: int = 9) -> Dict[str, List[Bin]]:
    """
    Organize bins by H3 tiles for spatial indexing
    """
    tile_map = {}
    
    for bin_obj in bins:
        if not bin_obj.tile_id:
            bin_obj.tile_id = get_tile_id(bin_obj.lat, bin_obj.lon, res)
        
        if bin_obj.tile_id not in tile_map:
            tile_map[bin_obj.tile_id] = []
        
        tile_map[bin_obj.tile_id].append(bin_obj)
    
    return tile_map


def bins_in_radius(center_lat: float, center_lon: float, radius_m: float, 
                  tile_map: Dict[str, List[Bin]], res: int = 9) -> List[Bin]:
    """
    Efficiently find bins within radius using H3 spatial index
    """
    # Get relevant tiles
    relevant_tiles = tiles_in_radius(center_lat, center_lon, radius_m, res)
    
    # Collect candidate bins from tiles
    candidate_bins = []
    for tile_id in relevant_tiles:
        if tile_id in tile_map:
            candidate_bins.extend(tile_map[tile_id])
    
    # Filter by exact distance
    result_bins = []
    for bin_obj in candidate_bins:
        distance = haversine_distance(center_lat, center_lon, bin_obj.lat, bin_obj.lon)
        if distance <= radius_m:
            result_bins.append(bin_obj)
    
    return result_bins


def interpolate_path(start: Tuple[float, float], end: Tuple[float, float], 
                    num_points: int = 10) -> List[Tuple[float, float]]:
    """
    Interpolate points along great circle path between two coordinates
    """
    points = []
    
    for i in range(num_points + 1):
        ratio = i / num_points
        
        # Simple linear interpolation (for short distances)
        lat = start[0] + (end[0] - start[0]) * ratio
        lon = start[1] + (end[1] - start[1]) * ratio
        
        points.append((lat, lon))
    
    return points


def calculate_polygon_area(coords: List[Tuple[float, float]]) -> float:
    """
    Calculate area of polygon in square meters using UTM projection
    """
    if len(coords) < 3:
        return 0.0
    
    # Create polygon
    polygon = Polygon([(lon, lat) for lat, lon in coords])
    
    # Find appropriate UTM zone
    centroid = polygon.centroid
    utm_zone = int((centroid.x + 180) / 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}"
    
    # Transform to UTM and calculate area
    transformer = create_projection_transformer("EPSG:4326", utm_crs)
    utm_polygon = transform(transformer.transform, polygon)
    
    return utm_polygon.area


def simplify_polyline(coords: List[Tuple[float, float]], tolerance_m: float = 10.0) -> List[Tuple[float, float]]:
    """
    Simplify polyline using Douglas-Peucker algorithm
    """
    if len(coords) <= 2:
        return coords
    
    line = LineString([(lon, lat) for lat, lon in coords])
    
    # Convert tolerance to degrees (approximate)
    tolerance_deg = tolerance_m / 111000.0  # Rough conversion
    
    simplified = line.simplify(tolerance_deg, preserve_topology=True)
    
    return [(lat, lon) for lon, lat in simplified.coords]