"""
Cleanify v2-alpha Corridor Agent
Implements two-layer corridor analysis for route-based bin discovery
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
import json

from .base import AgentBase
from core.models import Bin, CorridorConfig
from core.geo import build_corridor, distance_km
from core.settings import get_settings


class CorridorAgent(AgentBase):
    """
    Corridor agent that analyzes routes and identifies bins within corridors
    Implements exact two-layer filter as specified
    """
    
    def __init__(self):
        super().__init__("corridor", "corridor")
        
        # Corridor configuration from settings
        self.corridor_config = CorridorConfig()
        self._load_config_from_settings()
        
        # Corridor analysis cache
        self.corridor_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry_seconds = 300  # 5 minutes
        
        # Performance metrics
        self.corridors_analyzed = 0
        self.bins_found_in_corridors = 0
        self.cache_hits = 0
        
        # Register handlers
        self._register_corridor_handlers()
    
    async def initialize(self):
        """Initialize corridor agent"""
        self.logger.info("Initializing Corridor Agent")
        
        self.logger.info("Corridor agent initialized",
                        corridor_half_m=self.corridor_config.corridor_half_m,
                        max_detour_km=self.corridor_config.max_detour_km)
    
    async def main_loop(self):
        """Main corridor analysis loop"""
        while self.running:
            try:
                # Clean expired cache entries
                await self._clean_cache()
                
                # Sleep briefly
                await asyncio.sleep(30.0)
                
            except Exception as e:
                self.logger.error("Error in corridor main loop", error=str(e))
                await asyncio.sleep(60)
    
    async def cleanup(self):
        """Cleanup corridor agent"""
        self.logger.info("Corridor agent cleanup")
    
    async def build_corridor(self, polyline_latlon: List[Tuple[float, float]], 
                           route_wids: List[int], 
                           bins: List[Bin]) -> Set[str]:
        """
        EXACT implementation from specification:
        Build corridor using two-layer filter
        """
        
        if not polyline_latlon or len(polyline_latlon) < 2:
            self.logger.warning("Invalid polyline for corridor analysis")
            return set()
        
        # Check cache first
        cache_key = self._generate_cache_key(polyline_latlon, route_wids, [b.id for b in bins])
        if cache_key in self.corridor_cache:
            cache_entry = self.corridor_cache[cache_key]
            if (datetime.now() - cache_entry['timestamp']).total_seconds() < self.cache_expiry_seconds:
                self.cache_hits += 1
                return cache_entry['corridor_bins']
        
        try:
            # Use the exact implementation from core.geo
            corridor_bins = build_corridor(
                polyline_latlon, 
                route_wids, 
                bins, 
                self.corridor_config
            )
            
            # Cache result
            self.corridor_cache[cache_key] = {
                'corridor_bins': corridor_bins,
                'timestamp': datetime.now(),
                'polyline_length': len(polyline_latlon),
                'bins_analyzed': len(bins)
            }
            
            # Update metrics
            self.corridors_analyzed += 1
            self.bins_found_in_corridors += len(corridor_bins)
            
            self.logger.debug("Corridor analysis completed",
                            polyline_points=len(polyline_latlon),
                            route_ways=len(route_wids),
                            total_bins=len(bins),
                            corridor_bins=len(corridor_bins))
            
            return corridor_bins
            
        except Exception as e:
            self.logger.error("Error in corridor analysis", error=str(e))
            return set()
    
    def _load_config_from_settings(self):
        """Load corridor configuration from settings"""
        settings = get_settings()
        
        self.corridor_config.corridor_half_m = settings.corridor.CORRIDOR_HALF_M
        self.corridor_config.way_offset_thresh_m = settings.corridor.WAY_OFFSET_THRESH_M
        self.corridor_config.max_detour_km = settings.corridor.MAX_DETOUR_KM
        self.corridor_config.max_detour_ratio = settings.corridor.MAX_DETOUR_RATIO
        self.corridor_config.candidate_scan_radius = settings.corridor.CANDIDATE_SCAN_RADIUS
    
    def _generate_cache_key(self, polyline: List[Tuple[float, float]], 
                          route_wids: List[int], bin_ids: List[str]) -> str:
        """Generate cache key for corridor analysis"""
        
        # Create hash from polyline points (simplified)
        polyline_hash = hash(tuple(tuple(point) for point in polyline[:10]))  # First 10 points
        route_hash = hash(tuple(sorted(route_wids)))
        bins_hash = hash(tuple(sorted(bin_ids)))
        
        return f"corridor_{polyline_hash}_{route_hash}_{bins_hash}"
    
    async def _clean_cache(self):
        """Remove expired cache entries"""
        current_time = datetime.now()
        expired_keys = []
        
        for cache_key, cache_entry in self.corridor_cache.items():
            age_seconds = (current_time - cache_entry['timestamp']).total_seconds()
            if age_seconds > self.cache_expiry_seconds:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.corridor_cache[key]
        
        if expired_keys:
            self.logger.debug("Cleaned corridor cache", expired_entries=len(expired_keys))
    
    def _register_corridor_handlers(self):
        """Register corridor-specific message handlers"""
        self.register_handler("analyze_corridor", self._handle_analyze_corridor)
        self.register_handler("get_corridor_config", self._handle_get_corridor_config)
        self.register_handler("update_corridor_config", self._handle_update_corridor_config)
        self.register_handler("clear_corridor_cache", self._handle_clear_corridor_cache)
    
    async def _handle_analyze_corridor(self, data: Dict[str, Any]):
        """Handle corridor analysis request"""
        try:
            # Parse request data
            polyline_data = data.get("polyline", [])
            route_wids = data.get("route_wids", [])
            bins_data = data.get("bins", [])
            
            # Convert polyline data to coordinate tuples
            polyline_latlon = []
            for point in polyline_data:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    polyline_latlon.append((float(point[0]), float(point[1])))
            
            # Parse bins
            bins = []
            for bin_data in bins_data:
                bin_obj = Bin(
                    id=bin_data["id"],
                    lat=bin_data["lat"],
                    lon=bin_data["lon"],
                    capacity_l=bin_data.get("capacity_l", 1000),
                    fill_level=bin_data.get("fill_level", 50.0),
                    fill_rate_lph=bin_data.get("fill_rate_lph", 5.0),
                    tile_id=bin_data.get("tile_id", ""),
                    way_id=bin_data.get("way_id"),
                    snap_offset_m=bin_data.get("snap_offset_m", 0.0)
                )
                bins.append(bin_obj)
            
            # Perform corridor analysis
            corridor_bins = await self.build_corridor(polyline_latlon, route_wids, bins)
            
            # Send response
            await self.send_message(
                "corridor_analysis_result",
                {
                    "corridor_bins": list(corridor_bins),
                    "total_bins_analyzed": len(bins),
                    "corridor_bins_found": len(corridor_bins),
                    "polyline_points": len(polyline_latlon),
                    "config_used": {
                        "corridor_half_m": self.corridor_config.corridor_half_m,
                        "max_detour_km": self.corridor_config.max_detour_km,
                        "max_detour_ratio": self.corridor_config.max_detour_ratio,
                        "way_offset_thresh_m": self.corridor_config.way_offset_thresh_m,
                        "candidate_scan_radius": self.corridor_config.candidate_scan_radius
                    },
                    "correlation_id": data.get("correlation_id")
                }
            )
            
        except Exception as e:
            self.logger.error("Error handling corridor analysis request", error=str(e))
            
            await self.send_message(
                "corridor_analysis_error",
                {
                    "error": str(e),
                    "correlation_id": data.get("correlation_id")
                }
            )
    
    async def _handle_get_corridor_config(self, data: Dict[str, Any]):
        """Handle corridor configuration request"""
        config_data = {
            "corridor_half_m": self.corridor_config.corridor_half_m,
            "way_offset_thresh_m": self.corridor_config.way_offset_thresh_m,
            "max_detour_km": self.corridor_config.max_detour_km,
            "max_detour_ratio": self.corridor_config.max_detour_ratio,
            "candidate_scan_radius": self.corridor_config.candidate_scan_radius,
            "cache_expiry_seconds": self.cache_expiry_seconds,
            "correlation_id": data.get("correlation_id")
        }
        
        await self.send_message("corridor_config", config_data)
    
    async def _handle_update_corridor_config(self, data: Dict[str, Any]):
        """Handle corridor configuration update"""
        try:
            config_updates = data.get("config", {})
            
            # Update configuration with validation
            if "corridor_half_m" in config_updates:
                value = float(config_updates["corridor_half_m"])
                if 50.0 <= value <= 1000.0:
                    self.corridor_config.corridor_half_m = value
                else:
                    raise ValueError("corridor_half_m must be between 50.0 and 1000.0")
            
            if "way_offset_thresh_m" in config_updates:
                value = float(config_updates["way_offset_thresh_m"])
                if 10.0 <= value <= 1000.0:
                    self.corridor_config.way_offset_thresh_m = value
                else:
                    raise ValueError("way_offset_thresh_m must be between 10.0 and 1000.0")
            
            if "max_detour_km" in config_updates:
                value = float(config_updates["max_detour_km"])
                if 0.1 <= value <= 5.0:
                    self.corridor_config.max_detour_km = value
                else:
                    raise ValueError("max_detour_km must be between 0.1 and 5.0")
            
            if "max_detour_ratio" in config_updates:
                value = float(config_updates["max_detour_ratio"])
                if 0.01 <= value <= 0.5:
                    self.corridor_config.max_detour_ratio = value
                else:
                    raise ValueError("max_detour_ratio must be between 0.01 and 0.5")
            
            if "candidate_scan_radius" in config_updates:
                value = float(config_updates["candidate_scan_radius"])
                if 100.0 <= value <= 5000.0:
                    self.corridor_config.candidate_scan_radius = value
                else:
                    raise ValueError("candidate_scan_radius must be between 100.0 and 5000.0")
            
            # Clear cache after config update
            self.corridor_cache.clear()
            
            await self.send_message(
                "corridor_config_updated",
                {
                    "status": "success",
                    "updated_config": {
                        "corridor_half_m": self.corridor_config.corridor_half_m,
                        "way_offset_thresh_m": self.corridor_config.way_offset_thresh_m,
                        "max_detour_km": self.corridor_config.max_detour_km,
                        "max_detour_ratio": self.corridor_config.max_detour_ratio,
                        "candidate_scan_radius": self.corridor_config.candidate_scan_radius
                    },
                    "cache_cleared": True,
                    "correlation_id": data.get("correlation_id")
                }
            )
            
            self.logger.info("Corridor configuration updated",
                           updates=list(config_updates.keys()))
            
        except Exception as e:
            self.logger.error("Error updating corridor configuration", error=str(e))
            
            await self.send_message(
                "corridor_config_update_error",
                {
                    "error": str(e),
                    "correlation_id": data.get("correlation_id")
                }
            )
    
    async def _handle_clear_corridor_cache(self, data: Dict[str, Any]):
        """Handle cache clearing request"""
        cache_size_before = len(self.corridor_cache)
        self.corridor_cache.clear()
        
        await self.send_message(
            "corridor_cache_cleared",
            {
                "entries_cleared": cache_size_before,
                "correlation_id": data.get("correlation_id")
            }
        )
        
        self.logger.info("Corridor cache cleared", entries_cleared=cache_size_before)
    
    def get_corridor_metrics(self) -> Dict[str, Any]:
        """Get corridor agent performance metrics"""
        return {
            "corridors_analyzed": self.corridors_analyzed,
            "bins_found_in_corridors": self.bins_found_in_corridors,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.corridor_cache),
            "avg_bins_per_corridor": (
                self.bins_found_in_corridors / max(1, self.corridors_analyzed)
            ),
            "config": {
                "corridor_half_m": self.corridor_config.corridor_half_m,
                "way_offset_thresh_m": self.corridor_config.way_offset_thresh_m,
                "max_detour_km": self.corridor_config.max_detour_km,
                "max_detour_ratio": self.corridor_config.max_detour_ratio,
                "candidate_scan_radius": self.corridor_config.candidate_scan_radius
            }
        }
    
    async def analyze_route_corridor(self, route_polyline: List[Tuple[float, float]], 
                                   route_wids: List[int], 
                                   all_bins: List[Bin],
                                   min_urgency: float = 0.8) -> Dict[str, Any]:
        """
        Analyze corridor for a specific route and return detailed results
        """
        
        try:
            # Filter bins by minimum urgency
            candidate_bins = [
                bin_obj for bin_obj in all_bins 
                if bin_obj.urgency_score() >= min_urgency
            ]
            
            # Perform corridor analysis
            corridor_bin_ids = await self.build_corridor(
                route_polyline, route_wids, candidate_bins
            )
            
            # Get detailed information about corridor bins
            corridor_bins = [
                bin_obj for bin_obj in candidate_bins 
                if bin_obj.id in corridor_bin_ids
            ]
            
            # Calculate corridor statistics
            if corridor_bins:
                avg_urgency = sum(b.urgency_score() for b in corridor_bins) / len(corridor_bins)
                max_urgency = max(b.urgency_score() for b in corridor_bins)
                total_waste = sum(b.capacity_l * (b.fill_level / 100.0) for b in corridor_bins)
            else:
                avg_urgency = 0.0
                max_urgency = 0.0
                total_waste = 0.0
            
            # Calculate corridor geometry stats
            if len(route_polyline) >= 2:
                total_route_length = sum(
                    distance_km(route_polyline[i][0], route_polyline[i][1],
                              route_polyline[i+1][0], route_polyline[i+1][1])
                    for i in range(len(route_polyline) - 1)
                )
            else:
                total_route_length = 0.0
            
            return {
                "corridor_bin_ids": list(corridor_bin_ids),
                "corridor_bins_count": len(corridor_bins),
                "candidate_bins_count": len(candidate_bins),
                "corridor_efficiency": len(corridor_bins) / max(1, len(candidate_bins)),
                "avg_urgency": avg_urgency,
                "max_urgency": max_urgency,
                "total_waste_liters": total_waste,
                "route_length_km": total_route_length,
                "bins_per_km": len(corridor_bins) / max(0.1, total_route_length),
                "config_used": {
                    "corridor_half_m": self.corridor_config.corridor_half_m,
                    "max_detour_km": self.corridor_config.max_detour_km,
                    "max_detour_ratio": self.corridor_config.max_detour_ratio
                }
            }
            
        except Exception as e:
            self.logger.error("Error in detailed corridor analysis", error=str(e))
            return {
                "corridor_bin_ids": [],
                "error": str(e)
            }
    
    async def validate_corridor_config(self) -> Dict[str, Any]:
        """Validate current corridor configuration"""
        
        validation_results = {
            "valid": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Check corridor half width
        if self.corridor_config.corridor_half_m < 100:
            validation_results["warnings"].append(
                f"Corridor half width ({self.corridor_config.corridor_half_m}m) is quite narrow"
            )
        elif self.corridor_config.corridor_half_m > 500:
            validation_results["warnings"].append(
                f"Corridor half width ({self.corridor_config.corridor_half_m}m) is quite wide"
            )
        
        # Check detour limits
        if self.corridor_config.max_detour_km > 1.0:
            validation_results["warnings"].append(
                f"Maximum detour ({self.corridor_config.max_detour_km}km) might include too many bins"
            )
        
        if self.corridor_config.max_detour_ratio > 0.1:
            validation_results["warnings"].append(
                f"Maximum detour ratio ({self.corridor_config.max_detour_ratio:.0%}) might be too high"
            )
        
        # Generate recommendations
        if self.corridor_config.corridor_half_m > 300 and self.corridor_config.max_detour_km > 0.5:
            validation_results["recommendations"].append(
                "Consider reducing either corridor width or max detour to improve selectivity"
            )
        
        if self.corridor_config.candidate_scan_radius < self.corridor_config.corridor_half_m * 3:
            validation_results["recommendations"].append(
                "Candidate scan radius should be at least 3x corridor half width for good coverage"
            )
        
        return validation_results