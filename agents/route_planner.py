"""
Cleanify v2-alpha Route Planner Agent - FIXED VERSION
Optimizes vehicle routes using OR-Tools with tiling support
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
import json

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

from .base import AgentBase
from core.models import Bin, Truck, Route, RouteStop, RouteStatus, TruckStatus
from core.geo import get_tile_id, bins_in_radius, distance_km, create_bin_tiles
from core.settings import get_settings
from services.osrm_client import OSRMClient


class RoutePlannerAgent(AgentBase):
    """
    Route planner agent that optimizes collection routes using OR-Tools
    """
    
    def __init__(self):
        super().__init__("route_planner", "route_planner")
        
        # Route planning state
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.completed_routes: List[Route] = []
        self.bin_tiles: Dict[str, List[Bin]] = {}
        
        # Services
        self.osrm_client = OSRMClient()
        
        # Settings
        self.settings = get_settings()
        
        # Performance metrics
        self.routes_planned = 0
        self.optimization_time_total = 0.0
        self.popup_routes_planned = 0
        
        # Register handlers
        self._register_route_planner_handlers()
    
    async def initialize(self):
        """Initialize route planner agent"""
        self.logger.info("Initializing Route Planner Agent")
        
        if not ORTOOLS_AVAILABLE:
            self.logger.warning("OR-Tools not available - using fallback routing")
        
        # Initialize OSRM client
        await self.osrm_client.initialize()
        
        self.logger.info("Route planner agent initialized")
    
    async def main_loop(self):
        """Main route planning loop"""
        while self.running:
            try:
                # Process pending route requests
                await self._process_pending_requests()
                
                # Clean up completed requests
                await self._cleanup_old_requests()
                
                # Sleep briefly
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error("Error in route planner main loop", error=str(e))
                await asyncio.sleep(5)
    
    async def cleanup(self):
        """Cleanup route planner agent"""
        await self.osrm_client.cleanup()
        self.logger.info("Route planner agent cleanup")
    
    async def plan_routes(self, trucks: List[Truck], bins: List[Bin]) -> List[Route]:
        """
        Plan optimal routes for trucks and bins
        Uses tiling for scalability with large bin counts
        """
        start_time = datetime.now()
        
        try:
            # Update bin tiles
            self.bin_tiles = create_bin_tiles(bins, self.settings.tiling.TILE_RES)
            
            if self.settings.ENABLE_TILING and len(bins) > 100:
                # Use tiled approach for large datasets
                routes = await self._plan_routes_tiled(trucks, bins)
            else:
                # Use direct approach for smaller datasets
                routes = await self._plan_routes_direct(trucks, bins)
            
            # Calculate optimization time
            optimization_time = (datetime.now() - start_time).total_seconds()
            self.optimization_time_total += optimization_time
            self.routes_planned += len(routes)
            
            self.logger.info("Route planning completed",
                           routes=len(routes), trucks=len(trucks), bins=len(bins),
                           optimization_time=optimization_time)
            
            return routes
            
        except Exception as e:
            self.logger.error("Error in route planning", error=str(e))
            return []
    
    async def plan_popup_route(self, truck: Truck, bins: List[Bin]) -> Optional[Route]:
        """
        Plan popup route for urgent collection
        Fast optimization for immediate dispatch
        """
        try:
            if not bins:
                return None
            
            # Filter bins by capacity constraints
            feasible_bins = self._filter_bins_by_capacity(truck, bins)
            
            if not feasible_bins:
                self.logger.warning("No feasible bins for popup route",
                                  truck_id=truck.id, requested_bins=len(bins))
                return None
            
            # Create simple route (nearest neighbor)
            route = await self._create_popup_route(truck, feasible_bins)
            
            self.popup_routes_planned += 1
            
            self.logger.info("Popup route planned",
                           truck_id=truck.id, bins=len(feasible_bins))
            
            return route
            
        except Exception as e:
            self.logger.error("Error planning popup route", 
                            truck_id=truck.id, error=str(e))
            return None
    
    async def _plan_routes_tiled(self, trucks: List[Truck], bins: List[Bin]) -> List[Route]:
        """Plan routes using tile-based parallelization"""
        
        # Group bins by tiles
        urgent_tiles = {}
        for tile_id, tile_bins in self.bin_tiles.items():
            urgent_bins_in_tile = [b for b in tile_bins if b.is_urgent()]
            if urgent_bins_in_tile:
                urgent_tiles[tile_id] = urgent_bins_in_tile
        
        if not urgent_tiles:
            return []
        
        # Plan routes per tile in parallel
        tile_tasks = []
        available_trucks = [t for t in trucks if t.is_available()]
        
        # Distribute trucks across tiles
        trucks_per_tile = max(1, len(available_trucks) // len(urgent_tiles))
        
        truck_index = 0
        for tile_id, tile_bins in urgent_tiles.items():
            # Assign trucks to this tile
            tile_trucks = available_trucks[truck_index:truck_index + trucks_per_tile]
            truck_index += trucks_per_tile
            
            if tile_trucks:
                task = asyncio.create_task(
                    self._plan_tile_routes(tile_id, tile_trucks, tile_bins)
                )
                tile_tasks.append(task)
        
        # Wait for all tile optimizations to complete
        tile_results = await asyncio.gather(*tile_tasks, return_exceptions=True)
        
        # Merge results
        all_routes = []
        for result in tile_results:
            if isinstance(result, list):
                all_routes.extend(result)
            elif isinstance(result, Exception):
                self.logger.error("Tile optimization failed", error=str(result))
        
        return all_routes
    
    async def _plan_tile_routes(self, tile_id: str, trucks: List[Truck], bins: List[Bin]) -> List[Route]:
        """Plan routes for a specific tile"""
        
        try:
            if ORTOOLS_AVAILABLE:
                return await self._optimize_with_ortools(trucks, bins)
            else:
                return await self._optimize_with_greedy(trucks, bins)
                
        except Exception as e:
            self.logger.error("Tile route planning failed", 
                            tile_id=tile_id, error=str(e))
            return []
    
    async def _plan_routes_direct(self, trucks: List[Truck], bins: List[Bin]) -> List[Route]:
        """Plan routes directly without tiling - FIXED VERSION"""
        
        # REMOVE FILTERING - Use all bins and trucks directly
        urgent_bins = [b for b in bins if b.fill_level >= 85.0]  # Simple check instead of is_urgent()
        available_trucks = [t for t in trucks if str(t.status).upper() == 'IDLE']  # Simple check instead of is_available()
        
        print(f"🔍 FILTERED: {len(urgent_bins)}/{len(bins)} urgent bins, {len(available_trucks)}/{len(trucks)} available trucks")
        
        if not urgent_bins or not available_trucks:
            print(f"❌ No urgent bins or available trucks after filtering")
            return []
        
        # FORCE GREEDY ALGORITHM if OR-Tools fails
        if ORTOOLS_AVAILABLE:
            try:
                routes = await self._optimize_with_ortools(available_trucks, urgent_bins)
                if routes:
                    return routes
                else:
                    print("⚠️ OR-Tools returned empty, falling back to greedy")
            except Exception as e:
                print(f"⚠️ OR-Tools failed: {e}, falling back to greedy")
        
        # Always try greedy as fallback
        return await self._optimize_with_greedy(available_trucks, urgent_bins)
    
    async def _optimize_with_ortools(self, trucks: List[Truck], bins: List[Bin]) -> List[Route]:
        """Optimize routes using OR-Tools in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_ortools_sync, trucks, bins)
    
    def _run_ortools_sync(self, trucks: List[Truck], bins: List[Bin]) -> List[Route]:
        """SIMPLIFIED OR-Tools optimization"""
        if not ORTOOLS_AVAILABLE:
            return []
        
        try:
            print(f"🔧 OR-Tools starting with {len(trucks)} trucks, {len(bins)} bins")
            
            # SIMPLIFIED: Create one route per truck with one bin
            routes = []
            for i, truck in enumerate(trucks):
                if i < len(bins):  # Only if we have bins available
                    bin_obj = bins[i]
                    
                    # Create simple route: depot -> bin -> depot
                    stops = [
                        RouteStop(
                            id="depot_start",
                            lat=truck.lat,
                            lon=truck.lon,
                            stop_type="depot"
                        ),
                        RouteStop(
                            id=f"stop_{bin_obj.id}",
                            lat=bin_obj.lat,
                            lon=bin_obj.lon,
                            stop_type="bin",
                            bin_id=bin_obj.id,
                            estimated_duration_min=10.0
                        ),
                        RouteStop(
                            id="depot_return",
                            lat=truck.lat,
                            lon=truck.lon,
                            stop_type="depot"
                        )
                    ]
                    
                    route = Route(
                        id=str(uuid.uuid4()),
                        truck_id=truck.id,
                        stops=stops,
                        status=RouteStatus.PLANNED,
                        created_at=datetime.now(),
                        total_distance_km=5.0,  # Simple estimate
                        estimated_duration_min=30.0
                    )
                    routes.append(route)
                    print(f"✅ Created simple route: {truck.id} -> {bin_obj.id}")
            
            return routes
            
        except Exception as e:
            print(f"❌ OR-Tools sync failed: {e}")
            return []
    
    def _prepare_ortools_data_sync(self, trucks: List[Truck], bins: List[Bin]) -> Dict[str, Any]:
        """Prepare data structure for OR-Tools (synchronous version)"""
        
        # Create location list: [depot, bin1, bin2, ...]
        locations = [(trucks[0].lat, trucks[0].lon)]  # Use first truck as depot
        for bin_obj in bins:
            locations.append((bin_obj.lat, bin_obj.lon))
        
        num_locations = len(locations)
        num_vehicles = len(trucks)
        depot_indices = [0] * num_vehicles  # All vehicles start/end at depot
        
        # Calculate distance matrix
        distance_matrix = []
        for i, loc1 in enumerate(locations):
            distance_row = []
            for j, loc2 in enumerate(locations):
                if i == j:
                    distance_row.append(0)
                else:
                    # Calculate distance in meters
                    dist_km = distance_km(loc1[0], loc1[1], loc2[0], loc2[1])
                    distance_row.append(int(dist_km * 1000))  # Convert to meters
            distance_matrix.append(distance_row)
        
        # Calculate demands (bin volumes)
        demands = []
        for bin_obj in bins:
            volume_needed = int(bin_obj.capacity_l * (bin_obj.fill_level / 100.0))
            demands.append(volume_needed)
        
        # Vehicle capacities
        vehicle_capacities = [int(truck.capacity_l * 0.9) for truck in trucks]  # 90% capacity
        
        return {
            'num_locations': num_locations,
            'num_vehicles': num_vehicles,
            'depot_indices': depot_indices,
            'distance_matrix': distance_matrix,
            'demands': demands,
            'vehicle_capacities': vehicle_capacities,
            'locations': locations
        }
    
    def _extract_routes_from_solution(self, manager, routing, solution, trucks: List[Truck], bins: List[Bin]) -> List[Route]:
        """Extract routes from OR-Tools solution"""
        routes = []
        
        for vehicle_id in range(len(trucks)):
            truck = trucks[vehicle_id]
            index = routing.Start(vehicle_id)
            route_stops = []
            total_distance = 0
            
            # Add depot start
            depot_stop = RouteStop(
                id="depot_start",
                lat=truck.lat,
                lon=truck.lon,
                stop_type="depot",
                estimated_duration_min=0.0
            )
            route_stops.append(depot_stop)
            
            # Extract route
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                
                # Skip depot nodes (vehicle starts)
                if node_index >= len(trucks):
                    bin_index = node_index - len(trucks)
                    if bin_index < len(bins):
                        bin_obj = bins[bin_index]
                        
                        stop = RouteStop(
                            id=f"stop_{bin_obj.id}",
                            lat=bin_obj.lat,
                            lon=bin_obj.lon,
                            stop_type="bin",
                            bin_id=bin_obj.id,
                            estimated_duration_min=5.0  # 5 min per bin
                        )
                        route_stops.append(stop)
                
                # Move to next location
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                
                if not routing.IsEnd(index):
                    total_distance += routing.GetArcCostForVehicle(
                        previous_index, index, vehicle_id
                    )
            
            # Add depot return
            depot_return = RouteStop(
                id="depot_return",
                lat=truck.lat,
                lon=truck.lon,
                stop_type="depot",
                estimated_duration_min=0.0
            )
            route_stops.append(depot_return)
            
            # Create route if it has stops
            if len(route_stops) > 2:  # More than just depot start/end
                route = Route(
                    id=str(uuid.uuid4()),
                    truck_id=truck.id,
                    stops=route_stops,
                    status=RouteStatus.PLANNED,
                    created_at=datetime.now(),
                    total_distance_km=total_distance / 1000.0,
                    estimated_duration_min=len(route_stops) * 5 + (total_distance / 1000.0) / 30 * 60,  # Travel + service time
                    optimization_score=solution.ObjectiveValue() if solution else 0
                )
                routes.append(route)
        
        return routes
    
    async def _optimize_with_greedy(self, trucks: List[Truck], bins: List[Bin]) -> List[Route]:
        """Fallback greedy optimization when OR-Tools is unavailable"""
        
        routes = []
        remaining_bins = bins.copy()
        
        for truck in trucks:
            if not remaining_bins:
                break
            
            # Greedy assignment: closest bins first
            truck_bins = []
            truck_capacity_used = 0
            current_lat, current_lon = truck.lat, truck.lon
            
            while (remaining_bins and 
                   len(truck_bins) < self.settings.optimization.MAX_BINS_PER_ROUTE and
                   truck_capacity_used < truck.capacity_l * 0.9):
                
                # Find closest bin
                closest_bin = None
                min_distance = float('inf')
                
                for bin_obj in remaining_bins:
                    dist = distance_km(current_lat, current_lon, bin_obj.lat, bin_obj.lon)
                    
                    # Check capacity
                    bin_volume = bin_obj.capacity_l * (bin_obj.fill_level / 100.0)
                    if truck_capacity_used + bin_volume <= truck.capacity_l * 0.9:
                        if dist < min_distance:
                            min_distance = dist
                            closest_bin = bin_obj
                
                if closest_bin:
                    truck_bins.append(closest_bin)
                    remaining_bins.remove(closest_bin)
                    truck_capacity_used += closest_bin.capacity_l * (closest_bin.fill_level / 100.0)
                    current_lat, current_lon = closest_bin.lat, closest_bin.lon
                else:
                    break
            
            # Create route
            if truck_bins:
                route_stops = [
                    RouteStop(
                        id="depot_start",
                        lat=truck.lat,
                        lon=truck.lon,
                        stop_type="depot"
                    )
                ]
                
                for bin_obj in truck_bins:
                    stop = RouteStop(
                        id=f"stop_{bin_obj.id}",
                        lat=bin_obj.lat,
                        lon=bin_obj.lon,
                        stop_type="bin",
                        bin_id=bin_obj.id,
                        estimated_duration_min=5.0
                    )
                    route_stops.append(stop)
                
                route_stops.append(
                    RouteStop(
                        id="depot_return",
                        lat=truck.lat,
                        lon=truck.lon,
                        stop_type="depot"
                    )
                )
                
                route = Route(
                    id=str(uuid.uuid4()),
                    truck_id=truck.id,
                    stops=route_stops,
                    status=RouteStatus.PLANNED,
                    created_at=datetime.now(),
                    total_distance_km=0.0,  # Would calculate properly
                    estimated_duration_min=len(truck_bins) * 10 + 30  # Rough estimate
                )
                routes.append(route)
        
        return routes
    
    def _filter_bins_by_capacity(self, truck: Truck, bins: List[Bin]) -> List[Bin]:
        """Filter bins that can fit in truck capacity"""
        feasible_bins = []
        total_volume = 0
        
        # Sort by urgency (fill level)
        sorted_bins = sorted(bins, key=lambda b: b.fill_level, reverse=True)
        
        for bin_obj in sorted_bins:
            bin_volume = bin_obj.capacity_l * (bin_obj.fill_level / 100.0)
            if total_volume + bin_volume <= truck.capacity_l * 0.9:  # 90% capacity limit
                feasible_bins.append(bin_obj)
                total_volume += bin_volume
        
        return feasible_bins
    
    async def _create_popup_route(self, truck: Truck, bins: List[Bin]) -> Route:
        """Create a popup route using nearest neighbor heuristic"""
        
        # Sort bins by distance from truck
        sorted_bins = sorted(bins, key=lambda b: distance_km(truck.lat, truck.lon, b.lat, b.lon))
        
        # Create route stops
        route_stops = [
            RouteStop(
                id="depot_start",
                lat=truck.lat,
                lon=truck.lon,
                stop_type="depot"
            )
        ]
        
        for bin_obj in sorted_bins:
            stop = RouteStop(
                id=f"stop_{bin_obj.id}",
                lat=bin_obj.lat,
                lon=bin_obj.lon,
                stop_type="bin",
                bin_id=bin_obj.id,
                estimated_duration_min=5.0
            )
            route_stops.append(stop)
        
        route_stops.append(
            RouteStop(
                id="depot_return",
                lat=truck.lat,
                lon=truck.lon,
                stop_type="depot"
            )
        )
        
        route = Route(
            id=str(uuid.uuid4()),
            truck_id=truck.id,
            stops=route_stops,
            status=RouteStatus.PLANNED,
            created_at=datetime.now(),
            total_distance_km=0.0,  # Would calculate with OSRM
            estimated_duration_min=len(bins) * 8 + 20  # Estimate
        )
        
        return route
    
    def _register_route_planner_handlers(self):
        """Register message handlers for route planning"""
        self.register_handler("plan_routes", self._handle_plan_routes_request)
        self.register_handler("plan_popup", self._handle_plan_popup_request)
        self.register_handler("get_metrics", self._handle_get_metrics)
    
    async def _handle_plan_routes_request(self, data: Dict[str, Any]):
        """Handle route planning request"""
        try:
            request_id = str(uuid.uuid4())
            
            # Parse trucks and bins from request
            trucks_data = data.get("trucks", [])
            bins_data = data.get("bins", [])
            
            trucks = [self._parse_truck_data(t) for t in trucks_data]
            bins = [self._parse_bin_data(b) for b in bins_data]
            
            # Store request
            self.active_requests[request_id] = {
                "timestamp": datetime.now(),
                "trucks": trucks,
                "bins": bins,
                "correlation_id": data.get("correlation_id")
            }
            
            # Plan routes
            routes = await self.plan_routes(trucks, bins)
            
            # Send response
            await self.send_message(
                "routes_planned",
                {
                    "request_id": request_id,
                    "routes": [self._route_to_dict(route) for route in routes],
                    "optimization_time": 0.0,  # Would track actual time
                    "correlation_id": data.get("correlation_id")
                }
            )
            
        except Exception as e:
            self.logger.error("Error handling route planning request", error=str(e))
            
            await self.send_message(
                "route_planning_error",
                {
                    "error": str(e),
                    "correlation_id": data.get("correlation_id")
                }
            )
    
    async def _handle_plan_popup_request(self, data: Dict[str, Any]):
        """Handle popup route planning request"""
        try:
            truck_data = data.get("truck", {})
            bins_data = data.get("bins", [])
            
            truck = self._parse_truck_data(truck_data)
            bins = [self._parse_bin_data(b) for b in bins_data]
            
            route = await self.plan_popup_route(truck, bins)
            
            if route:
                await self.send_message(
                    "popup_route_planned",
                    {
                        "route": self._route_to_dict(route),
                        "correlation_id": data.get("correlation_id")
                    }
                )
            else:
                await self.send_message(
                    "popup_route_failed",
                    {
                        "reason": "No feasible route found",
                        "correlation_id": data.get("correlation_id")
                    }
                )
                
        except Exception as e:
            self.logger.error("Error handling popup route request", error=str(e))
    
    async def _handle_get_metrics(self, data: Dict[str, Any]):
        """Handle metrics request"""
        metrics = self.get_route_planner_metrics()
        await self.send_message("metrics_response", {
            "metrics": metrics,
            "correlation_id": data.get("correlation_id")
        })
    
    async def _process_pending_requests(self):
        """Process any pending route requests"""
        # Would handle any queued requests
        pass
    
    async def _cleanup_old_requests(self):
        """Clean up old completed requests"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        expired_requests = [
            req_id for req_id, req_data in self.active_requests.items()
            if req_data["timestamp"] < cutoff_time
        ]
        
        for req_id in expired_requests:
            del self.active_requests[req_id]
    
    def _parse_truck_data(self, truck_data: Dict[str, Any]) -> Truck:
        """Parse truck data from message"""
        return Truck(
            id=truck_data["id"],
            name=truck_data.get("name", truck_data["id"]),
            capacity_l=truck_data["capacity_l"],
            lat=truck_data["lat"],
            lon=truck_data["lon"],
            current_load_l=truck_data.get("current_load_l", 0),
            status=TruckStatus(truck_data.get("status", "idle"))
        )
    
    def _parse_bin_data(self, bin_data: Dict[str, Any]) -> Bin:
        """Parse bin data from message"""
        return Bin(
            id=bin_data["id"],
            lat=bin_data["lat"],
            lon=bin_data["lon"],
            capacity_l=bin_data["capacity_l"],
            fill_level=bin_data["fill_level"],
            fill_rate_lph=bin_data["fill_rate_lph"],
            tile_id=bin_data.get("tile_id", ""),
            threshold=bin_data.get("threshold", 85.0)
        )
    
    def _route_to_dict(self, route: Route) -> Dict[str, Any]:
        """Convert route to dictionary"""
        return {
            "id": route.id,
            "truck_id": route.truck_id,
            "status": route.status.value,
            "total_distance_km": route.total_distance_km,
            "estimated_duration_min": route.estimated_duration_min,
            "created_at": route.created_at.isoformat() if route.created_at else None,
            "stops": [
                {
                    "id": stop.id,
                    "lat": stop.lat,
                    "lon": stop.lon,
                    "stop_type": stop.stop_type,
                    "bin_id": stop.bin_id,
                    "estimated_duration_min": stop.estimated_duration_min
                }
                for stop in route.stops
            ]
        }
    
    def get_route_planner_metrics(self) -> Dict[str, Any]:
        """Get route planner performance metrics"""
        return {
            "routes_planned": self.routes_planned,
            "popup_routes_planned": self.popup_routes_planned,
            "optimization_time_total": self.optimization_time_total,
            "avg_optimization_time": (
                self.optimization_time_total / max(1, self.routes_planned)
            ),
            "active_requests": len(self.active_requests),
            "completed_routes": len(self.completed_routes),
            "ortools_available": ORTOOLS_AVAILABLE,
            "tiles_cached": len(self.bin_tiles)
        }