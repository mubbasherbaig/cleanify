"""
Cleanify v2-alpha Route Planner Agent
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
        """Plan routes directly without tiling"""
        
        # Filter to urgent bins only
        urgent_bins = [b for b in bins if b.is_urgent()]
        available_trucks = [t for t in trucks if t.is_available()]
        
        if not urgent_bins or not available_trucks:
            return []
        
        if ORTOOLS_AVAILABLE:
            return await self._optimize_with_ortools(available_trucks, urgent_bins)
        else:
            return await self._optimize_with_greedy(available_trucks, urgent_bins)
    
    # In route_planner.py, make optimization async
    async def _optimize_with_ortools(self, trucks, bins):
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_ortools_sync, trucks, bins)
    
    async def _prepare_ortools_data(self, trucks: List[Truck], bins: List[Bin]) -> Dict[str, Any]:
        """Prepare data structure for OR-Tools"""
        
        # Create location list: [depot, bin1, bin2, ...]
        locations = [(trucks[0].lat, trucks[0].lon)]  # Depot at truck location
        for bin_obj in bins:
            locations.append((bin_obj.lat, bin_obj.lon))
        
        num_locations = len(locations)
        num_vehicles = len(trucks)
        depot = 0  # Depot is first location
        
        # Calculate distance matrix
        distance_matrix = []
        time_matrix = []
        
        for i, loc1 in enumerate(locations):
            distance_row = []
            time_row = []
            
            for j, loc2 in enumerate(locations):
                if i == j:
                    distance_row.append(0)
                    time_row.append(0)
                else:
                    # Calculate distance (simplified - in reality would use OSRM)
                    dist_km = distance_km(loc1[0], loc1[1], loc2[0], loc2[1])
                    distance_row.append(int(dist_km * 1000))  # Convert to meters
                    
                    # Estimate travel time (simplified)
                    travel_time_min = (dist_km / 30.0) * 60  # 30 km/h average
                    time_row.append(int(travel_time_min))
            
            distance_matrix.append(distance_row)
            time_matrix.append(time_row)
        
        # Calculate demands (bin waste amounts)
        demands = [0]  # Depot has no demand
        for bin_obj in bins:
            waste_amount = int(bin_obj.capacity_l * (bin_obj.fill_level / 100.0))
            demands.append(waste_amount)
        
        # Vehicle capacities
        vehicle_capacities = [truck.capacity_l for truck in trucks]
        
        # Service times
        service_times = [0]  # No service time at depot
        for _ in bins:
            service_times.append(5)  # 5 minutes per bin collection
        
        # Time windows (optional)
        time_windows = None
        if self.settings.ENABLE_DYNAMIC_THRESHOLDS:
            time_windows = [(0, 1440)]  # Depot: 24 hours
            for bin_obj in bins:
                # Calculate time window based on urgency
                urgency = bin_obj.urgency_score()
                if urgency >= 1.5:
                    time_windows.append((0, 60))  # Collect within 1 hour
                elif urgency >= 1.2:
                    time_windows.append((0, 180))  # Collect within 3 hours
                else:
                    time_windows.append((0, 360))  # Collect within 6 hours
        
        return {
            'num_locations': num_locations,
            'num_vehicles': num_vehicles,
            'depot': depot,
            'distance_matrix': distance_matrix,
            'time_matrix': time_matrix,
            'demands': demands,
            'vehicle_capacities': vehicle_capacities,
            'service_times': service_times,
            'time_windows': time_windows,
            'max_time': 480  # 8 hours max route time
        }
    
    async def _extract_ortools_solution(self, manager, routing, solution, data, 
                                       trucks: List[Truck], bins: List[Bin]) -> List[Route]:
        """Extract routes from OR-Tools solution"""
        
        routes = []
        
        for vehicle_id in range(data['num_vehicles']):
            if vehicle_id >= len(trucks):
                break
                
            truck = trucks[vehicle_id]
            route_stops = []
            total_distance = 0
            total_time = 0
            
            index = routing.Start(vehicle_id)
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                
                if node_index != data['depot']:  # Skip depot
                    # This is a bin location
                    bin_index = node_index - 1  # Adjust for depot offset
                    if 0 <= bin_index < len(bins):
                        bin_obj = bins[bin_index]
                        
                        stop = RouteStop(
                            id=f"stop_{bin_obj.id}",
                            lat=bin_obj.lat,
                            lon=bin_obj.lon,
                            stop_type="bin",
                            bin_id=bin_obj.id,
                            estimated_duration_min=5.0
                        )
                        route_stops.append(stop)
                
                # Get next index
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                
                if not routing.IsEnd(index):
                    total_distance += routing.GetArcCostForVehicle(
                        previous_index, index, vehicle_id
                    )
            
            # Add depot return stop
            depot_stop = RouteStop(
                id="depot_return",
                lat=truck.lat,
                lon=truck.lon,
                stop_type="depot"
            )
            route_stops.append(depot_stop)
            
            if route_stops:
                route = Route(
                    id=str(uuid.uuid4()),
                    truck_id=truck.id,
                    stops=route_stops,
                    status=RouteStatus.PLANNED,
                    created_at=datetime.now(),
                    total_distance_km=total_distance / 1000.0,
                    estimated_duration_min=total_time + len(route_stops) * 5,
                    optimization_score=solution.ObjectiveValue()
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
            
            while (remaining_bins and 
                   len(truck_bins) < self.settings.optimization.MAX_BINS_PER_ROUTE):
                
                # Find closest bin
                closest_bin = None
                min_distance = float('inf')
                
                for bin_obj in remaining_bins:
                    # Check capacity constraint
                    bin_waste = bin_obj.capacity_l * (bin_obj.fill_level / 100.0)
                    if truck_capacity_used + bin_waste > truck.capacity_l * 0.9:
                        continue
                    
                    # Calculate distance from truck or last bin
                    if truck_bins:
                        last_bin = truck_bins[-1]
                        dist = distance_km(last_bin.lat, last_bin.lon, bin_obj.lat, bin_obj.lon)
                    else:
                        dist = distance_km(truck.lat, truck.lon, bin_obj.lat, bin_obj.lon)
                    
                    if dist < min_distance:
                        min_distance = dist
                        closest_bin = bin_obj
                
                if closest_bin:
                    truck_bins.append(closest_bin)
                    remaining_bins.remove(closest_bin)
                    truck_capacity_used += closest_bin.capacity_l * (closest_bin.fill_level / 100.0)
                else:
                    break
            
            if truck_bins:
                route = await self._create_route_from_bins(truck, truck_bins)
                routes.append(route)
        
        return routes
    
    async def _create_route_from_bins(self, truck: Truck, bins: List[Bin]) -> Route:
        """Create route object from truck and bin list"""
        
        route_stops = []
        total_distance = 0.0
        
        # Add bin stops
        for bin_obj in bins:
            stop = RouteStop(
                id=f"stop_{bin_obj.id}",
                lat=bin_obj.lat,
                lon=bin_obj.lon,
                stop_type="bin",
                bin_id=bin_obj.id,
                estimated_duration_min=5.0
            )
            route_stops.append(stop)
        
        # Add depot return
        depot_stop = RouteStop(
            id="depot_return",
            lat=truck.lat,
            lon=truck.lon,
            stop_type="depot"
        )
        route_stops.append(depot_stop)
        
        # Calculate total distance
        current_location = (truck.lat, truck.lon)
        for stop in route_stops:
            dist = distance_km(current_location[0], current_location[1], stop.lat, stop.lon)
            total_distance += dist
            current_location = (stop.lat, stop.lon)
        
        route = Route(
            id=str(uuid.uuid4()),
            truck_id=truck.id,
            stops=route_stops,
            status=RouteStatus.PLANNED,
            created_at=datetime.now(),
            total_distance_km=total_distance,
            estimated_duration_min=total_distance / 30.0 * 60 + len(bins) * 5  # 30 km/h + 5 min per bin
        )
        
        return route
    
    async def _create_popup_route(self, truck: Truck, bins: List[Bin]) -> Route:
        """Create popup route using simple nearest neighbor"""
        
        if not bins:
            return None
        
        # Sort bins by urgency and distance
        def urgency_distance_score(bin_obj):
            urgency = bin_obj.urgency_score()
            dist = distance_km(truck.lat, truck.lon, bin_obj.lat, bin_obj.lon)
            return urgency * 2.0 - dist  # Prioritize urgency over distance
        
        sorted_bins = sorted(bins, key=urgency_distance_score, reverse=True)
        
        # Take only top bins that fit capacity
        selected_bins = []
        capacity_used = 0
        
        for bin_obj in sorted_bins:
            bin_waste = bin_obj.capacity_l * (bin_obj.fill_level / 100.0)
            if capacity_used + bin_waste <= truck.capacity_l * 0.9:
                selected_bins.append(bin_obj)
                capacity_used += bin_waste
                
                if len(selected_bins) >= 3:  # Limit popup routes to 3 bins
                    break
        
        return await self._create_route_from_bins(truck, selected_bins)
    
    def _filter_bins_by_capacity(self, truck: Truck, bins: List[Bin]) -> List[Bin]:
        """Filter bins that can fit in truck capacity"""
        
        feasible_bins = []
        
        for bin_obj in bins:
            bin_waste = bin_obj.capacity_l * (bin_obj.fill_level / 100.0)
            if bin_waste <= truck.capacity_l * 0.9:  # 90% capacity limit
                feasible_bins.append(bin_obj)
        
        return feasible_bins
    
    def _register_route_planner_handlers(self):
        """Register route planner specific message handlers"""
        self.register_handler("plan_routes_request", self._handle_plan_routes_request)
        self.register_handler("plan_popup_request", self._handle_plan_popup_request)
        self.register_handler("get_route_status", self._handle_get_route_status)
        self.register_handler("cancel_route", self._handle_cancel_route)
    
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
    
    async def _handle_get_route_status(self, data: Dict[str, Any]):
        """Handle route status request"""
        route_id = data.get("route_id")
        
        # Find route in completed routes
        route = next((r for r in self.completed_routes if r.id == route_id), None)
        
        if route:
            status_data = {
                "route_id": route_id,
                "status": route.status.value,
                "progress": route.progress_percentage(),
                "correlation_id": data.get("correlation_id")
            }
        else:
            status_data = {
                "error": f"Route {route_id} not found",
                "correlation_id": data.get("correlation_id")
            }
        
        await self.send_message("route_status", status_data)
    
    async def _handle_cancel_route(self, data: Dict[str, Any]):
        """Handle route cancellation request"""
        route_id = data.get("route_id")
        
        # Find and cancel route
        route = next((r for r in self.completed_routes if r.id == route_id), None)
        
        if route and route.status == RouteStatus.PLANNED:
            route.status = RouteStatus.CANCELLED
            
            await self.send_message(
                "route_cancelled",
                {
                    "route_id": route_id,
                    "correlation_id": data.get("correlation_id")
                }
            )
        else:
            await self.send_message(
                "route_cancel_failed",
                {
                    "error": f"Cannot cancel route {route_id}",
                    "correlation_id": data.get("correlation_id")
                }
            )
    
    async def _process_pending_requests(self):
        """Process any pending route requests"""
        # Implementation for background processing if needed
        pass
    
    async def _cleanup_old_requests(self):
        """Clean up old completed requests"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        to_remove = [
            req_id for req_id, req_data in self.active_requests.items()
            if req_data["timestamp"] < cutoff_time
        ]
        
        for req_id in to_remove:
            del self.active_requests[req_id]
    
    def _parse_truck_data(self, truck_data: Dict[str, Any]) -> Truck:
        """Parse truck data from message"""
        return Truck(
            id=truck_data["id"],
            name=truck_data["name"],
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