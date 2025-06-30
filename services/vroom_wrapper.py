"""
Cleanify v2-alpha VROOM Wrapper
OR-Tools wrapper for vehicle routing optimization (local, no HTTP)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import json
import structlog

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

from core.models import Bin, Truck, Route, RouteStop, RouteStatus
from core.geo import distance_km
from services.osrm_client import OSRMClient

logger = structlog.get_logger()


class VROOMSolution:
    """VROOM optimization solution container"""
    
    def __init__(self):
        self.status = "unknown"
        self.objective_value = 0.0
        self.routes: List[Dict[str, Any]] = []
        self.optimization_time_sec = 0.0
        self.total_distance_km = 0.0
        self.total_duration_min = 0.0
        self.unassigned_jobs: List[int] = []


class VROOMWrapper:
    """
    VROOM-style wrapper around OR-Tools for local vehicle routing optimization
    Provides VROOM-compatible interface while using OR-Tools internally
    """
    
    def __init__(self, osrm_client: Optional[OSRMClient] = None):
        self.osrm_client = osrm_client or OSRMClient()
        
        # Optimization settings
        self.default_timeout_sec = 30
        self.default_max_vehicles = 20
        self.default_max_jobs = 1000
        
        # Performance metrics
        self.optimizations_run = 0
        self.total_optimization_time = 0.0
        self.solutions_found = 0
        
        # Configuration
        self.vehicle_capacity_margin = 0.05  # 5% safety margin
        self.max_route_duration_hours = 8
        
        logger.info("VROOM wrapper initialized",
                   ortools_available=ORTOOLS_AVAILABLE)
    
    async def solve(self, problem_data: Dict[str, Any]) -> VROOMSolution:
        """
        Solve vehicle routing problem with VROOM-style input format
        
        Args:
            problem_data: VROOM-format problem definition
            
        Returns:
            VROOMSolution with optimized routes
        """
        
        if not ORTOOLS_AVAILABLE:
            raise RuntimeError("OR-Tools not available for optimization")
        
        start_time = datetime.now()
        solution = VROOMSolution()
        
        try:
            # Parse problem data
            vehicles = problem_data.get('vehicles', [])
            jobs = problem_data.get('jobs', [])
            matrix = problem_data.get('matrix', {})
            options = problem_data.get('options', {})
            
            # Validate input
            if not vehicles or not jobs:
                raise ValueError("Problem must have vehicles and jobs")
            
            # Prepare OR-Tools data
            ortools_data = await self._prepare_ortools_data(vehicles, jobs, matrix)
            
            # Create and solve OR-Tools model
            manager, routing, routing_solution = await self._solve_with_ortools(
                ortools_data, options
            )
            
            if routing_solution:
                # Extract solution
                solution = await self._extract_solution(
                    manager, routing, routing_solution, ortools_data, vehicles, jobs
                )
                solution.status = "success"
                self.solutions_found += 1
            else:
                solution.status = "no_solution"
                logger.warning("OR-Tools failed to find solution")
            
            # Calculate optimization time
            optimization_time = (datetime.now() - start_time).total_seconds()
            solution.optimization_time_sec = optimization_time
            self.total_optimization_time += optimization_time
            self.optimizations_run += 1
            
            logger.info("VROOM optimization completed",
                       status=solution.status,
                       optimization_time=optimization_time,
                       routes=len(solution.routes))
            
            return solution
            
        except Exception as e:
            logger.error("VROOM optimization failed", error=str(e))
            solution.status = "error"
            solution.optimization_time_sec = (datetime.now() - start_time).total_seconds()
            return solution
    
    async def _prepare_ortools_data(self, vehicles: List[Dict], jobs: List[Dict], 
                                   matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for OR-Tools optimization"""
        
        # Get distance and duration matrices
        if 'distances' in matrix and 'durations' in matrix:
            # Use provided matrices
            distance_matrix = matrix['distances']
            duration_matrix = matrix['durations']
        else:
            # Calculate matrices using OSRM
            coordinates = []
            
            # Add depot/vehicle locations
            for vehicle in vehicles:
                start_location = vehicle.get('start', [0, 0])
                coordinates.append((start_location[1], start_location[0]))  # OSRM uses lat,lon
            
            # Add job locations
            for job in jobs:
                job_location = job.get('location', [0, 0])
                coordinates.append((job_location[1], job_location[0]))
            
            # Get matrices from OSRM
            distance_matrix = await self.osrm_client.get_distance_matrix(coordinates)
            duration_matrix = await self.osrm_client.get_duration_matrix(coordinates)
        
        # Vehicle capacities and constraints
        vehicle_capacities = []
        for vehicle in vehicles:
            capacity = vehicle.get('capacity', [5000])  # Default 5000L capacity
            if isinstance(capacity, list):
                vehicle_capacities.append(capacity[0])
            else:
                vehicle_capacities.append(capacity)
        
        # Job demands and constraints
        job_demands = []
        job_time_windows = []
        
        for job in jobs:
            # Demand (waste amount)
            delivery = job.get('delivery', [150])  # Default 150L waste
            if isinstance(delivery, list):
                job_demands.append(delivery[0])
            else:
                job_demands.append(delivery)
            
            # Time windows
            time_windows = job.get('time_windows', [[0, 1440]])  # Default: 24 hours
            if time_windows:
                job_time_windows.append(time_windows[0])
            else:
                job_time_windows.append([0, 1440])
        
        # Service times
        service_times = []
        for job in jobs:
            service = job.get('service', 5)  # Default 5 minutes
            service_times.append(service)
        
        return {
            'num_vehicles': len(vehicles),
            'num_locations': len(vehicles) + len(jobs),
            'depot_indices': list(range(len(vehicles))),  # Each vehicle starts at its own depot
            'distance_matrix': distance_matrix,
            'duration_matrix': duration_matrix,
            'vehicle_capacities': vehicle_capacities,
            'job_demands': job_demands,
            'job_time_windows': job_time_windows,
            'service_times': service_times,
            'vehicles': vehicles,
            'jobs': jobs
        }
    
    async def _solve_with_ortools(self, data: Dict[str, Any], 
                                 options: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        """Solve using OR-Tools constraint solver"""
        
        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(
            data['num_locations'],
            data['num_vehicles'],
            data['depot_indices'],  # Start indices
            data['depot_indices']   # End indices (return to start)
        )
        
        # Create routing model
        routing = pywrapcp.RoutingModel(manager)
        
        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(data['distance_matrix'][from_node][to_node] * 1000)  # Convert to meters
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Capacity constraints
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            # Depot nodes have no demand
            if from_node < data['num_vehicles']:
                return 0
            # Job nodes have demand
            job_index = from_node - data['num_vehicles']
            if job_index < len(data['job_demands']):
                return data['job_demands'][job_index]
            return 0
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Time windows constraints
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            # Travel time
            travel_time = int(data['duration_matrix'][from_node][to_node])
            
            # Service time at from_node
            service_time = 0
            if from_node >= data['num_vehicles']:  # Job node
                job_index = from_node - data['num_vehicles']
                if job_index < len(data['service_times']):
                    service_time = data['service_times'][job_index]
            
            return travel_time + service_time
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            30,  # allow waiting time
            int(self.max_route_duration_hours * 60),  # maximum time per vehicle
            False,  # don't force start cumul to zero
            'Time'
        )
        
        # Add time window constraints for jobs
        time_dimension = routing.GetDimensionOrDie('Time')
        for job_index, time_window in enumerate(data['job_time_windows']):
            location_index = job_index + data['num_vehicles']  # Offset by number of vehicles
            index = manager.NodeToIndex(location_index)
            if index >= 0:  # Valid index
                time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        
        # Search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        
        # Set timeout
        timeout_sec = options.get('timeout', self.default_timeout_sec)
        search_parameters.time_limit.FromSeconds(timeout_sec)
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        
        return manager, routing, solution
    
    async def _extract_solution(self, manager, routing, solution, data: Dict[str, Any],
                               vehicles: List[Dict], jobs: List[Dict]) -> VROOMSolution:
        """Extract solution from OR-Tools result"""
        
        vroom_solution = VROOMSolution()
        vroom_solution.objective_value = solution.ObjectiveValue()
        
        total_distance = 0
        total_duration = 0
        
        # Extract routes for each vehicle
        for vehicle_id in range(data['num_vehicles']):
            route_distance = 0
            route_duration = 0
            route_load = 0
            route_steps = []
            
            index = routing.Start(vehicle_id)
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                
                # Create step
                step = {
                    'type': 'start' if node_index < data['num_vehicles'] else 'job',
                    'location': self._get_node_location(node_index, vehicles, jobs),
                    'id': node_index,
                    'arrival': 0,
                    'duration': 0,
                    'load': [route_load]
                }
                
                # Add job-specific information
                if node_index >= data['num_vehicles']:  # Job node
                    job_index = node_index - data['num_vehicles']
                    if job_index < len(jobs):
                        job = jobs[job_index]
                        step['job'] = job_index
                        step['service'] = job.get('service', 5)
                        
                        # Update load
                        delivery = job.get('delivery', [150])
                        if isinstance(delivery, list):
                            route_load += delivery[0]
                        else:
                            route_load += delivery
                        step['load'] = [route_load]
                
                route_steps.append(step)
                
                # Move to next location
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                
                if not routing.IsEnd(index):
                    from_node = manager.IndexToNode(previous_index)
                    to_node = manager.IndexToNode(index)
                    
                    # Add distance and duration
                    segment_distance = data['distance_matrix'][from_node][to_node]
                    segment_duration = data['duration_matrix'][from_node][to_node]
                    
                    route_distance += segment_distance
                    route_duration += segment_duration
            
            # Add end step (return to depot)
            if route_steps:
                vehicle = vehicles[vehicle_id]
                end_step = {
                    'type': 'end',
                    'location': vehicle.get('start', [0, 0]),
                    'id': vehicle_id,
                    'arrival': int(route_duration),
                    'duration': int(route_duration),
                    'load': [0]  # Unloaded at depot
                }
                route_steps.append(end_step)
            
            # Create route if it has job stops
            if len(route_steps) > 2:  # More than just start and end
                route = {
                    'vehicle': vehicle_id,
                    'cost': int(route_distance * 1000),  # Convert to meters
                    'steps': route_steps,
                    'distance': int(route_distance * 1000),
                    'duration': int(route_duration),
                    'service': sum(step.get('service', 0) for step in route_steps),
                    'delivery': [route_load],
                    'geometry': await self._get_route_geometry(route_steps)
                }
                
                vroom_solution.routes.append(route)
                total_distance += route_distance
                total_duration += route_duration
        
        vroom_solution.total_distance_km = total_distance
        vroom_solution.total_duration_min = total_duration
        
        return vroom_solution
    
    def _get_node_location(self, node_index: int, vehicles: List[Dict], jobs: List[Dict]) -> List[float]:
        """Get location coordinates for a node"""
        
        if node_index < len(vehicles):
            # Vehicle/depot node
            return vehicles[node_index].get('start', [0, 0])
        else:
            # Job node
            job_index = node_index - len(vehicles)
            if job_index < len(jobs):
                return jobs[job_index].get('location', [0, 0])
        
        return [0, 0]
    
    async def _get_route_geometry(self, route_steps: List[Dict]) -> Optional[str]:
        """Get route geometry from OSRM"""
        
        if len(route_steps) < 2:
            return None
        
        try:
            # Extract coordinates
            coordinates = []
            for step in route_steps:
                location = step['location']
                coordinates.append((location[1], location[0]))  # Convert to lat,lon
            
            # Get route from OSRM
            if self.osrm_client:
                route_response = await self.osrm_client.route(
                    coordinates, 
                    geometries='polyline'
                )
                
                if 'routes' in route_response and route_response['routes']:
                    return route_response['routes'][0]['geometry']
            
            return None
            
        except Exception as e:
            logger.warning("Failed to get route geometry", error=str(e))
            return None
    
    def create_vroom_problem(self, trucks: List[Truck], bins: List[Bin], 
                           depot_location: Tuple[float, float]) -> Dict[str, Any]:
        """
        Create VROOM-format problem from Cleanify objects
        
        Args:
            trucks: List of available trucks
            bins: List of bins to collect
            depot_location: (lat, lon) of depot
            
        Returns:
            VROOM-format problem dict
        """
        
        # Create vehicles
        vehicles = []
        for i, truck in enumerate(trucks):
            vehicle = {
                'id': i,
                'start': [depot_location[1], depot_location[0]],  # lon, lat
                'end': [depot_location[1], depot_location[0]],
                'capacity': [truck.capacity_l],
                'skills': [],
                'time_window': [0, self.max_route_duration_hours * 60]  # minutes
            }
            vehicles.append(vehicle)
        
        # Create jobs
        jobs = []
        for i, bin_obj in enumerate(bins):
            # Calculate waste amount
            waste_amount = int(bin_obj.capacity_l * (bin_obj.fill_level / 100.0))
            
            # Estimate collection time based on bin type
            service_time = 5  # Default 5 minutes
            if hasattr(bin_obj, 'bin_type'):
                if bin_obj.bin_type.value == 'industrial':
                    service_time = 8
                elif bin_obj.bin_type.value == 'commercial':
                    service_time = 6
                elif bin_obj.bin_type.value == 'residential':
                    service_time = 4
            
            job = {
                'id': i,
                'location': [bin_obj.lon, bin_obj.lat],  # lon, lat
                'delivery': [waste_amount],
                'service': service_time,
                'priority': self._get_job_priority(bin_obj),
                'time_windows': [[0, 1440]]  # 24 hours
            }
            
            jobs.append(job)
        
        problem = {
            'vehicles': vehicles,
            'jobs': jobs,
            'options': {
                'timeout': self.default_timeout_sec,
                'exploration_level': 5
            }
        }
        
        return problem
    
    def _get_job_priority(self, bin_obj: Bin) -> int:
        """Get job priority based on bin urgency"""
        
        urgency = bin_obj.urgency_score()
        
        if urgency >= 1.5:
            return 100  # Critical
        elif urgency >= 1.2:
            return 75   # High
        elif urgency >= 1.0:
            return 50   # Medium
        else:
            return 25   # Low
    
    def convert_solution_to_routes(self, solution: VROOMSolution, 
                                  trucks: List[Truck], bins: List[Bin]) -> List[Route]:
        """
        Convert VROOM solution to Cleanify Route objects
        
        Args:
            solution: VROOM optimization solution
            trucks: Original truck list
            bins: Original bin list
            
        Returns:
            List of Route objects
        """
        
        routes = []
        
        for vroom_route in solution.routes:
            vehicle_id = vroom_route['vehicle']
            
            if vehicle_id < len(trucks):
                truck = trucks[vehicle_id]
                
                # Extract stops from VROOM route steps
                route_stops = []
                
                for step in vroom_route['steps']:
                    if step['type'] == 'job':
                        job_id = step.get('job')
                        if job_id is not None and job_id < len(bins):
                            bin_obj = bins[job_id]
                            
                            stop = RouteStop(
                                id=f"stop_{bin_obj.id}",
                                lat=bin_obj.lat,
                                lon=bin_obj.lon,
                                stop_type="bin",
                                bin_id=bin_obj.id,
                                estimated_duration_min=step.get('service', 5)
                            )
                            
                            route_stops.append(stop)
                
                # Add depot return stop
                depot_stop = RouteStop(
                    id="depot_return",
                    lat=truck.lat,
                    lon=truck.lon,
                    stop_type="depot"
                )
                route_stops.append(depot_stop)
                
                # Create route
                route = Route(
                    id=f"route_{truck.id}_{int(datetime.now().timestamp())}",
                    truck_id=truck.id,
                    stops=route_stops,
                    status=RouteStatus.PLANNED,
                    created_at=datetime.now(),
                    total_distance_km=vroom_route['distance'] / 1000.0,
                    estimated_duration_min=vroom_route['duration'],
                    optimization_score=solution.objective_value
                )
                
                routes.append(route)
        
        return routes
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get VROOM wrapper performance metrics"""
        
        avg_optimization_time = (
            self.total_optimization_time / max(1, self.optimizations_run)
        )
        
        return {
            "optimization": {
                "optimizations_run": self.optimizations_run,
                "solutions_found": self.solutions_found,
                "success_rate": self.solutions_found / max(1, self.optimizations_run),
                "total_optimization_time_sec": self.total_optimization_time,
                "avg_optimization_time_sec": avg_optimization_time
            },
            "configuration": {
                "default_timeout_sec": self.default_timeout_sec,
                "max_vehicles": self.default_max_vehicles,
                "max_jobs": self.default_max_jobs,
                "max_route_duration_hours": self.max_route_duration_hours
            },
            "capabilities": {
                "ortools_available": ORTOOLS_AVAILABLE,
                "osrm_client_available": self.osrm_client is not None
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on VROOM wrapper"""
        
        try:
            # Test with minimal problem
            test_problem = {
                'vehicles': [{
                    'id': 0,
                    'start': [0, 0],
                    'end': [0, 0],
                    'capacity': [1000]
                }],
                'jobs': [{
                    'id': 0,
                    'location': [0.01, 0.01],
                    'delivery': [100],
                    'service': 5
                }],
                'matrix': {
                    'distances': [[0, 1000], [1000, 0]],
                    'durations': [[0, 60], [60, 0]]
                },
                'options': {
                    'timeout': 5
                }
            }
            
            start_time = datetime.now()
            solution = await self.solve(test_problem)
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "healthy" if solution.status == "success" else "unhealthy",
                "ortools_available": ORTOOLS_AVAILABLE,
                "test_optimization_time_sec": response_time,
                "test_solution_found": solution.status == "success"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "ortools_available": ORTOOLS_AVAILABLE
            }