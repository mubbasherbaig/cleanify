"""
Cleanify v2-alpha OSRM Client
HTTP helpers for OSRM routing services: table, nearest, route
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import structlog

logger = structlog.get_logger()


class OSRMClient:
    """
    Asynchronous OSRM client for routing services
    Provides table, nearest, and route APIs
    """
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=30)
        
        # Performance metrics
        self.requests_made = 0
        self.total_response_time = 0.0
        self.errors_count = 0
        self.cache_hits = 0
        
        # Simple response cache
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_max_size = 1000
        self.cache_ttl_seconds = 300  # 5 minutes
    
    async def initialize(self):
        """Initialize OSRM client"""
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            connector=aiohttp.TCPConnector(limit=100)
        )
        
        # Test connection
        try:
            await self._test_connection()
            logger.info("OSRM client initialized", base_url=self.base_url)
        except Exception as e:
            logger.warning("OSRM connection test failed", error=str(e))
    
    async def cleanup(self):
        """Cleanup OSRM client"""
        if self.session:
            await self.session.close()
            logger.info("OSRM client cleaned up")
    
    async def _test_connection(self):
        """Test OSRM connection"""
        url = f"{self.base_url}/route/v1/driving/0,0;1,1"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                return True
            else:
                raise ConnectionError(f"OSRM test failed with status {response.status}")
    
    async def table(self, coordinates: List[Tuple[float, float]], 
                   sources: Optional[List[int]] = None,
                   destinations: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Get distance/duration table between coordinates
        
        Args:
            coordinates: List of (lat, lon) coordinate pairs
            sources: List of source indices (None = all coordinates)
            destinations: List of destination indices (None = all coordinates)
            
        Returns:
            Dict with durations and distances matrices
        """
        
        if len(coordinates) < 2:
            raise ValueError("At least 2 coordinates required for table service")
        
        # Build coordinate string (lon,lat format for OSRM)
        coord_string = ';'.join(f"{lon},{lat}" for lat, lon in coordinates)
        
        # Build URL
        url = f"{self.base_url}/table/v1/driving/{coord_string}"
        
        # Add parameters
        params = {}
        if sources is not None:
            params['sources'] = ';'.join(str(i) for i in sources)
        if destinations is not None:
            params['destinations'] = ';'.join(str(i) for i in destinations)
        
        # Check cache
        cache_key = f"table_{hash((coord_string, str(sources), str(destinations)))}"
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            self.cache_hits += 1
            return cached_response
        
        # Make request
        start_time = datetime.now()
        
        try:
            async with self.session.get(url, params=params) as response:
                response_time = (datetime.now() - start_time).total_seconds()
                self.total_response_time += response_time
                self.requests_made += 1
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache successful response
                    self._cache_response(cache_key, data)
                    
                    return data
                else:
                    error_text = await response.text()
                    raise Exception(f"OSRM table request failed: {response.status} - {error_text}")
                    
        except Exception as e:
            self.errors_count += 1
            logger.error("OSRM table request failed", error=str(e))
            raise
    
    async def nearest(self, coordinate: Tuple[float, float], 
                     number: int = 1) -> Dict[str, Any]:
        """
        Find nearest road network nodes to coordinate
        
        Args:
            coordinate: (lat, lon) coordinate pair
            number: Number of nearest points to return
            
        Returns:
            Dict with nearest waypoints
        """
        
        lat, lon = coordinate
        url = f"{self.base_url}/nearest/v1/driving/{lon},{lat}"
        
        params = {'number': number}
        
        # Check cache
        cache_key = f"nearest_{hash((lat, lon, number))}"
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            self.cache_hits += 1
            return cached_response
        
        # Make request
        start_time = datetime.now()
        
        try:
            async with self.session.get(url, params=params) as response:
                response_time = (datetime.now() - start_time).total_seconds()
                self.total_response_time += response_time
                self.requests_made += 1
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache successful response
                    self._cache_response(cache_key, data)
                    
                    return data
                else:
                    error_text = await response.text()
                    raise Exception(f"OSRM nearest request failed: {response.status} - {error_text}")
                    
        except Exception as e:
            self.errors_count += 1
            logger.error("OSRM nearest request failed", error=str(e))
            raise
    
    async def route(self, coordinates: List[Tuple[float, float]], 
                   alternatives: bool = False,
                   steps: bool = False,
                   geometries: str = "geojson",
                   overview: str = "full") -> Dict[str, Any]:
        """
        Get route between coordinates
        
        Args:
            coordinates: List of (lat, lon) coordinate pairs
            alternatives: Include alternative routes
            steps: Include turn-by-turn instructions
            geometries: Geometry format ('geojson', 'polyline', 'polyline6')
            overview: Geometry detail level ('full', 'simplified', 'false')
            
        Returns:
            Dict with route information
        """
        
        if len(coordinates) < 2:
            raise ValueError("At least 2 coordinates required for route service")
        
        # Build coordinate string (lon,lat format for OSRM)
        coord_string = ';'.join(f"{lon},{lat}" for lat, lon in coordinates)
        
        # Build URL
        url = f"{self.base_url}/route/v1/driving/{coord_string}"
        
        # Add parameters
        params = {
            'alternatives': 'true' if alternatives else 'false',
            'steps': 'true' if steps else 'false',
            'geometries': geometries,
            'overview': overview
        }
        
        # Check cache
        cache_key = f"route_{hash((coord_string, str(params)))}"
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            self.cache_hits += 1
            return cached_response
        
        # Make request
        start_time = datetime.now()
        
        try:
            async with self.session.get(url, params=params) as response:
                response_time = (datetime.now() - start_time).total_seconds()
                self.total_response_time += response_time
                self.requests_made += 1
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache successful response
                    self._cache_response(cache_key, data)
                    
                    return data
                else:
                    error_text = await response.text()
                    raise Exception(f"OSRM route request failed: {response.status} - {error_text}")
                    
        except Exception as e:
            self.errors_count += 1
            logger.error("OSRM route request failed", error=str(e))
            raise
    
    async def batch_table(self, coordinate_sets: List[List[Tuple[float, float]]]) -> List[Dict[str, Any]]:
        """
        Process multiple table requests in parallel
        
        Args:
            coordinate_sets: List of coordinate lists for separate table requests
            
        Returns:
            List of table response dicts
        """
        
        tasks = []
        for coordinates in coordinate_sets:
            task = asyncio.create_task(self.table(coordinates))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Batch table request failed", 
                           set_index=i, error=str(result))
            else:
                successful_results.append(result)
        
        return successful_results
    
    async def batch_routes(self, coordinate_sets: List[List[Tuple[float, float]]]) -> List[Dict[str, Any]]:
        """
        Process multiple route requests in parallel
        
        Args:
            coordinate_sets: List of coordinate lists for separate route requests
            
        Returns:
            List of route response dicts
        """
        
        tasks = []
        for coordinates in coordinate_sets:
            task = asyncio.create_task(self.route(coordinates))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Batch route request failed", 
                           set_index=i, error=str(result))
            else:
                successful_results.append(result)
        
        return successful_results
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if valid"""
        
        if cache_key in self.response_cache:
            cache_entry = self.response_cache[cache_key]
            
            # Check if cache entry is still valid
            age = (datetime.now() - cache_entry['timestamp']).total_seconds()
            if age < self.cache_ttl_seconds:
                return cache_entry['data']
            else:
                # Remove expired entry
                del self.response_cache[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, data: Dict[str, Any]):
        """Cache response data"""
        
        # Clean cache if too large
        if len(self.response_cache) >= self.cache_max_size:
            self._clean_cache()
        
        # Store new entry
        self.response_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def _clean_cache(self):
        """Clean old cache entries"""
        
        cutoff_time = datetime.now()
        cutoff_time = cutoff_time.replace(
            second=cutoff_time.second - self.cache_ttl_seconds
        )
        
        expired_keys = []
        for key, entry in self.response_cache.items():
            if entry['timestamp'] < cutoff_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.response_cache[key]
        
        # If still too large, remove oldest entries
        if len(self.response_cache) >= self.cache_max_size:
            sorted_entries = sorted(
                self.response_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            
            # Keep only newest half
            keep_count = self.cache_max_size // 2
            for key, _ in sorted_entries[:-keep_count]:
                del self.response_cache[key]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client performance metrics"""
        
        avg_response_time = (
            self.total_response_time / max(1, self.requests_made)
        )
        
        return {
            "requests": {
                "total_requests": self.requests_made,
                "total_errors": self.errors_count,
                "error_rate": self.errors_count / max(1, self.requests_made),
                "total_response_time_sec": self.total_response_time,
                "avg_response_time_sec": avg_response_time
            },
            "cache": {
                "cache_hits": self.cache_hits,
                "cache_size": len(self.response_cache),
                "cache_hit_rate": self.cache_hits / max(1, self.requests_made),
                "cache_ttl_seconds": self.cache_ttl_seconds
            },
            "configuration": {
                "base_url": self.base_url,
                "timeout_seconds": self.timeout.total,
                "cache_max_size": self.cache_max_size
            }
        }
    
    def clear_cache(self):
        """Clear all cached responses"""
        cleared_count = len(self.response_cache)
        self.response_cache.clear()
        logger.info("OSRM cache cleared", entries_cleared=cleared_count)
        return cleared_count
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on OSRM service"""
        
        try:
            # Test with simple route request
            test_coords = [(52.5, 13.4), (52.51, 13.41)]  # Berlin coordinates
            
            start_time = datetime.now()
            response = await self.route(test_coords)
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "healthy",
                "response_time_sec": response_time,
                "service_url": self.base_url,
                "test_route_found": len(response.get('routes', [])) > 0
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "service_url": self.base_url
            }
    
    async def get_distance_matrix(self, coordinates: List[Tuple[float, float]]) -> List[List[float]]:
        """
        Get simplified distance matrix in kilometers
        
        Args:
            coordinates: List of (lat, lon) coordinate pairs
            
        Returns:
            2D list of distances in kilometers
        """
        
        table_response = await self.table(coordinates)
        
        if 'distances' not in table_response:
            raise ValueError("No distances in table response")
        
        # Convert from meters to kilometers
        distances_km = []
        for row in table_response['distances']:
            km_row = [dist / 1000.0 if dist is not None else float('inf') for dist in row]
            distances_km.append(km_row)
        
        return distances_km
    
    async def get_duration_matrix(self, coordinates: List[Tuple[float, float]]) -> List[List[float]]:
        """
        Get simplified duration matrix in minutes
        
        Args:
            coordinates: List of (lat, lon) coordinate pairs
            
        Returns:
            2D list of durations in minutes
        """
        
        table_response = await self.table(coordinates)
        
        if 'durations' not in table_response:
            raise ValueError("No durations in table response")
        
        # Convert from seconds to minutes
        durations_min = []
        for row in table_response['durations']:
            min_row = [dur / 60.0 if dur is not None else float('inf') for dur in row]
            durations_min.append(min_row)
        
        return durations_min
    
    async def snap_to_road(self, coordinate: Tuple[float, float]) -> Tuple[float, float]:
        """
        Snap coordinate to nearest road
        
        Args:
            coordinate: (lat, lon) coordinate pair
            
        Returns:
            Snapped (lat, lon) coordinate pair
        """
        
        nearest_response = await self.nearest(coordinate, number=1)
        
        if 'waypoints' not in nearest_response or not nearest_response['waypoints']:
            raise ValueError("No waypoints in nearest response")
        
        waypoint = nearest_response['waypoints'][0]
        location = waypoint['location']
        
        # OSRM returns [lon, lat]
        return (location[1], location[0])
    
    def __str__(self) -> str:
        return f"OSRMClient(base_url='{self.base_url}')"
    
    def __repr__(self) -> str:
        return (f"OSRMClient(base_url='{self.base_url}', "
                f"requests_made={self.requests_made}, "
                f"cache_size={len(self.response_cache)})")