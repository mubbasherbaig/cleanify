"""
Cleanify v2-alpha Traffic Agent
Predicts traffic conditions and provides route delay estimates
"""

import asyncio
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json

from .base import AgentBase
from core.models import TrafficCondition
from core.settings import get_settings


class TrafficAgent(AgentBase):
    """
    Traffic agent that predicts traffic conditions and calculates route delays
    """
    
    def __init__(self):
        super().__init__("traffic", "traffic")
        
        # Traffic state
        self.current_conditions: Dict[str, TrafficCondition] = {}
        self.historical_patterns: Dict[int, Dict[str, float]] = {}  # hour -> region -> multiplier
        self.last_update = datetime.now()
        
        # Prediction models (simplified)
        self.rush_hour_patterns = {
            'morning': {'start': 7, 'end': 9, 'peak': 8, 'multiplier': 2.0},
            'lunch': {'start': 12, 'end': 14, 'peak': 13, 'multiplier': 1.4},
            'evening': {'start': 17, 'end': 19, 'peak': 18, 'multiplier': 2.2}
        }
        
        # Settings
        self.settings = get_settings()
        self.update_interval_sec = 30.0  # Update every 30 seconds
        
        # Performance metrics
        self.predictions_made = 0
        self.delay_calculations = 0
        
        # Register handlers
        self._register_traffic_handlers()
    
    async def initialize(self):
        """Initialize traffic agent"""
        self.logger.info("Initializing Traffic Agent")
        
        # Initialize base traffic conditions
        await self._initialize_traffic_conditions()
        
        # Load historical patterns
        self._initialize_historical_patterns()
        
        self.logger.info("Traffic agent initialized")
    
    async def main_loop(self):
        """Main traffic monitoring and prediction loop"""
        while self.running:
            try:
                # Update traffic conditions
                await self._update_traffic_conditions()
                
                # Publish traffic updates
                await self._publish_traffic_update()
                
                # Sleep until next update
                await asyncio.sleep(self.update_interval_sec)
                
            except Exception as e:
                self.logger.error("Error in traffic main loop", error=str(e))
                await asyncio.sleep(30)
    
    async def cleanup(self):
        """Cleanup traffic agent"""
        self.logger.info("Traffic agent cleanup")
    
    async def get_delay(self, route_id: str) -> float:
        """
        Get traffic delay multiplier for a route
        Returns delay factor (1.0 = no delay, 2.0 = double travel time)
        """
        try:
            # Get current conditions for default region
            condition = self.current_conditions.get("default")
            
            if not condition:
                # Fallback to real-time calculation
                delay = await self._calculate_realtime_delay(route_id)
            else:
                delay = condition.multiplier
            
            self.delay_calculations += 1
            
            self.logger.debug("Delay calculated", 
                            route_id=route_id, delay=delay)
            
            return delay
            
        except Exception as e:
            self.logger.error("Error calculating delay", 
                            route_id=route_id, error=str(e))
            return 1.2  # Default slight delay
    
    async def _initialize_traffic_conditions(self):
        """Initialize base traffic conditions for different regions"""
        
        # Default region condition
        self.current_conditions["default"] = TrafficCondition(
            timestamp=datetime.now(),
            level="light",
            multiplier=1.2,
            region="default",
            source="initialization"
        )
        
        self.logger.info("Base traffic conditions initialized")
    
    def _initialize_historical_patterns(self):
        """Initialize historical traffic patterns"""
        
        # Simplified patterns - in real system would load from database
        for hour in range(24):
            self.historical_patterns[hour] = {
                "default": self._calculate_base_multiplier(hour)
            }
        
        self.logger.info("Historical traffic patterns loaded")
    
    def _calculate_base_multiplier(self, hour: int) -> float:
        """Calculate base traffic multiplier for given hour"""
        
        # Night hours (22-5): Very light traffic
        if 22 <= hour or hour <= 5:
            return 1.0
        
        # Early morning (6-7): Building up
        elif 6 <= hour <= 7:
            return 1.1 + (hour - 6) * 0.1
        
        # Morning rush (7-9): Heavy traffic
        elif 7 <= hour <= 9:
            rush = self.rush_hour_patterns['morning']
            return self._gaussian_curve(hour, rush['peak'], rush['multiplier'])
        
        # Mid-morning (10-11): Moderate
        elif 10 <= hour <= 11:
            return 1.3
        
        # Lunch time (12-14): Moderate busy
        elif 12 <= hour <= 14:
            rush = self.rush_hour_patterns['lunch']
            return self._gaussian_curve(hour, rush['peak'], rush['multiplier'])
        
        # Afternoon (15-16): Building up
        elif 15 <= hour <= 16:
            return 1.4 + (hour - 15) * 0.1
        
        # Evening rush (17-19): Heavy traffic
        elif 17 <= hour <= 19:
            rush = self.rush_hour_patterns['evening']
            return self._gaussian_curve(hour, rush['peak'], rush['multiplier'])
        
        # Evening (20-21): Calming down
        else:
            return 1.4 - (hour - 19) * 0.1
    
    def _gaussian_curve(self, hour: int, peak_hour: int, peak_multiplier: float) -> float:
        """Generate gaussian curve around peak hour"""
        sigma = 0.5  # Standard deviation
        exponent = -((hour - peak_hour) ** 2) / (2 * sigma ** 2)
        curve_value = math.exp(exponent)
        
        # Scale to reach peak multiplier at peak hour
        base_multiplier = 1.0
        return base_multiplier + (peak_multiplier - base_multiplier) * curve_value
    
    async def _update_traffic_conditions(self):
        """Update current traffic conditions"""
        now = datetime.now()
        current_hour = now.hour
        day_of_week = now.weekday()  # 0=Monday, 6=Sunday
        
        # Get base multiplier from historical patterns
        base_multiplier = self.historical_patterns[current_hour]["default"]
        
        # Apply day-of-week adjustments
        day_multiplier = self._get_day_multiplier(day_of_week)
        
        # Apply random variation
        random_factor = random.uniform(0.9, 1.1)
        
        # Calculate final multiplier
        final_multiplier = base_multiplier * day_multiplier * random_factor
        
        # Determine traffic level
        traffic_level = self._categorize_traffic_level(final_multiplier)
        
        # Update current conditions
        self.current_conditions["default"] = TrafficCondition(
            timestamp=now,
            level=traffic_level,
            multiplier=final_multiplier,
            region="default",
            source="prediction"
        )
        
        self.last_update = now
    
    def _get_day_multiplier(self, day_of_week: int) -> float:
        """Get traffic multiplier based on day of week"""
        
        # Monday=0, Sunday=6
        if day_of_week == 6:  # Sunday
            return 0.6  # Much lighter traffic
        elif day_of_week == 5:  # Saturday
            return 0.7  # Lighter traffic
        elif day_of_week == 4:  # Friday
            return 1.1  # Slightly heavier
        else:  # Monday-Thursday
            return 1.0  # Normal
    
    def _categorize_traffic_level(self, multiplier: float) -> str:
        """Categorize traffic multiplier into level"""
        
        if multiplier <= 1.1:
            return "free"
        elif multiplier <= 1.3:
            return "light"
        elif multiplier <= 1.7:
            return "moderate"
        else:
            return "heavy"
    
    async def _calculate_realtime_delay(self, route_id: str) -> float:
        """Calculate real-time delay for specific route"""
        
        # In real implementation, this would:
        # 1. Query external traffic APIs
        # 2. Analyze route segments
        # 3. Apply ML models
        
        # For now, use current conditions with some route-specific variation
        base_delay = self.current_conditions["default"].multiplier
        route_variation = random.uniform(0.8, 1.2)
        
        return base_delay * route_variation
    
    async def _publish_traffic_update(self):
        """Publish traffic conditions to other agents"""
        
        traffic_data = {
            "conditions": {
                region: {
                    "timestamp": condition.timestamp.isoformat(),
                    "level": condition.level,
                    "multiplier": condition.multiplier,
                    "region": condition.region,
                    "source": condition.source
                }
                for region, condition in self.current_conditions.items()
            },
            "update_timestamp": datetime.now().isoformat()
        }
        
        await self.send_message(
            "traffic_update",
            traffic_data,
            target_stream="cleanify:traffic:updates"
        )
    
    def _register_traffic_handlers(self):
        """Register traffic-specific message handlers"""
        self.register_handler("get_delay", self._handle_get_delay)
        self.register_handler("get_conditions", self._handle_get_conditions)
        self.register_handler("predict_conditions", self._handle_predict_conditions)
        self.register_handler("update_patterns", self._handle_update_patterns)
    
    async def _handle_get_delay(self, data: Dict[str, Any]):
        """Handle delay request for specific route"""
        try:
            route_id = data.get("route_id", "default")
            delay = await self.get_delay(route_id)
            
            await self.send_message(
                "delay_response",
                {
                    "route_id": route_id,
                    "delay_multiplier": delay,
                    "timestamp": datetime.now().isoformat(),
                    "correlation_id": data.get("correlation_id")
                }
            )
            
        except Exception as e:
            self.logger.error("Error handling delay request", error=str(e))
            
            await self.send_message(
                "delay_error",
                {
                    "error": str(e),
                    "correlation_id": data.get("correlation_id")
                }
            )
    
    async def _handle_get_conditions(self, data: Dict[str, Any]):
        """Handle current conditions request"""
        try:
            region = data.get("region", "default")
            condition = self.current_conditions.get(region)
            
            if condition:
                response_data = {
                    "region": region,
                    "timestamp": condition.timestamp.isoformat(),
                    "level": condition.level,
                    "multiplier": condition.multiplier,
                    "source": condition.source,
                    "correlation_id": data.get("correlation_id")
                }
            else:
                response_data = {
                    "error": f"No conditions available for region {region}",
                    "correlation_id": data.get("correlation_id")
                }
            
            await self.send_message("conditions_response", response_data)
            
        except Exception as e:
            self.logger.error("Error handling conditions request", error=str(e))
    
    async def _handle_predict_conditions(self, data: Dict[str, Any]):
        """Handle traffic prediction request"""
        try:
            hours_ahead = data.get("hours_ahead", 1)
            region = data.get("region", "default")
            
            predictions = []
            
            for hour_offset in range(hours_ahead):
                future_time = datetime.now() + timedelta(hours=hour_offset)
                future_hour = future_time.hour
                future_day = future_time.weekday()
                
                # Predict based on historical patterns
                base_multiplier = self.historical_patterns[future_hour]["default"]
                day_multiplier = self._get_day_multiplier(future_day)
                predicted_multiplier = base_multiplier * day_multiplier
                
                predictions.append({
                    "timestamp": future_time.isoformat(),
                    "hour": future_hour,
                    "predicted_multiplier": predicted_multiplier,
                    "predicted_level": self._categorize_traffic_level(predicted_multiplier)
                })
            
            self.predictions_made += len(predictions)
            
            await self.send_message(
                "traffic_predictions",
                {
                    "region": region,
                    "predictions": predictions,
                    "correlation_id": data.get("correlation_id")
                }
            )
            
        except Exception as e:
            self.logger.error("Error handling prediction request", error=str(e))
    
    async def _handle_update_patterns(self, data: Dict[str, Any]):
        """Handle historical pattern updates"""
        try:
            new_patterns = data.get("patterns", {})
            
            # Update historical patterns
            for hour_str, regions in new_patterns.items():
                hour = int(hour_str)
                if hour not in self.historical_patterns:
                    self.historical_patterns[hour] = {}
                
                self.historical_patterns[hour].update(regions)
            
            self.logger.info("Traffic patterns updated",
                           updated_hours=len(new_patterns))
            
        except Exception as e:
            self.logger.error("Error updating patterns", error=str(e))
    
    def get_traffic_metrics(self) -> Dict[str, Any]:
        """Get traffic agent performance metrics"""
        return {
            "predictions_made": self.predictions_made,
            "delay_calculations": self.delay_calculations,
            "last_update": self.last_update.isoformat(),
            "current_conditions": {
                region: {
                    "level": condition.level,
                    "multiplier": condition.multiplier,
                    "age_seconds": (datetime.now() - condition.timestamp).total_seconds()
                }
                for region, condition in self.current_conditions.items()
            },
            "pattern_hours_loaded": len(self.historical_patterns)
        }
    
    async def simulate_incident(self, region: str = "default", 
                              severity: float = 2.0, 
                              duration_minutes: int = 30):
        """Simulate traffic incident for testing"""
        
        self.logger.info("Simulating traffic incident",
                        region=region, severity=severity, 
                        duration_minutes=duration_minutes)
        
        # Store original condition
        original_condition = self.current_conditions.get(region)
        
        # Create incident condition
        incident_condition = TrafficCondition(
            timestamp=datetime.now(),
            level="heavy",
            multiplier=severity,
            region=region,
            source="incident_simulation"
        )
        
        self.current_conditions[region] = incident_condition
        
        # Publish incident update
        await self._publish_traffic_update()
        
        # Schedule restoration
        await asyncio.sleep(duration_minutes * 60)
        
        # Restore original condition
        if original_condition:
            self.current_conditions[region] = original_condition
        else:
            await self._update_traffic_conditions()
        
        await self._publish_traffic_update()
        
        self.logger.info("Traffic incident simulation ended", region=region)