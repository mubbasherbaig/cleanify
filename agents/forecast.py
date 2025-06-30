"""
Cleanify v2-alpha Forecast Agent
Predicts bin fill levels and times to 120% capacity using XGBoost
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .base import AgentBase
from core.models import Bin, BinType
from core.settings import get_settings


class ForecastAgent(AgentBase):
    """
    Forecast agent that predicts bin fill levels and overflow times
    """
    
    def __init__(self):
        super().__init__("forecast", "forecast")
        
        # ML models
        self.fill_model: Optional[xgb.XGBRegressor] = None
        self.time_model: Optional[xgb.XGBRegressor] = None
        self.model_trained = False
        self.last_training = None
        
        # Data storage
        self.historical_data: List[Dict[str, Any]] = []
        self.predictions_cache: Dict[str, Dict[str, float]] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        
        # Features for prediction
        self.feature_columns = [
            'hour', 'day_of_week', 'month', 'season',
            'bin_type_encoded', 'capacity_l', 'current_fill_level',
            'fill_rate_lph', 'last_collection_hours_ago',
            'temperature', 'weather_encoded', 'is_holiday',
            'nearby_bins_avg_fill', 'traffic_multiplier'
        ]
        
        # Settings
        self.settings = get_settings()
        self.cache_ttl_sec = self.settings.ml.PREDICTION_CACHE_TTL_SEC
        
        # Register handlers
        self._register_forecast_handlers()
    
    async def initialize(self):
        """Initialize forecast agent"""
        self.logger.info("Initializing Forecast Agent")
        
        if not XGBOOST_AVAILABLE:
            self.logger.warning("XGBoost not available - using fallback predictions")
        else:
            # Initialize models
            self._initialize_models()
        
        # Start data collection
        self.logger.info("Forecast agent initialized")
    
    async def main_loop(self):
        """Main forecast loop"""
        while self.running:
            try:
                # Update models if needed
                await self._update_models()
                
                # Clean expired cache entries
                await self._clean_cache()
                
                # Sleep based on update interval
                await asyncio.sleep(self.settings.ml.MODEL_UPDATE_INTERVAL_MIN * 60)
                
            except Exception as e:
                self.logger.error("Error in forecast main loop", error=str(e))
                await asyncio.sleep(60)
    
    async def cleanup(self):
        """Cleanup forecast agent"""
        self.logger.info("Forecast agent cleanup")
    
    def _initialize_models(self):
        """Initialize XGBoost models"""
        if not XGBOOST_AVAILABLE:
            return
        
        # Fill level prediction model
        self.fill_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Time to overflow prediction model
        self.time_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.logger.info("XGBoost models initialized")
    
    async def predict_minutes_to_120(self, bins: List[Bin]) -> Dict[str, float]:
        """
        Predict minutes until each bin reaches 120% capacity
        Returns dict of bin_id -> minutes
        """
        predictions = {}
        
        for bin_obj in bins:
            # Check cache first
            if self._is_prediction_cached(bin_obj.id):
                predictions[bin_obj.id] = self.predictions_cache[bin_obj.id]['minutes_to_120']
                continue
            
            # Generate prediction
            minutes_to_120 = await self._predict_single_bin_overflow(bin_obj)
            predictions[bin_obj.id] = minutes_to_120
            
            # Cache result
            self._cache_prediction(bin_obj.id, {'minutes_to_120': minutes_to_120})
        
        self.logger.debug("Predicted overflow times", 
                         bin_count=len(bins),
                         min_time=min(predictions.values()) if predictions else 0,
                         max_time=max(predictions.values()) if predictions else 0)
        
        return predictions
    
    async def _predict_single_bin_overflow(self, bin_obj: Bin) -> float:
        """Predict minutes until single bin reaches 120%"""
        
        if not XGBOOST_AVAILABLE or not self.model_trained:
            return self._fallback_linear_prediction(bin_obj)
        
        try:
            # Prepare features
            features = self._extract_features(bin_obj)
            feature_array = np.array([features]).reshape(1, -1)
            
            # Predict using trained model
            minutes_pred = self.time_model.predict(feature_array)[0]
            
            # Ensure reasonable bounds
            return max(0.0, min(10080.0, minutes_pred))  # Max 1 week
            
        except Exception as e:
            self.logger.warning("ML prediction failed, using fallback", 
                              bin_id=bin_obj.id, error=str(e))
            return self._fallback_linear_prediction(bin_obj)
    
    def _fallback_linear_prediction(self, bin_obj: Bin) -> float:
        """Fallback linear prediction when ML is unavailable"""
        
        if bin_obj.fill_rate_lph <= 0:
            return float('inf')  # Never fills
        
        current_fill = bin_obj.fill_level
        target_fill = 120.0
        
        if current_fill >= target_fill:
            return 0.0  # Already at target
        
        remaining_fill = target_fill - current_fill
        fill_rate_per_minute = bin_obj.fill_rate_lph / 60.0
        
        # Apply time-of-day multiplier
        now = datetime.now()
        time_multiplier = self._get_time_multiplier(now.hour, bin_obj.bin_type)
        
        adjusted_rate = fill_rate_per_minute * time_multiplier
        
        if adjusted_rate <= 0:
            return float('inf')
        
        minutes_to_target = remaining_fill / adjusted_rate
        return max(0.0, minutes_to_target)
    
    def _extract_features(self, bin_obj: Bin) -> List[float]:
        """Extract ML features from bin object"""
        now = datetime.now()
        
        # Time features
        hour = now.hour
        day_of_week = now.weekday()
        month = now.month
        season = (month - 1) // 3  # 0-3 for seasons
        
        # Bin type encoding
        bin_type_encoding = {
            BinType.RESIDENTIAL: 0,
            BinType.COMMERCIAL: 1,
            BinType.INDUSTRIAL: 2,
            BinType.ORGANIC: 3,
            BinType.RECYCLING: 4,
            BinType.GENERAL: 5,
            BinType.MEDICAL: 6
        }
        bin_type_encoded = bin_type_encoding.get(bin_obj.bin_type, 5)
        
        # Last collection estimate (simplified)
        last_collection_hours_ago = 24.0  # Default assumption
        if bin_obj.last_collected:
            last_collection_hours_ago = (now - bin_obj.last_collected).total_seconds() / 3600
        
        # Environmental factors (simplified - would come from weather service)
        temperature = 20.0  # Default temperature
        weather_encoded = 0  # 0=sunny, 1=rainy, 2=cloudy
        is_holiday = 0  # Binary holiday indicator
        
        # Spatial features (simplified)
        nearby_bins_avg_fill = bin_obj.fill_level  # Would calculate from nearby bins
        traffic_multiplier = 1.2  # Would come from traffic agent
        
        features = [
            hour, day_of_week, month, season,
            bin_type_encoded, bin_obj.capacity_l, bin_obj.fill_level,
            bin_obj.fill_rate_lph, last_collection_hours_ago,
            temperature, weather_encoded, is_holiday,
            nearby_bins_avg_fill, traffic_multiplier
        ]
        
        return features
    
    def _get_time_multiplier(self, hour: int, bin_type: BinType) -> float:
        """Get time-of-day multiplier for fill rate"""
        
        # Base patterns by bin type
        if bin_type in [BinType.COMMERCIAL, BinType.GENERAL]:
            # Business hours pattern
            business_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            if hour in business_hours:
                return 1.2
            elif 18 <= hour <= 20:  # Evening
                return 0.8
            elif 22 <= hour or hour <= 5:  # Night
                return 0.2
            else:
                return 1.0
                
        elif bin_type == BinType.RESIDENTIAL:
            # Home activity pattern
            if hour in [7, 8, 18, 19, 20]:  # Morning and evening peaks
                return 1.3
            elif 9 <= hour <= 17:  # Work hours (people away)
                return 0.5
            elif 21 <= hour or hour <= 6:  # Night
                return 0.3
            else:
                return 1.0
                
        elif bin_type == BinType.ORGANIC:
            # Meal time peaks
            meal_times = [7, 8, 12, 13, 18, 19, 20]
            if hour in meal_times:
                return 2.0
            elif 21 <= hour or hour <= 6:
                return 0.1
            else:
                return 0.8
                
        else:
            # Default pattern
            return 1.0
    
    async def _update_models(self):
        """Update ML models with new data"""
        if not XGBOOST_AVAILABLE:
            return
        
        # Check if we have enough data
        if len(self.historical_data) < self.settings.ml.MIN_TRAINING_SAMPLES:
            self.logger.debug("Insufficient training data", 
                            samples=len(self.historical_data))
            return
        
        # Check if enough time has passed since last training
        if (self.last_training and 
            (datetime.now() - self.last_training).total_seconds() < 
            self.settings.ml.MODEL_UPDATE_INTERVAL_MIN * 60):
            return
        
        try:
            await self._train_models()
            self.last_training = datetime.now()
            self.model_trained = True
            
            self.logger.info("Models updated", 
                           training_samples=len(self.historical_data))
            
        except Exception as e:
            self.logger.error("Failed to update models", error=str(e))
    
    async def _train_models(self):
        """Train XGBoost models on historical data"""
        
        # Convert historical data to DataFrame
        df = pd.DataFrame(self.historical_data)
        
        if df.empty:
            return
        
        # Prepare features and targets
        X = df[self.feature_columns].values
        y_fill = df['actual_fill_level'].values
        y_time = df['minutes_to_overflow'].values
        
        # Train fill level prediction model
        self.fill_model.fit(X, y_fill)
        
        # Train time to overflow model (only for samples that overflowed)
        overflow_mask = ~np.isinf(y_time)
        if np.sum(overflow_mask) > 10:  # Need enough overflow samples
            X_overflow = X[overflow_mask]
            y_time_overflow = y_time[overflow_mask]
            self.time_model.fit(X_overflow, y_time_overflow)
        
        self.logger.info("XGBoost models trained",
                        total_samples=len(df),
                        overflow_samples=np.sum(overflow_mask))
    
    def _is_prediction_cached(self, bin_id: str) -> bool:
        """Check if prediction is cached and valid"""
        if bin_id not in self.predictions_cache:
            return False
        
        if bin_id not in self.cache_expiry:
            return False
        
        return datetime.now() < self.cache_expiry[bin_id]
    
    def _cache_prediction(self, bin_id: str, prediction: Dict[str, float]):
        """Cache prediction with expiry"""
        self.predictions_cache[bin_id] = prediction
        self.cache_expiry[bin_id] = datetime.now() + timedelta(seconds=self.cache_ttl_sec)
    
    async def _clean_cache(self):
        """Remove expired cache entries"""
        now = datetime.now()
        expired_bins = [
            bin_id for bin_id, expiry in self.cache_expiry.items()
            if now >= expiry
        ]
        
        for bin_id in expired_bins:
            self.predictions_cache.pop(bin_id, None)
            self.cache_expiry.pop(bin_id, None)
        
        if expired_bins:
            self.logger.debug("Cleaned cache", expired_count=len(expired_bins))
    
    def _register_forecast_handlers(self):
        """Register forecast-specific message handlers"""
        self.register_handler("predict_overflow", self._handle_predict_overflow)
        self.register_handler("update_historical_data", self._handle_update_historical_data)
        self.register_handler("get_model_status", self._handle_get_model_status)
    
    async def _handle_predict_overflow(self, data: Dict[str, Any]):
        """Handle overflow prediction request"""
        try:
            # Parse bins from request
            bins_data = data.get("bins", [])
            bins = []
            
            for bin_data in bins_data:
                bin_obj = Bin(
                    id=bin_data["id"],
                    lat=bin_data["lat"],
                    lon=bin_data["lon"],
                    capacity_l=bin_data["capacity_l"],
                    fill_level=bin_data["fill_level"],
                    fill_rate_lph=bin_data["fill_rate_lph"],
                    tile_id=bin_data.get("tile_id", ""),
                    bin_type=BinType(bin_data.get("bin_type", "general"))
                )
                bins.append(bin_obj)
            
            # Generate predictions
            predictions = await self.predict_minutes_to_120(bins)
            
            # Send response
            await self.send_message(
                "overflow_predictions",
                {
                    "predictions": predictions,
                    "timestamp": datetime.now().isoformat(),
                    "correlation_id": data.get("correlation_id")
                }
            )
            
        except Exception as e:
            self.logger.error("Error handling prediction request", error=str(e))
            
            await self.send_message(
                "prediction_error",
                {
                    "error": str(e),
                    "correlation_id": data.get("correlation_id")
                }
            )
    
    async def _handle_update_historical_data(self, data: Dict[str, Any]):
        """Handle historical data update"""
        try:
            new_data = data.get("data", [])
            self.historical_data.extend(new_data)
            
            # Keep only recent data to prevent memory issues
            max_samples = 50000
            if len(self.historical_data) > max_samples:
                self.historical_data = self.historical_data[-max_samples:]
            
            self.logger.debug("Historical data updated", 
                            new_samples=len(new_data),
                            total_samples=len(self.historical_data))
            
        except Exception as e:
            self.logger.error("Error updating historical data", error=str(e))
    
    async def _handle_get_model_status(self, data: Dict[str, Any]):
        """Handle model status request"""
        status = {
            "xgboost_available": XGBOOST_AVAILABLE,
            "model_trained": self.model_trained,
            "last_training": self.last_training.isoformat() if self.last_training else None,
            "historical_samples": len(self.historical_data),
            "cached_predictions": len(self.predictions_cache),
            "correlation_id": data.get("correlation_id")
        }
        
        await self.send_message("model_status", status)