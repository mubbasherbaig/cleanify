"""
Cleanify v2-alpha LLM Advisor Agent
Optional CPU-only LLM for route choice suggestions using Phi-3-mini-4k-instruct
"""

import asyncio
import json
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .base import AgentBase
from core.models import LLMRecommendation
from core.settings import get_settings


class LLMAdvisorAgent(AgentBase):
    """
    LLM Advisor agent that provides intelligent route optimization suggestions
    Feature-gated and only loads if sufficient memory is available
    """
    
    def __init__(self):
        super().__init__("llm_advisor", "llm_advisor")
        
        # LLM state
        self.model = None
        self.tokenizer = None
        self.llm_pipeline = None
        self.model_loaded = False
        self.model_loading = False
        
        # Request queue
        self.pending_requests: Dict[str, Dict[str, Any]] = {}
        self.recommendation_history: List[LLMRecommendation] = []
        
        # Settings
        self.settings = get_settings()
        
        # Performance metrics
        self.recommendations_generated = 0
        self.total_inference_time = 0.0
        self.model_load_time = 0.0
        
        # Register handlers
        self._register_llm_handlers()
    
    async def initialize(self):
        """Initialize LLM advisor agent"""
        self.logger.info("Initializing LLM Advisor Agent")
        
        # Check if LLM is enabled and system meets requirements
        if not self.settings.llm.ENABLE_LLM_ADVISOR:
            self.logger.info("LLM Advisor disabled in settings")
            return
        
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers library not available")
            return
        
        # Check memory requirements
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        if total_memory_gb < self.settings.llm.MIN_MEMORY_GB:
            self.logger.warning("Insufficient memory for LLM",
                              available_gb=total_memory_gb,
                              required_gb=self.settings.llm.MIN_MEMORY_GB)
            return
        
        # Load model in background
        asyncio.create_task(self._load_llm_model())
        
        self.logger.info("LLM Advisor agent initialized")
    
    async def main_loop(self):
        """Main LLM advisor loop"""
        while self.running:
            try:
                # Process pending requests
                await self._process_pending_requests()
                
                # Clean up old recommendations
                await self._cleanup_old_recommendations()
                
                # Sleep briefly
                await asyncio.sleep(5.0)
                
            except Exception as e:
                self.logger.error("Error in LLM advisor main loop", error=str(e))
                await asyncio.sleep(30)
    
    async def cleanup(self):
        """Cleanup LLM advisor agent"""
        if self.model_loaded:
            # Clear model from memory
            del self.model
            del self.tokenizer
            del self.llm_pipeline
            
            # Force garbage collection
            if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.logger.info("LLM Advisor agent cleanup")
    
    async def suggest_route_choice(self, stats_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main LLM method: suggest route choice based on statistics
        """
        
        if not self.model_loaded:
            return {
                "status": "error",
                "message": "LLM model not loaded",
                "fallback_suggestion": "Use shortest distance routing"
            }
        
        try:
            start_time = datetime.now()
            
            # Prepare prompt for LLM
            prompt = self._create_route_choice_prompt(stats_json)
            
            # Generate response using LLM
            response = await self._generate_llm_response(prompt)
            
            # Parse and structure response
            recommendation = self._parse_llm_response(response, stats_json)
            
            # Calculate inference time
            inference_time = (datetime.now() - start_time).total_seconds()
            self.total_inference_time += inference_time
            self.recommendations_generated += 1
            
            # Store recommendation
            self.recommendation_history.append(recommendation)
            
            self.logger.debug("LLM recommendation generated",
                            inference_time=inference_time,
                            confidence=recommendation.confidence_score)
            
            return {
                "status": "success",
                "recommendation": {
                    "id": recommendation.recommendation_id,
                    "suggested_action": recommendation.suggested_action,
                    "confidence_score": recommendation.confidence_score,
                    "reasoning": recommendation.reasoning,
                    "alternatives": recommendation.alternative_options
                },
                "inference_time_sec": inference_time
            }
            
        except Exception as e:
            self.logger.error("Error generating LLM recommendation", error=str(e))
            
            return {
                "status": "error",
                "message": str(e),
                "fallback_suggestion": "Use heuristic-based routing"
            }
    
    async def _load_llm_model(self):
        """Load LLM model in background"""
        
        if self.model_loading or self.model_loaded:
            return
        
        self.model_loading = True
        start_time = datetime.now()
        
        try:
            self.logger.info("Loading LLM model", model=self.settings.llm.MODEL_NAME)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.settings.llm.MODEL_NAME,
                trust_remote_code=True
            )
            
            # Load model with CPU-only configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.llm.MODEL_NAME,
                device_map=self.settings.llm.DEVICE_MAP,
                load_in_4bit=self.settings.llm.LOAD_IN_4BIT,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Create pipeline
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.settings.llm.DEVICE_MAP
            )
            
            self.model_loaded = True
            self.model_load_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info("LLM model loaded successfully",
                           load_time=self.model_load_time,
                           model_size_gb=self._estimate_model_size())
            
        except Exception as e:
            self.logger.error("Failed to load LLM model", error=str(e))
            self.model_loaded = False
        finally:
            self.model_loading = False
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in GB"""
        if not self.model:
            return 0.0
        
        try:
            param_count = sum(p.numel() for p in self.model.parameters())
            # Rough estimate: 4 bytes per parameter for float32, 2 for float16
            bytes_per_param = 2 if self.settings.llm.LOAD_IN_4BIT else 4
            total_bytes = param_count * bytes_per_param
            return total_bytes / (1024**3)
        except:
            return 0.0
    
    def _create_route_choice_prompt(self, stats_json: Dict[str, Any]) -> str:
        """Create prompt for route choice recommendation"""
        
        # Extract key statistics
        truck_count = stats_json.get("truck_count", 0)
        bin_count = stats_json.get("bin_count", 0)
        urgent_bins = stats_json.get("urgent_bins", 0)
        avg_distance = stats_json.get("avg_route_distance_km", 0.0)
        traffic_level = stats_json.get("traffic_level", "unknown")
        system_load = stats_json.get("system_load_percent", 0.0)
        
        prompt = f"""<|system|>
You are an expert waste collection route optimization advisor. Analyze the given statistics and provide a concise recommendation for route planning strategy.

<|user|>
Current waste collection system status:
- Available trucks: {truck_count}
- Total bins: {bin_count}
- Urgent bins needing collection: {urgent_bins}
- Average route distance: {avg_distance:.1f} km
- Traffic conditions: {traffic_level}
- System load: {system_load:.1f}%

Based on these conditions, what routing strategy would you recommend? Consider factors like:
1. Efficiency vs urgency balance
2. Traffic impact on route timing
3. System capacity utilization
4. Risk of bin overflow

Provide a brief, actionable recommendation with reasoning.

<|assistant|>
"""
        
        return prompt
    
    async def _generate_llm_response(self, prompt: str) -> str:
        """Generate response using LLM pipeline"""
        
        if not self.llm_pipeline:
            raise RuntimeError("LLM pipeline not available")
        
        try:
            # Run LLM inference
            response = self.llm_pipeline(
                prompt,
                max_new_tokens=self.settings.llm.MAX_NEW_TOKENS,
                temperature=self.settings.llm.TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            full_text = response[0]["generated_text"]
            
            # Remove prompt from response
            if "<|assistant|>" in full_text:
                generated_text = full_text.split("<|assistant|>")[-1].strip()
            else:
                generated_text = full_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            self.logger.error("LLM inference failed", error=str(e))
            raise
    
    def _parse_llm_response(self, response: str, stats_json: Dict[str, Any]) -> LLMRecommendation:
        """Parse LLM response into structured recommendation"""
        
        # Simple parsing - in production would use more sophisticated NLP
        response_lower = response.lower()
        
        # Determine suggested action
        if "prioritize urgent" in response_lower or "urgent first" in response_lower:
            suggested_action = "prioritize_urgent_bins"
        elif "shortest" in response_lower and "distance" in response_lower:
            suggested_action = "optimize_for_distance"
        elif "traffic" in response_lower and ("avoid" in response_lower or "delay" in response_lower):
            suggested_action = "traffic_aware_routing"
        elif "balance" in response_lower or "hybrid" in response_lower:
            suggested_action = "balanced_optimization"
        elif "capacity" in response_lower and "limit" in response_lower:
            suggested_action = "capacity_constrained_routing"
        else:
            suggested_action = "default_optimization"
        
        # Calculate confidence based on response characteristics
        confidence_score = self._calculate_confidence(response)
        
        # Generate alternatives
        alternatives = self._generate_alternatives(suggested_action)
        
        recommendation = LLMRecommendation(
            recommendation_id=f"llm_{int(datetime.now().timestamp())}",
            route_stats=stats_json,
            suggested_action=suggested_action,
            confidence_score=confidence_score,
            reasoning=response[:200] + "..." if len(response) > 200 else response,
            alternative_options=alternatives
        )
        
        return recommendation
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score based on response characteristics"""
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence for specific keywords
        high_confidence_words = ["recommend", "should", "optimal", "best", "prioritize"]
        for word in high_confidence_words:
            if word in response.lower():
                confidence += 0.1
        
        # Increase confidence for reasoning indicators
        reasoning_indicators = ["because", "due to", "since", "therefore", "as a result"]
        for indicator in reasoning_indicators:
            if indicator in response.lower():
                confidence += 0.1
        
        # Decrease confidence for uncertainty indicators
        uncertainty_words = ["maybe", "possibly", "might", "could", "uncertain"]
        for word in uncertainty_words:
            if word in response.lower():
                confidence -= 0.1
        
        # Clamp between 0.1 and 1.0
        return max(0.1, min(1.0, confidence))
    
    def _generate_alternatives(self, suggested_action: str) -> List[str]:
        """Generate alternative actions"""
        
        all_actions = [
            "prioritize_urgent_bins",
            "optimize_for_distance", 
            "traffic_aware_routing",
            "balanced_optimization",
            "capacity_constrained_routing",
            "time_window_optimization"
        ]
        
        # Return 2-3 alternatives excluding the suggested action
        alternatives = [action for action in all_actions if action != suggested_action]
        return alternatives[:3]
    
    async def _process_pending_requests(self):
        """Process pending LLM requests"""
        
        if not self.model_loaded:
            return
        
        completed_requests = []
        
        for request_id, request_data in self.pending_requests.items():
            try:
                stats_json = request_data.get("stats_json", {})
                
                # Generate recommendation
                recommendation_result = await self.suggest_route_choice(stats_json)
                
                # Send response
                await self.send_message(
                    "llm_recommendation",
                    {
                        "request_id": request_id,
                        "recommendation": recommendation_result,
                        "correlation_id": request_data.get("correlation_id")
                    }
                )
                
                completed_requests.append(request_id)
                
            except Exception as e:
                self.logger.error("Error processing LLM request", 
                                request_id=request_id, error=str(e))
                completed_requests.append(request_id)
        
        # Remove completed requests
        for request_id in completed_requests:
            self.pending_requests.pop(request_id, None)
    
    async def _cleanup_old_recommendations(self):
        """Clean up old recommendations"""
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=2)
        
        self.recommendation_history = [
            rec for rec in self.recommendation_history
            if rec.timestamp > cutoff_time
        ]
    
    def _register_llm_handlers(self):
        """Register LLM-specific message handlers"""
        self.register_handler("request_recommendation", self._handle_request_recommendation)
        self.register_handler("get_llm_status", self._handle_get_llm_status)
        self.register_handler("reload_model", self._handle_reload_model)
        self.register_handler("get_recommendations_history", self._handle_get_recommendations_history)
    
    async def _handle_request_recommendation(self, data: Dict[str, Any]):
        """Handle recommendation request"""
        try:
            if not self.model_loaded:
                await self.send_message(
                    "llm_recommendation_error",
                    {
                        "error": "LLM model not loaded",
                        "correlation_id": data.get("correlation_id")
                    }
                )
                return
            
            stats_json = data.get("stats_json", {})
            
            # Generate recommendation directly
            recommendation_result = await self.suggest_route_choice(stats_json)
            
            await self.send_message(
                "llm_recommendation",
                {
                    "recommendation": recommendation_result,
                    "correlation_id": data.get("correlation_id")
                }
            )
            
        except Exception as e:
            self.logger.error("Error handling recommendation request", error=str(e))
            
            await self.send_message(
                "llm_recommendation_error",
                {
                    "error": str(e),
                    "correlation_id": data.get("correlation_id")
                }
            )
    
    async def _handle_get_llm_status(self, data: Dict[str, Any]):
        """Handle LLM status request"""
        
        status = {
            "model_loaded": self.model_loaded,
            "model_loading": self.model_loading,
            "model_name": self.settings.llm.MODEL_NAME,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "device_map": self.settings.llm.DEVICE_MAP,
            "load_in_4bit": self.settings.llm.LOAD_IN_4BIT,
            "model_load_time_sec": self.model_load_time,
            "estimated_model_size_gb": self._estimate_model_size(),
            "recommendations_generated": self.recommendations_generated,
            "avg_inference_time_sec": (
                self.total_inference_time / max(1, self.recommendations_generated)
            ),
            "pending_requests": len(self.pending_requests),
            "correlation_id": data.get("correlation_id")
        }
        
        await self.send_message("llm_status", status)
    
    async def _handle_reload_model(self, data: Dict[str, Any]):
        """Handle model reload request"""
        try:
            if self.model_loaded:
                # Cleanup existing model
                await self.cleanup()
                self.model_loaded = False
            
            # Reload model
            await self._load_llm_model()
            
            await self.send_message(
                "model_reload_complete",
                {
                    "success": self.model_loaded,
                    "load_time_sec": self.model_load_time,
                    "correlation_id": data.get("correlation_id")
                }
            )
            
        except Exception as e:
            self.logger.error("Error reloading model", error=str(e))
            
            await self.send_message(
                "model_reload_error",
                {
                    "error": str(e),
                    "correlation_id": data.get("correlation_id")
                }
            )
    
    async def _handle_get_recommendations_history(self, data: Dict[str, Any]):
        """Handle recommendations history request"""
        
        limit = data.get("limit", 10)
        recent_recommendations = self.recommendation_history[-limit:]
        
        history_data = []
        for rec in recent_recommendations:
            history_data.append({
                "recommendation_id": rec.recommendation_id,
                "suggested_action": rec.suggested_action,
                "confidence_score": rec.confidence_score,
                "reasoning": rec.reasoning,
                "timestamp": rec.timestamp.isoformat(),
                "alternatives": rec.alternative_options
            })
        
        await self.send_message(
            "recommendations_history",
            {
                "recommendations": history_data,
                "total_in_history": len(self.recommendation_history),
                "correlation_id": data.get("correlation_id")
            }
        )
    
    def get_llm_metrics(self) -> Dict[str, Any]:
        """Get LLM advisor performance metrics"""
        
        return {
            "model_status": {
                "loaded": self.model_loaded,
                "loading": self.model_loading,
                "model_name": self.settings.llm.MODEL_NAME,
                "estimated_size_gb": self._estimate_model_size()
            },
            "performance": {
                "recommendations_generated": self.recommendations_generated,
                "total_inference_time_sec": self.total_inference_time,
                "avg_inference_time_sec": (
                    self.total_inference_time / max(1, self.recommendations_generated)
                ),
                "model_load_time_sec": self.model_load_time
            },
            "queue": {
                "pending_requests": len(self.pending_requests),
                "recommendations_in_history": len(self.recommendation_history)
            },
            "capabilities": {
                "transformers_available": TRANSFORMERS_AVAILABLE,
                "enabled_in_settings": self.settings.llm.ENABLE_LLM_ADVISOR,
                "memory_sufficient": (
                    psutil.virtual_memory().total / (1024**3) >= 
                    self.settings.llm.MIN_MEMORY_GB
                )
            }
        }