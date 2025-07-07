# """
# Cleanify v2-alpha LLM Advisor Agent
# Uses OpenAI GPT API for route choice suggestions
# """

# import asyncio
# import json
# import os
# from datetime import datetime
# from typing import Dict, List, Any, Optional
# import warnings

# # Suppress warnings for cleaner logs
# warnings.filterwarnings("ignore")

# try:
#     import openai
#     OPENAI_AVAILABLE = True
# except ImportError:
#     OPENAI_AVAILABLE = False

# from .base import AgentBase
# from core.models import LLMRecommendation
# from core.settings import get_settings


# class LLMAdvisorAgent(AgentBase):
#     """
#     LLM Advisor agent that provides intelligent route optimization suggestions using OpenAI GPT
#     """
    
#     def __init__(self):
#         super().__init__("llm_advisor", "llm_advisor")
        
#         # OpenAI client
#         self.openai_client = None
#         self.model_loaded = False
        
#         # Request queue
#         self.pending_requests: Dict[str, Dict[str, Any]] = {}
#         self.recommendation_history: List[LLMRecommendation] = []
        
#         # Settings
#         self.settings = get_settings()
        
#         # Performance metrics
#         self.recommendations_generated = 0
#         self.total_inference_time = 0.0
#         self.api_errors = 0
        
#         # Register handlers
#         self._register_llm_handlers()
    
#     async def initialize(self):
#         """Initialize LLM advisor agent"""
#         self.logger.info("Initializing LLM Advisor Agent")
        
#         # Check if LLM is enabled
#         if not self.settings.llm.ENABLE_LLM_ADVISOR:
#             self.logger.info("LLM Advisor disabled in settings")
#             return
        
#         if not OPENAI_AVAILABLE:
#             self.logger.warning("OpenAI library not available")
#             return
        
#         # Initialize OpenAI client
#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             self.logger.warning("OPENAI_API_KEY not found in environment")
#             return
        
#         try:
#             openai.api_key = "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"
#             self.openai_client = openai
#             self.model_loaded = True
            
#             self.logger.info("OpenAI client initialized successfully")
            
#         except Exception as e:
#             self.logger.error("Failed to initialize OpenAI client", error=str(e))
        
#         self.logger.info("LLM Advisor agent initialized")
    
#     async def main_loop(self):
#         """Main LLM advisor loop"""
#         while self.running:
#             try:
#                 # Process pending requests
#                 await self._process_pending_requests()
                
#                 # Clean up old recommendations
#                 await self._cleanup_old_recommendations()
                
#                 # Sleep briefly
#                 await asyncio.sleep(5.0)
                
#             except Exception as e:
#                 self.logger.error("Error in LLM advisor main loop", error=str(e))
#                 await asyncio.sleep(30)
    
#     async def cleanup(self):
#         """Cleanup LLM advisor agent"""
#         self.logger.info("LLM Advisor agent cleanup")
    
#     async def suggest_route_choice(self, stats_json: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Main LLM method: suggest route choice based on statistics using OpenAI GPT
#         """
        
#         if not self.model_loaded:
#             return {
#                 "status": "error",
#                 "message": "OpenAI client not initialized",
#                 "fallback_suggestion": "Use shortest distance routing"
#             }
        
#         try:
#             start_time = datetime.now()
            
#             # Prepare prompt for GPT
#             prompt = self._create_route_choice_prompt(stats_json)
            
#             # Generate response using OpenAI API
#             response = await self._generate_gpt_response(prompt)
            
#             # Parse and structure response
#             recommendation = self._parse_llm_response(response, stats_json)
            
#             # Calculate inference time
#             inference_time = (datetime.now() - start_time).total_seconds()
#             self.total_inference_time += inference_time
#             self.recommendations_generated += 1
            
#             # Store recommendation
#             self.recommendation_history.append(recommendation)
            
#             self.logger.debug("LLM recommendation generated",
#                             inference_time=inference_time,
#                             confidence=recommendation.confidence_score)
            
#             return {
#                 "status": "success",
#                 "recommendation": {
#                     "id": recommendation.recommendation_id,
#                     "suggested_action": recommendation.suggested_action,
#                     "confidence_score": recommendation.confidence_score,
#                     "reasoning": recommendation.reasoning,
#                     "alternatives": recommendation.alternative_options
#                 },
#                 "inference_time_sec": inference_time
#             }
            
#         except Exception as e:
#             self.api_errors += 1
#             self.logger.error("Error generating LLM recommendation", error=str(e))
            
#             return {
#                 "status": "error",
#                 "message": str(e),
#                 "fallback_suggestion": "Use heuristic-based routing"
#             }
    
#     def _create_route_choice_prompt(self, stats_json: Dict[str, Any]) -> str:
#         """Create prompt for route choice recommendation"""
        
#         # Extract key statistics
#         truck_count = stats_json.get("truck_count", 0)
#         bin_count = stats_json.get("bin_count", 0)
#         urgent_bins = stats_json.get("urgent_bins", 0)
#         avg_distance = stats_json.get("avg_route_distance_km", 0.0)
#         traffic_level = stats_json.get("traffic_level", "unknown")
#         system_load = stats_json.get("system_load_percent", 0.0)
        
#         prompt = f"""You are an expert waste collection route optimization advisor. Analyze the given statistics and provide a concise recommendation for route planning strategy.

# Current waste collection system status:
# - Available trucks: {truck_count}
# - Total bins: {bin_count}
# - Urgent bins needing collection: {urgent_bins}
# - Average route distance: {avg_distance:.1f} km
# - Traffic conditions: {traffic_level}
# - System load: {system_load:.1f}%

# Based on these conditions, what routing strategy would you recommend? Consider factors like:
# 1. Efficiency vs urgency balance
# 2. Traffic impact on route timing
# 3. System capacity utilization
# 4. Risk of bin overflow

# Provide a brief, actionable recommendation with reasoning. Keep your response focused and under 200 words."""
        
#         return prompt
    
#     async def _generate_gpt_response(self, prompt: str) -> str:
#         """Generate response using OpenAI GPT API"""
        
#         if not self.openai_client:
#             raise RuntimeError("OpenAI client not available")
        
#         try:
#             response = await asyncio.to_thread(
#                 self.openai_client.ChatCompletion.create,
#                 model=self.settings.llm.MODEL_NAME or "gpt-3.5-turbo",
#                 messages=[
#                     {"role": "system", "content": "You are an expert waste collection optimization advisor."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 max_tokens=self.settings.llm.MAX_NEW_TOKENS or 300,
#                 temperature=self.settings.llm.TEMPERATURE or 0.7,
#                 timeout=self.settings.llm.REQUEST_TIMEOUT_SEC or 30
#             )
            
#             return response.choices[0].message.content.strip()
            
#         except Exception as e:
#             self.logger.error("OpenAI API call failed", error=str(e))
#             raise
    
#     def _parse_llm_response(self, response: str, stats_json: Dict[str, Any]) -> LLMRecommendation:
#         """Parse LLM response into structured recommendation"""
        
#         # Simple parsing - in production would use more sophisticated NLP
#         response_lower = response.lower()
        
#         # Determine suggested action
#         if "prioritize urgent" in response_lower or "urgent first" in response_lower:
#             suggested_action = "prioritize_urgent_bins"
#         elif "shortest" in response_lower and "distance" in response_lower:
#             suggested_action = "optimize_for_distance"
#         elif "traffic" in response_lower and ("avoid" in response_lower or "delay" in response_lower):
#             suggested_action = "traffic_aware_routing"
#         elif "balance" in response_lower or "hybrid" in response_lower:
#             suggested_action = "balanced_optimization"
#         elif "capacity" in response_lower and "limit" in response_lower:
#             suggested_action = "capacity_constrained_routing"
#         else:
#             suggested_action = "default_optimization"
        
#         # Calculate confidence based on response characteristics
#         confidence_score = self._calculate_confidence(response)
        
#         # Generate alternatives
#         alternatives = self._generate_alternatives(suggested_action)
        
#         recommendation = LLMRecommendation(
#             recommendation_id=f"llm_{int(datetime.now().timestamp())}",
#             route_stats=stats_json,
#             suggested_action=suggested_action,
#             confidence_score=confidence_score,
#             reasoning=response[:200] + "..." if len(response) > 200 else response,
#             alternative_options=alternatives
#         )
        
#         return recommendation
    
#     def _calculate_confidence(self, response: str) -> float:
#         """Calculate confidence score based on response characteristics"""
        
#         confidence = 0.5  # Base confidence
        
#         # Increase confidence for specific keywords
#         high_confidence_words = ["recommend", "should", "optimal", "best", "prioritize"]
#         for word in high_confidence_words:
#             if word in response.lower():
#                 confidence += 0.1
        
#         # Increase confidence for reasoning indicators
#         reasoning_indicators = ["because", "due to", "since", "therefore", "as a result"]
#         for indicator in reasoning_indicators:
#             if indicator in response.lower():
#                 confidence += 0.1
        
#         # Decrease confidence for uncertainty indicators
#         uncertainty_words = ["maybe", "possibly", "might", "could", "uncertain"]
#         for word in uncertainty_words:
#             if word in response.lower():
#                 confidence -= 0.1
        
#         # Clamp between 0.1 and 1.0
#         return max(0.1, min(1.0, confidence))
    
#     def _generate_alternatives(self, suggested_action: str) -> List[str]:
#         """Generate alternative actions"""
        
#         all_actions = [
#             "prioritize_urgent_bins",
#             "optimize_for_distance", 
#             "traffic_aware_routing",
#             "balanced_optimization",
#             "capacity_constrained_routing",
#             "time_window_optimization"
#         ]
        
#         # Return 2-3 alternatives excluding the suggested action
#         alternatives = [action for action in all_actions if action != suggested_action]
#         return alternatives[:3]
    
#     async def _process_pending_requests(self):
#         """Process pending LLM requests"""
        
#         if not self.model_loaded:
#             return
        
#         completed_requests = []
        
#         for request_id, request_data in self.pending_requests.items():
#             try:
#                 stats_json = request_data.get("stats_json", {})
                
#                 # Generate recommendation
#                 recommendation_result = await self.suggest_route_choice(stats_json)
                
#                 # Send response
#                 await self.send_message(
#                     "llm_recommendation",
#                     {
#                         "request_id": request_id,
#                         "recommendation": recommendation_result,
#                         "correlation_id": request_data.get("correlation_id")
#                     }
#                 )
                
#                 completed_requests.append(request_id)
                
#             except Exception as e:
#                 self.logger.error("Error processing LLM request", 
#                                 request_id=request_id, error=str(e))
#                 completed_requests.append(request_id)
        
#         # Remove completed requests
#         for request_id in completed_requests:
#             self.pending_requests.pop(request_id, None)
    
#     async def _cleanup_old_recommendations(self):
#         """Clean up old recommendations"""
#         from datetime import timedelta
        
#         cutoff_time = datetime.now() - timedelta(hours=2)
        
#         self.recommendation_history = [
#             rec for rec in self.recommendation_history
#             if rec.timestamp > cutoff_time
#         ]
    
#     def _register_llm_handlers(self):
#         """Register LLM-specific message handlers"""
#         self.register_handler("request_recommendation", self._handle_request_recommendation)
#         self.register_handler("get_llm_status", self._handle_get_llm_status)
#         self.register_handler("get_recommendations_history", self._handle_get_recommendations_history)
    
#     async def _handle_request_recommendation(self, data: Dict[str, Any]):
#         """Handle recommendation request"""
#         try:
#             if not self.model_loaded:
#                 await self.send_message(
#                     "llm_recommendation_error",
#                     {
#                         "error": "OpenAI client not initialized",
#                         "correlation_id": data.get("correlation_id")
#                     }
#                 )
#                 return
            
#             stats_json = data.get("stats_json", {})
            
#             # Generate recommendation directly
#             recommendation_result = await self.suggest_route_choice(stats_json)
            
#             await self.send_message(
#                 "llm_recommendation",
#                 {
#                     "recommendation": recommendation_result,
#                     "correlation_id": data.get("correlation_id")
#                 }
#             )
            
#         except Exception as e:
#             self.logger.error("Error handling recommendation request", error=str(e))
            
#             await self.send_message(
#                 "llm_recommendation_error",
#                 {
#                     "error": str(e),
#                     "correlation_id": data.get("correlation_id")
#                 }
#             )
    
#     async def _handle_get_llm_status(self, data: Dict[str, Any]):
#         """Handle LLM status request"""
        
#         status = {
#             "model_loaded": self.model_loaded,
#             "model_name": self.settings.llm.MODEL_NAME or "gpt-3.5-turbo",
#             "openai_available": OPENAI_AVAILABLE,
#             "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
#             "recommendations_generated": self.recommendations_generated,
#             "api_errors": self.api_errors,
#             "avg_inference_time_sec": (
#                 self.total_inference_time / max(1, self.recommendations_generated)
#             ),
#             "pending_requests": len(self.pending_requests),
#             "correlation_id": data.get("correlation_id")
#         }
        
#         await self.send_message("llm_status", status)
    
#     async def _handle_get_recommendations_history(self, data: Dict[str, Any]):
#         """Handle recommendations history request"""
        
#         limit = data.get("limit", 10)
#         recent_recommendations = self.recommendation_history[-limit:]
        
#         history_data = []
#         for rec in recent_recommendations:
#             history_data.append({
#                 "recommendation_id": rec.recommendation_id,
#                 "suggested_action": rec.suggested_action,
#                 "confidence_score": rec.confidence_score,
#                 "reasoning": rec.reasoning,
#                 "timestamp": rec.timestamp.isoformat(),
#                 "alternatives": rec.alternative_options
#             })
        
#         await self.send_message(
#             "recommendations_history",
#             {
#                 "recommendations": history_data,
#                 "total_in_history": len(self.recommendation_history),
#                 "correlation_id": data.get("correlation_id")
#             }
#         )
    
#     def get_llm_metrics(self) -> Dict[str, Any]:
#         """Get LLM advisor performance metrics"""
        
#         return {
#             "model_status": {
#                 "loaded": self.model_loaded,
#                 "model_name": self.settings.llm.MODEL_NAME or "gpt-3.5-turbo",
#                 "api_key_configured": bool(os.getenv("OPENAI_API_KEY"))
#             },
#             "performance": {
#                 "recommendations_generated": self.recommendations_generated,
#                 "api_errors": self.api_errors,
#                 "total_inference_time_sec": self.total_inference_time,
#                 "avg_inference_time_sec": (
#                     self.total_inference_time / max(1, self.recommendations_generated)
#                 )
#             },
#             "queue": {
#                 "pending_requests": len(self.pending_requests),
#                 "recommendations_in_history": len(self.recommendation_history)
#             },
#             "capabilities": {
#                 "openai_available": OPENAI_AVAILABLE,
#                 "enabled_in_settings": self.settings.llm.ENABLE_LLM_ADVISOR
#             }
#         }



"""
llm_advisor.py
────────────────────────────────────────────────────────────────────────────
Ultra-light, safe GPT helper for Cleanify.

Exposed helper:
    >>> extras = await ask_llm(payload_dict)

`payload_dict` must be:
{
  "truck_capacity_left": <int>,
  "candidates": [
      {"id": <int>, "proj_fill": <float>, "detour_m": <int>},
      ...
  ]
}

Returns:
    • list[int]  → extra bin IDs the truck should also collect
    • None       → on error / disabled / rate-limited / invalid JSON
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import hashlib
from typing import Any, Dict, List, Optional

import openai

from core.settings import Settings

_cfg = Settings().llm

# ── API key (never hard-code) ────────────────────────────────────────────
openai.api_key = "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"
if not openai.api_key and _cfg.ENABLE_LLM_ADVISOR:
    raise RuntimeError("OPENAI_API_KEY env variable not set")

# ── Global rate-limit (MAX_CALLS_PER_HOUR across the whole process) ─────
_RATE_WINDOW_SEC: float = 3600.0 / max(1, _cfg.MAX_CALLS_PER_HOUR)
_last_call_ts: float = 0.0

async def _rate_limit() -> None:
    global _last_call_ts
    wait = _last_call_ts + _RATE_WINDOW_SEC - time.time()
    if wait > 0:
        await asyncio.sleep(wait)
    _last_call_ts = time.time()

# ── Simple in-memory cache (keyed by payload hash) ───────────────────────
_cache: Dict[str, Optional[List[int]]] = {}

def _hash_payload(payload: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()

# ── Public helper ────────────────────────────────────────────────────────
async def ask_llm(payload: Dict[str, Any]) -> Optional[List[int]]:
    """
    Query GPT for extra bins. Cheap, cached, validated.
    """
    if not _cfg.ENABLE_LLM_ADVISOR:
        return None

    key = _hash_payload(payload)
    if key in _cache:
        return _cache[key]                      # ≤ milliseconds

    await _rate_limit()                        # global throttle

    system_prompt = (
        "You are a waste-collection micro-advisor.\n"
        'Return ONLY valid JSON exactly like {"extra_bins":[1,2]}.\n'
        "No commentary, no other keys."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": json.dumps(payload, separators=(",", ":"))},
    ]

    try:
        rsp = await openai.ChatCompletion.acreate(
            model          = _cfg.MODEL_NAME,
            messages       = messages,
            response_format= {"type": "json_object"},
            max_tokens     = _cfg.MAX_NEW_TOKENS,
            temperature    = _cfg.TEMPERATURE,
            timeout        = 20,
        )
        data = json.loads(rsp.choices[0].message.content)
        extras = data.get("extra_bins")
        if isinstance(extras, list) and all(isinstance(i, int) for i in extras):
            _cache[key] = extras
            return extras
    except Exception as exc:
        print(f"[LLM] call failed → {exc}")

    _cache[key] = None                          # avoid retrying same payload
    return None

# ── Export list ──────────────────────────────────────────────────────────
__all__ = ["ask_llm"]
