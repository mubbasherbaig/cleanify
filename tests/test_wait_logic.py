# tests/test_wait_logic.py
"""
Test waiting-time decision logic for edge cases: 0, 3, 6 minute traffic delays
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.departure import DepartureAgent
from core.models import Truck, Bin, TruckStatus
from core.settings import settings


class TestWaitLogic:
    """Test waiting-time recipe edge cases"""
    
    @pytest.fixture
    def departure_agent(self):
        """Create DepartureAgent for testing"""
        agent = DepartureAgent()
        return agent
    
    @pytest.fixture
    def mock_truck(self):
        """Mock truck for testing"""
        return Truck(
            id="truck_1",
            name="Test Truck",
            capacity_l=5000,
            lat=40.7589,
            lon=-73.9851,
            current_load_l=0,
            status=TruckStatus.IDLE
        )
    
    @pytest.fixture
    def bin_eta_map_scenarios(self):
        """Different bin ETA scenarios for testing"""
        return {
            "tight_schedule": {"bin_1": 2.0, "bin_2": 1.5, "bin_3": 3.0},  # Tmin = 1.5
            "moderate_schedule": {"bin_1": 8.0, "bin_2": 5.0, "bin_3": 7.0},  # Tmin = 5.0
            "loose_schedule": {"bin_1": 15.0, "bin_2": 12.0, "bin_3": 18.0},  # Tmin = 12.0
        }
    
    @pytest.mark.asyncio
    async def test_zero_traffic_delay(self, departure_agent, mock_truck, bin_eta_map_scenarios):
        """Test decision with 0 minutes traffic delay"""
        # Test with moderate schedule (Tmin = 5.0)
        bin_eta_map = bin_eta_map_scenarios["moderate_schedule"]
        delta_traffic = 0
        
        decision = await departure_agent.evaluate_wait(mock_truck, bin_eta_map, delta_traffic)
        
        # Tmin = 5.0, allowed_wait = 5.0 - 1 = 4.0
        # delta_traffic = 0 <= 4.0, so should wait
        assert decision == "WAIT_0_MIN"
    
    @pytest.mark.asyncio
    async def test_three_minute_delay_go_case(self, departure_agent, mock_truck, bin_eta_map_scenarios):
        """Test 3-minute delay forces GO_NOW with tight schedule"""
        # Test with tight schedule (Tmin = 1.5)
        bin_eta_map = bin_eta_map_scenarios["tight_schedule"]
        delta_traffic = 3
        
        decision = await departure_agent.evaluate_wait(mock_truck, bin_eta_map, delta_traffic)
        
        # Tmin = 1.5, allowed_wait = 1.5 - 1 = 0.5
        # delta_traffic = 3 > 0.5, so should GO_NOW
        assert decision == "GO_NOW"
    
    @pytest.mark.asyncio
    async def test_three_minute_delay_wait_case(self, departure_agent, mock_truck, bin_eta_map_scenarios):
        """Test 3-minute delay allows waiting with loose schedule"""
        # Test with loose schedule (Tmin = 12.0)
        bin_eta_map = bin_eta_map_scenarios["loose_schedule"]
        delta_traffic = 3
        
        decision = await departure_agent.evaluate_wait(mock_truck, bin_eta_map, delta_traffic)
        
        # Tmin = 12.0, allowed_wait = 12.0 - 1 = 11.0
        # delta_traffic = 3 <= 11.0, so should wait
        assert decision == "WAIT_3_MIN"
    
    @pytest.mark.asyncio
    async def test_six_minute_delay_boundary(self, departure_agent, mock_truck, bin_eta_map_scenarios):
        """Test 6-minute delay at boundary condition"""
        # Test with moderate schedule (Tmin = 5.0)
        bin_eta_map = bin_eta_map_scenarios["moderate_schedule"]
        delta_traffic = 6
        
        decision = await departure_agent.evaluate_wait(mock_truck, bin_eta_map, delta_traffic)
        
        # Tmin = 5.0, allowed_wait = 5.0 - 1 = 4.0
        # delta_traffic = 6 > 4.0, so should GO_NOW
        assert decision == "GO_NOW"
    
    @pytest.mark.asyncio
    async def test_six_minute_delay_within_limit(self, departure_agent, mock_truck, bin_eta_map_scenarios):
        """Test 6-minute delay within allowed wait time"""
        # Test with loose schedule (Tmin = 12.0)
        bin_eta_map = bin_eta_map_scenarios["loose_schedule"]
        delta_traffic = 6
        
        decision = await departure_agent.evaluate_wait(mock_truck, bin_eta_map, delta_traffic)
        
        # Tmin = 12.0, allowed_wait = 12.0 - 1 = 11.0
        # delta_traffic = 6 <= 11.0, so should wait
        assert decision == "WAIT_6_MIN"
    
    @pytest.mark.asyncio
    async def test_negative_allowed_wait(self, departure_agent, mock_truck):
        """Test edge case where allowed_wait <= 0"""
        # Very tight schedule where Tmin <= SAFETY_PAD_MIN
        bin_eta_map = {"bin_1": 0.5, "bin_2": 1.0, "bin_3": 0.8}  # Tmin = 0.5
        delta_traffic = 0
        
        decision = await departure_agent.evaluate_wait(mock_truck, bin_eta_map, delta_traffic)
        
        # Tmin = 0.5, allowed_wait = 0.5 - 1 = -0.5 <= 0, so should GO_NOW
        assert decision == "GO_NOW"
    
    @pytest.mark.asyncio
    async def test_exact_boundary_condition(self, departure_agent, mock_truck):
        """Test exact boundary where delta_traffic equals allowed_wait"""
        # Create scenario where allowed_wait exactly equals delta_traffic
        bin_eta_map = {"bin_1": 6.0, "bin_2": 5.0, "bin_3": 7.0}  # Tmin = 5.0
        delta_traffic = 4  # allowed_wait = 5.0 - 1 = 4.0
        
        decision = await departure_agent.evaluate_wait(mock_truck, bin_eta_map, delta_traffic)
        
        # delta_traffic = 4 = allowed_wait = 4, so should wait (not > allowed_wait)
        assert decision == "WAIT_4_MIN"
    
    @pytest.mark.asyncio
    async def test_empty_bin_eta_map(self, departure_agent, mock_truck):
        """Test edge case with empty bin ETA map"""
        bin_eta_map = {}
        delta_traffic = 3
        
        # Should handle gracefully - likely go now if no bins to wait for
        decision = await departure_agent.evaluate_wait(mock_truck, bin_eta_map, delta_traffic)
        
        # With empty bin_eta_map, min() would fail, expect GO_NOW
        assert decision == "GO_NOW"