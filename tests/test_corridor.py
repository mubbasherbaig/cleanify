# tests/test_corridor.py
"""
Test corridor logic: synthetic straight road, bins at 100m, 300m, 700m distances
Expect inclusion: ✔ first two by tube & way-ID, ✖ last unless detour ≤ 0.3 km
"""
import pytest
from unittest.mock import AsyncMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.corridor import CorridorAgent
from core.models import Bin
from core.settings import settings


class TestCorridorLogic:
    """Test two-layer corridor filtering logic"""
    
    @pytest.fixture
    def corridor_agent(self):
        """Create CorridorAgent for testing"""
        return CorridorAgent()
    
    @pytest.fixture
    def straight_road_polyline(self):
        """Synthetic straight road polyline (lat, lon tuples)"""
        # Create a straight north-south road, 1km long
        start_lat, start_lon = 40.7589, -73.9851
        end_lat = start_lat + 0.009  # ~1km north
        
        # Generate polyline with points every ~100m
        polyline = []
        for i in range(11):  # 0 to 10, giving 11 points
            progress = i / 10.0
            lat = start_lat + (end_lat - start_lat) * progress
            polyline.append((lat, start_lon))
        
        return polyline
    
    @pytest.fixture
    def test_bins(self):
        """Create bins at 100m, 300m, 700m from road start"""
        base_lat, base_lon = 40.7589, -73.9851
        
        bins = [
            Bin(
                id="bin_100m",
                lat=base_lat + 0.0009,  # ~100m north
                lon=base_lon + 0.0001,  # slight offset east
                capacity_l=240,
                fill_level=85.0,
                fill_rate_lph=2.5,
                tile_id="891f0d2834fffff",
                way_id=12345,  # matches route
                snap_offset_m=100.0
            ),
            Bin(
                id="bin_300m",
                lat=base_lat + 0.0027,  # ~300m north
                lon=base_lon - 0.0001,  # slight offset west
                capacity_l=240,
                fill_level=92.0,
                fill_rate_lph=3.0,
                tile_id="891f0d2834fffff",
                way_id=12345,  # matches route
                snap_offset_m=300.0
            ),
            Bin(
                id="bin_700m",
                lat=base_lat + 0.0063,  # ~700m north
                lon=base_lon + 0.0002,  # slight offset east
                capacity_l=240,
                fill_level=78.0,
                fill_rate_lph=1.8,
                tile_id="891f0d2834fffff",
                way_id=54321,  # different way_id
                snap_offset_m=700.0
            )
        ]
        return bins
    
    @pytest.fixture
    def route_way_ids(self):
        """Way IDs that are part of the route"""
        return [12345, 67890, 11111]  # bin_700m's way_id (54321) not included
    
    @pytest.fixture
    def route_offset_map(self):
        """Mock route offset mapping"""
        return {
            12345: 200.0,  # bins at 100m and 300m should match within threshold
            67890: 500.0,
            11111: 800.0,
            54321: 700.0
        }
    
    @pytest.mark.asyncio
    async def test_tube_filter_inclusion(self, corridor_agent, straight_road_polyline, test_bins):
        """Test that bins within corridor tube are included"""
        with patch.object(corridor_agent, 'tube_filter') as mock_tube:
            # Mock tube filter to include first two bins (within 250m of polyline)
            mock_tube.return_value = {"bin_100m", "bin_300m"}
            
            # Mock other methods to return empty for isolation
            with patch.object(corridor_agent, 'crowflight_candidates', return_value=set()):
                result = await corridor_agent.build_corridor(
                    straight_road_polyline, 
                    [12345, 67890]
                )
            
            # Should include bins from tube filter
            assert "bin_100m" in result
            assert "bin_300m" in result
            mock_tube.assert_called_once_with(straight_road_polyline, half_m=settings.CORRIDOR_HALF_M)
    
    @pytest.mark.asyncio
    async def test_edge_match_inclusion(self, corridor_agent, straight_road_polyline, test_bins, route_way_ids, route_offset_map):
        """Test edge-matched bins are included when way_id matches and offset within threshold"""
        with patch.object(corridor_agent, 'tube_filter', return_value=set()):
            with patch.object(corridor_agent, 'crowflight_candidates', return_value=set(test_bins)):
                with patch.object(corridor_agent, 'detour_km', return_value=0.5):  # > threshold
                    
                    # Mock the route_offset_m lookup
                    corridor_agent.route_offset_m = route_offset_map
                    
                    result = await corridor_agent.build_corridor(
                        straight_road_polyline,
                        route_way_ids
                    )
                    
                    # bin_100m: way_id=12345 (in route), offset diff = |100 - 200| = 100 <= 400 ✓
                    # bin_300m: way_id=12345 (in route), offset diff = |300 - 200| = 100 <= 400 ✓
                    # bin_700m: way_id=54321 (not in route) ✗
                    assert "bin_100m" in result
                    assert "bin_300m" in result
                    assert "bin_700m" not in result
    
    @pytest.mark.asyncio
    async def test_detour_exclusion(self, corridor_agent, straight_road_polyline, test_bins, route_way_ids):
        """Test that bins beyond detour limits are excluded"""
        with patch.object(corridor_agent, 'tube_filter', return_value=set()):
            with patch.object(corridor_agent, 'crowflight_candidates', return_value={test_bins[2]}):  # only bin_700m
                
                # Mock detour calculation to exceed limits
                def mock_detour(polyline, bin_obj):
                    if bin_obj.id == "bin_700m":
                        return 0.4  # > MAX_DETOUR_KM (0.3)
                    return 0.1
                
                with patch.object(corridor_agent, 'detour_km', side_effect=mock_detour):
                    # Mock route length for ratio calculation
                    with patch.object(corridor_agent, 'calculate_route_length_km', return_value=1.0):
                        
                        result = await corridor_agent.build_corridor(
                            straight_road_polyline,
                            route_way_ids
                        )
                        
                        # bin_700m should be excluded: detour 0.4 km > 0.3 km AND ratio 0.4/1.0 > 0.05
                        assert "bin_700m" not in result
    
    @pytest.mark.asyncio
    async def test_detour_inclusion_by_distance(self, corridor_agent, straight_road_polyline, test_bins, route_way_ids):
        """Test that bins within detour distance limit are included"""
        with patch.object(corridor_agent, 'tube_filter', return_value=set()):
            with patch.object(corridor_agent, 'crowflight_candidates', return_value={test_bins[2]}):  # only bin_700m
                
                # Mock detour to be within distance limit
                def mock_detour(polyline, bin_obj):
                    if bin_obj.id == "bin_700m":
                        return 0.2  # <= MAX_DETOUR_KM (0.3)
                    return 0.1
                
                with patch.object(corridor_agent, 'detour_km', side_effect=mock_detour):
                    
                    result = await corridor_agent.build_corridor(
                        straight_road_polyline,
                        route_way_ids
                    )
                    
                    # bin_700m should be included: detour 0.2 km <= 0.3 km
                    assert "bin_700m" in result
    
    @pytest.mark.asyncio
    async def test_detour_inclusion_by_ratio(self, corridor_agent, straight_road_polyline, test_bins, route_way_ids):
        """Test that bins within detour ratio limit are included"""
        with patch.object(corridor_agent, 'tube_filter', return_value=set()):
            with patch.object(corridor_agent, 'crowflight_candidates', return_value={test_bins[2]}):  # only bin_700m
                
                # Mock detour to exceed distance but satisfy ratio
                def mock_detour(polyline, bin_obj):
                    if bin_obj.id == "bin_700m":
                        return 0.4  # > MAX_DETOUR_KM (0.3) but...
                    return 0.1
                
                with patch.object(corridor_agent, 'detour_km', side_effect=mock_detour):
                    # Mock long route to make ratio acceptable
                    with patch.object(corridor_agent, 'calculate_route_length_km', return_value=10.0):
                        
                        result = await corridor_agent.build_corridor(
                            straight_road_polyline,
                            route_way_ids
                        )
                        
                        # bin_700m should be included: ratio 0.4/10.0 = 0.04 <= 0.05
                        assert "bin_700m" in result
    
    @pytest.mark.asyncio
    async def test_complete_corridor_logic(self, corridor_agent, straight_road_polyline, test_bins, route_way_ids, route_offset_map):
        """Test complete corridor logic with expected inclusion pattern"""
        # Mock tube filter to include first bin only
        with patch.object(corridor_agent, 'tube_filter', return_value={"bin_100m"}):
            # All bins are candidates
            with patch.object(corridor_agent, 'crowflight_candidates', return_value=set(test_bins)):
                
                # Mock route offset mapping
                corridor_agent.route_offset_m = route_offset_map
                
                # Mock detour for bin_700m to exceed limits
                def mock_detour(polyline, bin_obj):
                    if bin_obj.id == "bin_700m":
                        return 0.5  # > both distance and ratio limits
                    return 0.1
                
                with patch.object(corridor_agent, 'detour_km', side_effect=mock_detour):
                    with patch.object(corridor_agent, 'calculate_route_length_km', return_value=1.0):
                        
                        result = await corridor_agent.build_corridor(
                            straight_road_polyline,
                            route_way_ids
                        )
                        
                        # Expected results per specification:
                        # ✔ bin_100m: in tube
                        # ✔ bin_300m: edge match (way_id=12345, offset within threshold)
                        # ✖ bin_700m: detour exceeds limits
                        assert "bin_100m" in result
                        assert "bin_300m" in result
                        assert "bin_700m" not in result
    
    @pytest.mark.asyncio
    async def test_settings_compliance(self, corridor_agent):
        """Test that corridor agent uses correct settings values"""
        # Verify default settings match specification
        assert settings.CORRIDOR_HALF_M == 250
        assert settings.WAY_OFFSET_THRESH_M == 400
        assert settings.MAX_DETOUR_KM == 0.3
        assert settings.MAX_DETOUR_RATIO == 0.05
        assert settings.CANDIDATE_SCAN_RADIUS == 1000