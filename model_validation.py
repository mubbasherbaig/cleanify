#!/usr/bin/env python3
"""
Model Validation Script - Check if the core models work correctly
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_model_imports():
    """Test if all required models can be imported"""
    print("📦 Testing model imports...")
    
    try:
        from core.models import Bin, Truck, SystemState, TruckStatus, BinType
        print("✅ Core models imported successfully")
        return True
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False

def test_bin_model():
    """Test Bin model creation and attributes"""
    print("\n🗑️ Testing Bin model...")
    
    try:
        from core.models import Bin, BinType
        
        # Test basic creation
        bin_obj = Bin(
            id="TEST_BIN",
            lat=33.6844,
            lon=73.0479,
            capacity_l=100,
            fill_level=75.0,
            fill_rate_lph=3.0,
            tile_id=""
        )
        
        print(f"✅ Bin created: {bin_obj.id}")
        print(f"   - Location: ({bin_obj.lat}, {bin_obj.lon})")
        print(f"   - Capacity: {bin_obj.capacity_l}L")
        print(f"   - Fill level: {bin_obj.fill_level}%")
        print(f"   - Fill rate: {bin_obj.fill_rate_lph}L/h")
        
        # Test optional attributes
        print("🔧 Testing optional attributes...")
        print(f"   - Tile ID: {getattr(bin_obj, 'tile_id', 'MISSING')}")
        print(f"   - Threshold: {getattr(bin_obj, 'threshold', 'MISSING')}")
        print(f"   - Assigned truck: {getattr(bin_obj, 'assigned_truck', 'MISSING')}")
        print(f"   - Being collected: {getattr(bin_obj, 'being_collected', 'MISSING')}")
        print(f"   - Bin type: {getattr(bin_obj, 'bin_type', 'MISSING')}")
        
        # Test methods
        if hasattr(bin_obj, 'urgency_score'):
            urgency = bin_obj.urgency_score()
            print(f"   - Urgency score: {urgency}")
        
        if hasattr(bin_obj, 'is_urgent'):
            urgent = bin_obj.is_urgent()
            print(f"   - Is urgent: {urgent}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bin model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_truck_model():
    """Test Truck model creation and attributes"""
    print("\n🚛 Testing Truck model...")
    
    try:
        from core.models import Truck, TruckStatus
        
        # Test basic creation
        truck_obj = Truck(
            id="TEST_TRUCK",
            name="Test Truck",
            capacity_l=5000,
            lat=33.6844,
            lon=73.0479
        )
        
        print(f"✅ Truck created: {truck_obj.id}")
        print(f"   - Name: {truck_obj.name}")
        print(f"   - Location: ({truck_obj.lat}, {truck_obj.lon})")
        print(f"   - Capacity: {truck_obj.capacity_l}L")
        
        # Test optional attributes
        print("🔧 Testing optional attributes...")
        print(f"   - Current load: {getattr(truck_obj, 'current_load_l', 'MISSING')}")
        print(f"   - Status: {getattr(truck_obj, 'status', 'MISSING')}")
        print(f"   - Speed: {getattr(truck_obj, 'speed_kmh', 'MISSING')}")
        print(f"   - Route ID: {getattr(truck_obj, 'route_id', 'MISSING')}")
        
        # Test methods
        if hasattr(truck_obj, 'capacity_utilization'):
            util = truck_obj.capacity_utilization()
            print(f"   - Capacity utilization: {util}%")
        
        if hasattr(truck_obj, 'is_available'):
            available = truck_obj.is_available()
            print(f"   - Is available: {available}")
        
        return True
        
    except Exception as e:
        print(f"❌ Truck model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_state_model():
    """Test SystemState model creation"""
    print("\n🌐 Testing SystemState model...")
    
    try:
        from core.models import SystemState, Bin, Truck
        
        # Create test bin and truck
        test_bin = Bin(
            id="TEST_BIN",
            lat=33.6844,
            lon=73.0479,
            capacity_l=100,
            fill_level=75.0,
            fill_rate_lph=3.0,
            tile_id=""
        )
        
        test_truck = Truck(
            id="TEST_TRUCK",
            name="Test Truck",
            capacity_l=5000,
            lat=33.6844,
            lon=73.0479
        )
        
        # Create SystemState
        system_state = SystemState(
            timestamp=datetime.now(),
            bins=[test_bin],
            trucks=[test_truck],
            active_routes=[],
            traffic_conditions=[],
            simulation_running=False,
            current_time=datetime.now()
        )
        
        print(f"✅ SystemState created")
        print(f"   - Timestamp: {system_state.timestamp}")
        print(f"   - Bins: {len(system_state.bins)}")
        print(f"   - Trucks: {len(system_state.trucks)}")
        print(f"   - Active routes: {len(system_state.active_routes)}")
        print(f"   - Simulation running: {system_state.simulation_running}")
        
        # Test methods
        if hasattr(system_state, 'urgent_bins'):
            urgent = system_state.urgent_bins()
            print(f"   - Urgent bins: {len(urgent)}")
        
        if hasattr(system_state, 'available_trucks'):
            available = system_state.available_trucks()
            print(f"   - Available trucks: {len(available)}")
        
        return True
        
    except Exception as e:
        print(f"❌ SystemState model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_supervisor_dict_methods():
    """Test the supervisor's dict conversion methods"""
    print("\n🔧 Testing supervisor dict conversion methods...")
    
    try:
        from agents.supervisor import SupervisorAgent
        from core.models import Bin, Truck, TruckStatus
        
        supervisor = SupervisorAgent()
        
        # Test bin conversion
        test_bin = Bin(
            id="TEST_BIN",
            lat=33.6844,
            lon=73.0479,
            capacity_l=100,
            fill_level=75.0,
            fill_rate_lph=3.0,
            tile_id=""
        )
        
        print("🔧 Testing _bin_to_dict...")
        bin_dict = supervisor._bin_to_dict(test_bin)
        print(f"✅ Bin dict: {bin_dict}")
        
        # Test truck conversion
        test_truck = Truck(
            id="TEST_TRUCK",
            name="Test Truck",
            capacity_l=5000,
            lat=33.6844,
            lon=73.0479
        )
        
        print("🔧 Testing _truck_to_dict...")
        truck_dict = supervisor._truck_to_dict(test_truck)
        print(f"✅ Truck dict: {truck_dict}")
        
        return True
        
    except Exception as e:
        print(f"❌ Supervisor dict methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all model validation tests"""
    print("🧪 MODEL VALIDATION TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Model Imports", test_model_imports),
        ("Bin Model", test_bin_model),
        ("Truck Model", test_truck_model),
        ("SystemState Model", test_system_state_model),
        ("Supervisor Dict Methods", test_supervisor_dict_methods)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("📊 MODEL VALIDATION RESULTS:")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   - {test_name}: {status}")
    
    if all(results.values()):
        print("\n🎉 All model validation tests PASSED!")
        print("💡 The core models are working correctly.")
    else:
        failed_tests = [name for name, result in results.items() if not result]
        print(f"\n❌ Failed tests: {', '.join(failed_tests)}")
        print("💡 Fix the model issues before testing the supervisor.")
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)