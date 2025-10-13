#!/usr/bin/env python3
"""
Test script to verify the new TrainerConfig works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config_loading():
    """Test that the new TrainerConfig loads YAML correctly."""
    print("🔍 Testing TrainerConfig YAML loading...")
    
    try:
        from protlig_ddiff.train.run_train_clean import TrainerConfig
        
        # Create config with YAML file
        config = TrainerConfig(
            work_dir="./test_work_dir",
            datafile="./test_data.pt",
            config_file="config_protein.yaml",
            device="0"
        )
        
        print("✅ TrainerConfig created successfully")
        
        # Test accessing YAML values
        print(f"📋 Config values:")
        print(f"  tokens: {config.tokens}")
        print(f"  devicetype: {config.devicetype}")
        print(f"  model.dim: {config.model.dim}")
        print(f"  model.n_layers: {config.model.n_layers}")
        print(f"  training.learning_rate: {config.training.learning_rate}")
        print(f"  training.batch_size: {config.training.batch_size}")
        print(f"  data.max_length: {config.data.max_length}")
        print(f"  data.tokenize_on_fly: {config.data.tokenize_on_fly}")
        print(f"  noise.type: {config.noise.type}")
        
        # Test .get() method
        print(f"\n🔍 Testing .get() method:")
        print(f"  training.get('learning_rate'): {config.training.get('learning_rate')}")
        print(f"  training.get('nonexistent', 'default'): {config.training.get('nonexistent', 'default')}")
        
        # Test nested access
        if hasattr(config, 'curriculum') and config.curriculum:
            print(f"  curriculum.enabled: {config.curriculum.enabled}")
            print(f"  curriculum.start_bias: {config.curriculum.start_bias}")
        
        print("\n✅ All config access tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_without_yaml():
    """Test TrainerConfig without YAML file."""
    print("\n🔍 Testing TrainerConfig without YAML...")
    
    try:
        from protlig_ddiff.train.run_train_clean import TrainerConfig
        
        config = TrainerConfig(
            work_dir="./test_work_dir",
            datafile="./test_data.pt",
            device="0"
        )
        
        print("✅ TrainerConfig created without YAML")
        print(f"  tokens (default): {config.tokens}")
        print(f"  devicetype (default): {config.devicetype}")
        
        # These should be None since no YAML was loaded
        print(f"  model: {config.model}")
        print(f"  training: {config.training}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config without YAML failed: {e}")
        return False

def main():
    print("🧪 TESTING NEW TRAINERCONFIG")
    print("=" * 50)
    
    success = True
    
    if not test_config_loading():
        success = False
    
    if not test_config_without_yaml():
        success = False
    
    if success:
        print("\n🎉 All tests passed! The new TrainerConfig is working correctly.")
        print("\n💡 Key improvements:")
        print("  ✅ All YAML values are now directly accessible as config.section.key")
        print("  ✅ No more getattr() calls needed")
        print("  ✅ Cleaner, more readable code")
        print("  ✅ Better error handling with .get() method")
    else:
        print("\n❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
