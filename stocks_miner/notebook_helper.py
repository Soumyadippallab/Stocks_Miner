# notebook_helper.py
# Place this file in: C:\Users\Soumyadip\Desktop\Stocks_Miner\stocks_miner\notebook_helper.py

import sys
import os

def setup_stocks_miner():
    """Setup Stocks Miner environment and return imported modules"""
    
    # Get the parent directory (one level up from stocks_miner)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Add parent directory to path if not already there
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import modules as part of the stocks_miner package
    try:
        # Import from the stocks_miner package
        from stocks_miner import market_indices
        from stocks_miner import nse_stocks
        from stocks_miner import tda_crash_detection
        from stocks_miner import utils
        
        # Create a simple namespace object to hold the modules
        class StocksMiner:
            """Container for Stocks Miner modules"""
            pass
        
        sm = StocksMiner()
        sm.market_indices = market_indices
        sm.nse_stocks = nse_stocks
        sm.tda_crash = tda_crash_detection  # Note: file is tda_crash_detection.py
        sm.utils = utils
        
        print("✓ Stocks Miner modules loaded successfully!")
        print("  - market_indices")
        print("  - nse_stocks")
        print("  - tda_crash (tda_crash_detection)")
        print("  - utils")
        print(f"\nModules loaded from package: stocks_miner")
        print(f"Package location: {parent_dir}")
        
        return sm
        
    except ImportError as e:
        print(f"✗ Error importing Stocks Miner modules: {e}")
        print(f"\nCurrent directory: {current_dir}")
        print(f"Parent directory: {parent_dir}")
        print(f"\nFiles in stocks_miner directory:")
        for file in os.listdir(current_dir):
            if file.endswith('.py'):
                print(f"  - {file}")
        print(f"\nCurrent sys.path:")
        for path in sys.path[:5]:
            print(f"  - {path}")
        raise


def setup_stocks_miner_global():
    """Setup Stocks Miner and inject modules into caller's namespace (alternative method)"""
    import inspect
    
    # Get the parent directory (one level up from stocks_miner)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Add parent directory to path
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import modules as part of the package
    try:
        from stocks_miner import market_indices
        from stocks_miner import nse_stocks
        from stocks_miner import tda_crash_detection
        from stocks_miner import utils
        
        # Get the caller's globals (the notebook's namespace)
        caller_globals = inspect.currentframe().f_back.f_globals
        
        # Inject into caller's namespace
        caller_globals['market_indices'] = market_indices
        caller_globals['nse_stocks'] = nse_stocks
        caller_globals['tda_crash'] = tda_crash_detection
        caller_globals['utils'] = utils
        
        print("✓ Stocks Miner modules loaded into notebook namespace")
        print("  You can now use: market_indices, nse_stocks, tda_crash (tda_crash_detection), utils")
        print(f"\nModules loaded from package: stocks_miner")
        print(f"Package location: {parent_dir}")
        
    except ImportError as e:
        print(f"✗ Error importing Stocks Miner modules: {e}")
        print(f"\nParent directory: {parent_dir}")
        raise


if __name__ == "__main__":
    # Test the setup
    print("Testing notebook_helper.py...")
    sm = setup_stocks_miner()
    print("\n✓ Test successful! Modules are accessible via sm.module_name")