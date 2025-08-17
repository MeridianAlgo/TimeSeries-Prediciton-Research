"""
Unit tests for multi-asset data coordinator.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from enhanced_timeseries.multi_asset.data_coordinator import (
    AssetData, DataSyncStatus, DataStorage, DataSynchronizer,
    BatchProcessor, MultiAssetDataCoordinator
)


class TestAssetData(unittest.TestCase):
    """Test AssetData class."""
    
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        self.test_data = pd.DataFrame({
            'Open': np.random.randn(len(dates)) * 10 + 100,
            'High': np.random.randn(len(dates)) * 10 + 105,
            'Low': np.random.randn(len(dates)) * 10 + 95,
            'Close': np.random.randn(len(dates)) * 10 + 100,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
    
    def test_asset_data_creation(self):
        """Test AssetData creation."""
        asset = AssetData(
            symbol='AAPL',
            data=self.test_data,
            last_updated=datetime.now()
        )
        
        self.assertEqual(asset.symbol, 'AAPL')
        self.assertIsInstance(asset.data, pd.DataFrame)
        self.assertIsInstance(asset.last_updated, datetime)
        self.assertGreaterEqual(asset.data_quality_score, 0)
        self.assertLessEqual(asset.data_quality_score, 1)
        self.assertGreaterEqual(asset.missing_data_pct, 0)
    
    def test_asset_data_with_missing_values(self):
        """Test AssetData with missing values."""
        # Add some NaN values
        data_with_nan = self.test_data.copy()
        data_with_nan.iloc[0:10, 0] = np.nan
        data_with_nan.iloc[20:25, 1] = np.nan
        
        asset = AssetData(
            symbol='TEST',
            data=data_with_nan,
            last_updated=datetime.now()
        )
        
        self.assertGreater(asset.missing_data_pct, 0)
        self.assertLess(asset.data_quality_score, 1)
    
    def test_asset_data_empty_dataframe(self):
        """Test AssetData with empty DataFrame."""
        empty_data = pd.DataFrame()
        
        asset = AssetData(
            symbol='EMPTY',
            data=empty_data,
            last_updated=datetime.now()
        )
        
        self.assertEqual(asset.missing_data_pct, 0)
        self.assertEqual(asset.data_quality_score, 1.0)


class TestDataSyncStatus(unittest.TestCase):
    """Test DataSyncStatus class."""
    
    def test_sync_status_creation(self):
        """Test DataSyncStatus creation."""
        status = DataSyncStatus(
            total_assets=10,
            synchronized_assets=8,
            common_start_date=datetime(2020, 1, 1),
            common_end_date=datetime(2020, 12, 31),
            alignment_method='inner',
            missing_data_handled=True
        )
        
        self.assertEqual(status.total_assets, 10)
        self.assertEqual(status.synchronized_assets, 8)
        self.assertEqual(status.sync_percentage, 80.0)
        self.assertEqual(status.alignment_method, 'inner')
        self.assertTrue(status.missing_data_handled)
    
    def test_sync_percentage_zero_assets(self):
        """Test sync percentage with zero assets."""
        status = DataSyncStatus(
            total_assets=0,
            synchronized_assets=0,
            common_start_date=None,
            common_end_date=None,
            alignment_method='inner',
            missing_data_handled=False
        )
        
        self.assertEqual(status.sync_percentage, 0.0)


class TestDataStorage(unittest.TestCase):
    """Test DataStorage class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = DataStorage(self.temp_dir)
        
        # Create test data
        dates = pd.date_range('2020-01-01', '2020-03-31', freq='D')
        self.test_data = pd.DataFrame({
            'Close': np.random.randn(len(dates)) * 10 + 100,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        self.asset_data = AssetData(
            symbol='TEST',
            data=self.test_data,
            last_updated=datetime.now(),
            metadata={'source': 'test'}
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_storage_initialization(self):
        """Test storage initialization."""
        self.assertTrue(os.path.exists(self.storage.storage_path))
        self.assertTrue(os.path.exists(self.storage.db_path))
    
    def test_store_and_load_asset_data(self):
        """Test storing and loading asset data."""
        # Store data
        success = self.storage.store_asset_data(self.asset_data)
        self.assertTrue(success)
        
        # Load data
        loaded_asset = self.storage.load_asset_data('TEST')
        self.assertIsNotNone(loaded_asset)
        self.assertEqual(loaded_asset.symbol, 'TEST')
        self.assertEqual(len(loaded_asset.data), len(self.test_data))
        self.assertEqual(loaded_asset.metadata['source'], 'test')
    
    def test_get_stored_symbols(self):
        """Test getting stored symbols."""
        # Initially empty
        symbols = self.storage.get_stored_symbols()
        self.assertEqual(len(symbols), 0)
        
        # Store some data
        self.storage.store_asset_data(self.asset_data)
        
        symbols = self.storage.get_stored_symbols()
        self.assertEqual(len(symbols), 1)
        self.assertIn('TEST', symbols)
    
    def test_load_nonexistent_asset(self):
        """Test loading non-existent asset."""
        loaded_asset = self.storage.load_asset_data('NONEXISTENT')
        self.assertIsNone(loaded_asset)
    
    def test_cleanup_old_data(self):
        """Test cleanup of old data."""
        # Store data with old timestamp
        old_asset = AssetData(
            symbol='OLD',
            data=self.test_data,
            last_updated=datetime.now() - timedelta(days=35)
        )
        
        self.storage.store_asset_data(old_asset)
        self.storage.store_asset_data(self.asset_data)  # Recent data
        
        # Check both are stored
        symbols = self.storage.get_stored_symbols()
        self.assertEqual(len(symbols), 2)
        
        # Cleanup old data (older than 30 days)
        self.storage.cleanup_old_data(30)
        
        # Check only recent data remains
        symbols = self.storage.get_stored_symbols()
        self.assertEqual(len(symbols), 1)
        self.assertIn('TEST', symbols)
        self.assertNotIn('OLD', symbols)


class TestDataSynchronizer(unittest.TestCase):
    """Test DataSynchronizer class."""
    
    def setUp(self):
        """Set up test data."""
        self.synchronizer = DataSynchronizer('inner')
        
        # Create test assets with different date ranges
        dates1 = pd.date_range('2020-01-01', '2020-06-30', freq='D')
        dates2 = pd.date_range('2020-02-01', '2020-08-31', freq='D')
        dates3 = pd.date_range('2020-03-01', '2020-05-31', freq='D')
        
        self.asset1 = AssetData(
            symbol='ASSET1',
            data=pd.DataFrame({
                'Close': np.random.randn(len(dates1)) + 100,
                'Volume': np.random.randint(1000, 5000, len(dates1))
            }, index=dates1),
            last_updated=datetime.now()
        )
        
        self.asset2 = AssetData(
            symbol='ASSET2',
            data=pd.DataFrame({
                'Close': np.random.randn(len(dates2)) + 200,
                'Volume': np.random.randint(2000, 6000, len(dates2))
            }, index=dates2),
            last_updated=datetime.now()
        )
        
        self.asset3 = AssetData(
            symbol='ASSET3',
            data=pd.DataFrame({
                'Close': np.random.randn(len(dates3)) + 300,
                'Volume': np.random.randint(3000, 7000, len(dates3))
            }, index=dates3),
            last_updated=datetime.now()
        )
        
        self.assets = {
            'ASSET1': self.asset1,
            'ASSET2': self.asset2,
            'ASSET3': self.asset3
        }
    
    def test_synchronizer_creation(self):
        """Test synchronizer creation."""
        sync = DataSynchronizer('outer')
        self.assertEqual(sync.alignment_method, 'outer')
    
    def test_inner_synchronization(self):
        """Test inner join synchronization."""
        sync_assets, status = self.synchronizer.synchronize_assets(self.assets)
        
        self.assertEqual(len(sync_assets), 3)
        self.assertEqual(status.total_assets, 3)
        self.assertGreater(status.synchronized_assets, 0)
        self.assertEqual(status.alignment_method, 'inner')
        
        # Check that all assets have the same date range (inner join)
        indices = [asset.data.index for asset in sync_assets.values()]
        for i in range(1, len(indices)):
            pd.testing.assert_index_equal(indices[0], indices[i])
    
    def test_outer_synchronization(self):
        """Test outer join synchronization."""
        outer_sync = DataSynchronizer('outer')
        sync_assets, status = outer_sync.synchronize_assets(self.assets)
        
        self.assertEqual(len(sync_assets), 3)
        self.assertEqual(status.alignment_method, 'outer')
        
        # Check that all assets have the same date range (outer join)
        indices = [asset.data.index for asset in sync_assets.values()]
        for i in range(1, len(indices)):
            pd.testing.assert_index_equal(indices[0], indices[i])
    
    def test_synchronization_with_empty_assets(self):
        """Test synchronization with empty assets dictionary."""
        sync_assets, status = self.synchronizer.synchronize_assets({})
        
        self.assertEqual(len(sync_assets), 0)
        self.assertEqual(status.total_assets, 0)
        self.assertEqual(status.synchronized_assets, 0)
    
    def test_fill_methods(self):
        """Test different fill methods."""
        # Add some NaN values
        self.asset1.data.iloc[10:15, 0] = np.nan
        
        # Test forward fill
        sync_assets, _ = self.synchronizer.synchronize_assets(self.assets, fill_method='forward')
        self.assertFalse(sync_assets['ASSET1'].data['Close'].isnull().any())
        
        # Test interpolation
        sync_assets, _ = self.synchronizer.synchronize_assets(self.assets, fill_method='interpolate')
        self.assertFalse(sync_assets['ASSET1'].data['Close'].isnull().any())


class TestBatchProcessor(unittest.TestCase):
    """Test BatchProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.processor = BatchProcessor(max_workers=2, batch_size=3)
        
        # Mock data loader function
        def mock_data_loader(symbol, **kwargs):
            dates = pd.date_range('2020-01-01', '2020-03-31', freq='D')
            return pd.DataFrame({
                'Close': np.random.randn(len(dates)) + 100,
                'Volume': np.random.randint(1000, 5000, len(dates))
            }, index=dates)
        
        self.mock_data_loader = mock_data_loader
    
    def test_processor_creation(self):
        """Test batch processor creation."""
        self.assertEqual(self.processor.max_workers, 2)
        self.assertEqual(self.processor.batch_size, 3)
    
    def test_process_assets_batch(self):
        """Test batch processing of assets."""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        results = self.processor.process_assets_batch(
            symbols, self.mock_data_loader
        )
        
        self.assertEqual(len(results), 5)
        
        for symbol in symbols:
            self.assertIn(symbol, results)
            self.assertIsInstance(results[symbol], AssetData)
            self.assertEqual(results[symbol].symbol, symbol)
            self.assertFalse(results[symbol].data.empty)
    
    def test_batch_processing_with_processing_func(self):
        """Test batch processing with additional processing function."""
        def processing_func(data, **kwargs):
            # Add a simple moving average
            data['SMA_5'] = data['Close'].rolling(5).mean()
            return data
        
        symbols = ['AAPL', 'GOOGL']
        
        results = self.processor.process_assets_batch(
            symbols, self.mock_data_loader, processing_func
        )
        
        for symbol in symbols:
            self.assertIn('SMA_5', results[symbol].data.columns)
    
    def test_batch_processing_with_failed_symbol(self):
        """Test batch processing with a failing symbol."""
        def failing_data_loader(symbol, **kwargs):
            if symbol == 'FAIL':
                raise Exception("Simulated failure")
            return self.mock_data_loader(symbol, **kwargs)
        
        symbols = ['AAPL', 'FAIL', 'GOOGL']
        
        results = self.processor.process_assets_batch(
            symbols, failing_data_loader
        )
        
        # Should have results for successful symbols only
        self.assertEqual(len(results), 2)
        self.assertIn('AAPL', results)
        self.assertIn('GOOGL', results)
        self.assertNotIn('FAIL', results)


class TestMultiAssetDataCoordinator(unittest.TestCase):
    """Test MultiAssetDataCoordinator class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.coordinator = MultiAssetDataCoordinator(
            storage_path=self.temp_dir,
            max_assets=5,
            max_workers=2,
            batch_size=2
        )
        
        # Mock data loader
        def mock_data_loader(symbol, **kwargs):
            dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
            return pd.DataFrame({
                'Open': np.random.randn(len(dates)) + 100,
                'High': np.random.randn(len(dates)) + 105,
                'Low': np.random.randn(len(dates)) + 95,
                'Close': np.random.randn(len(dates)) + 100,
                'Volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
        
        self.mock_data_loader = mock_data_loader
    
    def tearDown(self):
        """Clean up test environment."""
        # Ensure all database connections are closed
        if hasattr(self, 'coordinator'):
            # Force cleanup of coordinator resources
            self.coordinator.assets.clear()
        
        # Small delay to ensure file handles are released
        import time
        time.sleep(0.1)
        
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            # On Windows, sometimes files are still locked
            # Try multiple times with increasing delays
            for i in range(3):
                time.sleep(0.5 * (i + 1))
                try:
                    shutil.rmtree(self.temp_dir)
                    break
                except PermissionError:
                    if i == 2:  # Last attempt
                        # If still failing, just skip cleanup
                        # The temp directory will be cleaned up by the OS eventually
                        pass
    
    def test_coordinator_creation(self):
        """Test coordinator creation."""
        self.assertEqual(self.coordinator.max_assets, 5)
        self.assertEqual(len(self.coordinator.assets), 0)
        self.assertIsNone(self.coordinator.sync_status)
    
    def test_add_assets(self):
        """Test adding assets to coordinator."""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        results = self.coordinator.add_assets(symbols, self.mock_data_loader)
        
        self.assertEqual(len(results), 3)
        for symbol in symbols:
            self.assertTrue(results[symbol])
        
        # Check assets are stored
        self.assertEqual(len(self.coordinator.assets), 3)
        for symbol in symbols:
            self.assertIn(symbol, self.coordinator.assets)
    
    def test_add_assets_exceeding_capacity(self):
        """Test adding more assets than capacity."""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NFLX', 'META']  # 7 symbols, capacity is 5
        
        results = self.coordinator.add_assets(symbols, self.mock_data_loader)
        
        # Should only process first 5 symbols
        self.assertEqual(len(results), 5)
        self.assertEqual(len(self.coordinator.assets), 5)
    
    def test_remove_assets(self):
        """Test removing assets from coordinator."""
        # First add some assets
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        self.coordinator.add_assets(symbols, self.mock_data_loader)
        
        # Remove some assets
        remove_results = self.coordinator.remove_assets(['AAPL', 'GOOGL'])
        
        self.assertTrue(remove_results['AAPL'])
        self.assertTrue(remove_results['GOOGL'])
        self.assertEqual(len(self.coordinator.assets), 1)
        self.assertIn('MSFT', self.coordinator.assets)
    
    def test_get_asset_data(self):
        """Test getting individual asset data."""
        symbols = ['AAPL']
        self.coordinator.add_assets(symbols, self.mock_data_loader)
        
        asset_data = self.coordinator.get_asset_data('AAPL')
        
        self.assertIsNotNone(asset_data)
        self.assertEqual(asset_data.symbol, 'AAPL')
        self.assertFalse(asset_data.data.empty)
    
    def test_get_nonexistent_asset(self):
        """Test getting non-existent asset data."""
        asset_data = self.coordinator.get_asset_data('NONEXISTENT')
        self.assertIsNone(asset_data)
    
    def test_synchronize_all_assets(self):
        """Test synchronizing all assets."""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        self.coordinator.add_assets(symbols, self.mock_data_loader)
        
        sync_status = self.coordinator.synchronize_all_assets()
        
        self.assertIsNotNone(sync_status)
        self.assertEqual(sync_status.total_assets, 3)
        self.assertGreater(sync_status.synchronized_assets, 0)
        self.assertIsNotNone(self.coordinator.sync_status)
    
    def test_get_synchronized_dataframe(self):
        """Test getting synchronized DataFrame."""
        symbols = ['AAPL', 'GOOGL']
        self.coordinator.add_assets(symbols, self.mock_data_loader)
        self.coordinator.synchronize_all_assets()
        
        df = self.coordinator.get_synchronized_dataframe()
        
        self.assertFalse(df.empty)
        self.assertIsInstance(df.columns, pd.MultiIndex)
        
        # Test with specific columns
        df_close = self.coordinator.get_synchronized_dataframe(['Close'])
        self.assertFalse(df_close.empty)
        
        # Check that only Close columns are included
        for symbol in symbols:
            self.assertIn((symbol, 'Close'), df_close.columns)
    
    def test_get_asset_statistics(self):
        """Test getting asset statistics."""
        symbols = ['AAPL', 'GOOGL']
        self.coordinator.add_assets(symbols, self.mock_data_loader)
        
        stats = self.coordinator.get_asset_statistics()
        
        self.assertEqual(len(stats), 2)
        
        for symbol in symbols:
            self.assertIn(symbol, stats)
            asset_stats = stats[symbol]
            
            expected_keys = [
                'symbol', 'data_points', 'date_range', 'columns',
                'data_quality_score', 'missing_data_pct', 'last_updated'
            ]
            
            for key in expected_keys:
                self.assertIn(key, asset_stats)
    
    def test_get_memory_usage(self):
        """Test getting memory usage statistics."""
        symbols = ['AAPL', 'GOOGL']
        self.coordinator.add_assets(symbols, self.mock_data_loader)
        
        memory_stats = self.coordinator.get_memory_usage()
        
        expected_keys = [
            'total_memory_bytes', 'total_memory_mb', 'asset_count',
            'avg_memory_per_asset_mb', 'asset_memory_breakdown'
        ]
        
        for key in expected_keys:
            self.assertIn(key, memory_stats)
        
        self.assertEqual(memory_stats['asset_count'], 2)
        self.assertGreater(memory_stats['total_memory_bytes'], 0)
    
    def test_export_coordinator_state(self):
        """Test exporting coordinator state."""
        import json
        
        symbols = ['AAPL']
        self.coordinator.add_assets(symbols, self.mock_data_loader)
        self.coordinator.synchronize_all_assets()
        
        # Export to temporary file
        export_path = os.path.join(self.temp_dir, 'coordinator_state.json')
        self.coordinator.export_coordinator_state(export_path)
        
        # Check file exists and contains valid JSON
        self.assertTrue(os.path.exists(export_path))
        
        with open(export_path, 'r') as f:
            state = json.load(f)
        
        expected_keys = [
            'assets_count', 'asset_symbols', 'sync_status',
            'asset_statistics', 'memory_usage', 'export_timestamp'
        ]
        
        for key in expected_keys:
            self.assertIn(key, state)
        
        self.assertEqual(state['assets_count'], 1)
        self.assertEqual(state['asset_symbols'], ['AAPL'])
    
    def test_cleanup_coordinator(self):
        """Test coordinator cleanup."""
        symbols = ['AAPL', 'GOOGL']
        self.coordinator.add_assets(symbols, self.mock_data_loader)
        
        # Cleanup shouldn't remove recent data
        initial_count = len(self.coordinator.assets)
        self.coordinator.cleanup_coordinator(days_old=1)
        
        # Should still have the same number of assets
        self.assertEqual(len(self.coordinator.assets), initial_count)


if __name__ == '__main__':
    unittest.main()