"""
Multi-asset data coordination system for batch processing and synchronization.
Handles efficient data management for up to 50 symbols simultaneously.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import queue
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import pickle
import os
from pathlib import Path
import logging

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AssetData:
    """Container for individual asset data."""
    symbol: str
    data: pd.DataFrame
    last_updated: datetime
    data_quality_score: float = 1.0
    missing_data_pct: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.data is not None and not self.data.empty:
            self.missing_data_pct = self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])
            self.data_quality_score = max(0.0, 1.0 - self.missing_data_pct)


@dataclass
class DataSyncStatus:
    """Status of data synchronization across assets."""
    total_assets: int
    synchronized_assets: int
    common_start_date: Optional[datetime]
    common_end_date: Optional[datetime]
    alignment_method: str
    missing_data_handled: bool
    sync_timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def sync_percentage(self) -> float:
        """Calculate synchronization percentage."""
        return (self.synchronized_assets / self.total_assets * 100) if self.total_assets > 0 else 0.0


class DataStorage:
    """Efficient data storage and retrieval system."""
    
    def __init__(self, storage_path: str = "data/multi_asset_cache"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "asset_metadata.db"
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS asset_metadata (
                    symbol TEXT PRIMARY KEY,
                    last_updated TEXT,
                    data_quality_score REAL,
                    missing_data_pct REAL,
                    row_count INTEGER,
                    column_count INTEGER,
                    start_date TEXT,
                    end_date TEXT,
                    file_path TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()
    
    def store_asset_data(self, asset_data: AssetData) -> bool:
        """Store asset data to disk."""
        try:
            # Store DataFrame as pickle for efficiency
            file_path = self.storage_path / f"{asset_data.symbol}_data.pkl"
            
            with open(file_path, 'wb') as f:
                pickle.dump(asset_data.data, f)
            
            # Store metadata in database
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO asset_metadata 
                    (symbol, last_updated, data_quality_score, missing_data_pct, 
                     row_count, column_count, start_date, end_date, file_path, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    asset_data.symbol,
                    asset_data.last_updated.isoformat(),
                    asset_data.data_quality_score,
                    asset_data.missing_data_pct,
                    len(asset_data.data),
                    len(asset_data.data.columns),
                    asset_data.data.index[0].isoformat() if not asset_data.data.empty else None,
                    asset_data.data.index[-1].isoformat() if not asset_data.data.empty else None,
                    str(file_path),
                    pickle.dumps(asset_data.metadata)
                ))
                conn.commit()
            finally:
                conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store data for {asset_data.symbol}: {e}")
            return False
    
    def load_asset_data(self, symbol: str) -> Optional[AssetData]:
        """Load asset data from disk."""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("""
                    SELECT * FROM asset_metadata WHERE symbol = ?
                """, (symbol,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                # Load DataFrame
                file_path = Path(row[8])  # file_path column
                if not file_path.exists():
                    return None
                
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Reconstruct AssetData
                asset_data = AssetData(
                    symbol=row[0],
                    data=data,
                    last_updated=datetime.fromisoformat(row[1]),
                    data_quality_score=row[2],
                    missing_data_pct=row[3],
                    metadata=pickle.loads(row[9]) if row[9] else {}
                )
                
                return asset_data
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            return None
    
    def get_stored_symbols(self) -> List[str]:
        """Get list of stored symbols."""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("SELECT symbol FROM asset_metadata")
                return [row[0] for row in cursor.fetchall()]
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Failed to get stored symbols: {e}")
            return []
    
    def cleanup_old_data(self, days_old: int = 30):
        """Clean up data older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("""
                    SELECT symbol, file_path FROM asset_metadata 
                    WHERE last_updated < ?
                """, (cutoff_date.isoformat(),))
                
                old_records = cursor.fetchall()
                
                for symbol, file_path in old_records:
                    # Remove file
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
                    # Remove database record
                    conn.execute("DELETE FROM asset_metadata WHERE symbol = ?", (symbol,))
                
                conn.commit()
                logger.info(f"Cleaned up {len(old_records)} old data records")
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")


class DataSynchronizer:
    """Synchronize data across multiple assets."""
    
    def __init__(self, alignment_method: str = 'inner'):
        """
        Initialize data synchronizer.
        
        Args:
            alignment_method: 'inner', 'outer', 'left', 'right' for DataFrame alignment
        """
        self.alignment_method = alignment_method
        
    def synchronize_assets(self, assets: Dict[str, AssetData], 
                          fill_method: str = 'forward') -> Tuple[Dict[str, AssetData], DataSyncStatus]:
        """
        Synchronize data across multiple assets.
        
        Args:
            assets: Dictionary of asset data
            fill_method: Method to fill missing data ('forward', 'backward', 'interpolate', 'drop')
            
        Returns:
            Synchronized assets and sync status
        """
        if not assets:
            return {}, DataSyncStatus(0, 0, None, None, self.alignment_method, False)
        
        logger.info(f"Synchronizing {len(assets)} assets using {self.alignment_method} alignment")
        
        # Find common date range
        start_dates = []
        end_dates = []
        
        for asset_data in assets.values():
            if not asset_data.data.empty:
                start_dates.append(asset_data.data.index[0])
                end_dates.append(asset_data.data.index[-1])
        
        if not start_dates:
            return assets, DataSyncStatus(len(assets), 0, None, None, self.alignment_method, False)
        
        if self.alignment_method == 'inner':
            common_start = max(start_dates)
            common_end = min(end_dates)
        elif self.alignment_method == 'outer':
            common_start = min(start_dates)
            common_end = max(end_dates)
        else:
            # Use first asset as reference
            first_asset = list(assets.values())[0]
            common_start = first_asset.data.index[0]
            common_end = first_asset.data.index[-1]
        
        # Create common date index
        if common_start <= common_end:
            # Determine frequency from first asset
            first_asset_data = list(assets.values())[0].data
            if len(first_asset_data) > 1:
                freq = pd.infer_freq(first_asset_data.index)
                if freq is None:
                    # Fallback to daily frequency
                    freq = 'D'
            else:
                freq = 'D'
            
            try:
                common_index = pd.date_range(start=common_start, end=common_end, freq=freq)
            except Exception:
                # If frequency inference fails, use the index from the first asset
                common_index = first_asset_data.index
        else:
            logger.warning("No overlapping date range found")
            return assets, DataSyncStatus(len(assets), 0, None, None, self.alignment_method, False)
        
        # Synchronize each asset
        synchronized_assets = {}
        synchronized_count = 0
        
        for symbol, asset_data in assets.items():
            try:
                # Reindex to common index
                synchronized_data = asset_data.data.reindex(common_index)
                
                # Handle missing data
                if fill_method == 'forward':
                    synchronized_data = synchronized_data.ffill()
                elif fill_method == 'backward':
                    synchronized_data = synchronized_data.bfill()
                elif fill_method == 'interpolate':
                    synchronized_data = synchronized_data.interpolate(method='linear')
                elif fill_method == 'drop':
                    synchronized_data = synchronized_data.dropna()
                
                # Update asset data
                new_asset_data = AssetData(
                    symbol=symbol,
                    data=synchronized_data,
                    last_updated=asset_data.last_updated,
                    metadata=asset_data.metadata.copy()
                )
                
                synchronized_assets[symbol] = new_asset_data
                synchronized_count += 1
                
            except Exception as e:
                logger.error(f"Failed to synchronize {symbol}: {e}")
                # Keep original data if synchronization fails
                synchronized_assets[symbol] = asset_data
        
        sync_status = DataSyncStatus(
            total_assets=len(assets),
            synchronized_assets=synchronized_count,
            common_start_date=common_start,
            common_end_date=common_end,
            alignment_method=self.alignment_method,
            missing_data_handled=fill_method != 'drop'
        )
        
        logger.info(f"Synchronized {synchronized_count}/{len(assets)} assets successfully")
        return synchronized_assets, sync_status


class BatchProcessor:
    """Batch processing for multiple assets."""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 10):
        self.max_workers = max_workers
        self.batch_size = batch_size
        
    def process_assets_batch(self, symbols: List[str], 
                           data_loader_func, 
                           processing_func=None,
                           **kwargs) -> Dict[str, AssetData]:
        """
        Process multiple assets in batches.
        
        Args:
            symbols: List of asset symbols
            data_loader_func: Function to load data for a symbol
            processing_func: Optional processing function to apply to each asset
            **kwargs: Additional arguments for processing functions
            
        Returns:
            Dictionary of processed asset data
        """
        results = {}
        total_symbols = len(symbols)
        
        logger.info(f"Processing {total_symbols} symbols in batches of {self.batch_size}")
        
        # Process in batches
        for i in range(0, total_symbols, self.batch_size):
            batch_symbols = symbols[i:i + self.batch_size]
            batch_results = self._process_batch(batch_symbols, data_loader_func, processing_func, **kwargs)
            results.update(batch_results)
            
            logger.info(f"Processed batch {i//self.batch_size + 1}/{(total_symbols-1)//self.batch_size + 1}")
        
        return results
    
    def _process_batch(self, symbols: List[str], 
                      data_loader_func, 
                      processing_func=None,
                      **kwargs) -> Dict[str, AssetData]:
        """Process a single batch of symbols."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit loading tasks
            future_to_symbol = {
                executor.submit(self._load_and_process_symbol, symbol, data_loader_func, processing_func, **kwargs): symbol
                for symbol in symbols
            }
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    asset_data = future.result()
                    if asset_data is not None:
                        results[symbol] = asset_data
                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {e}")
        
        return results
    
    def _load_and_process_symbol(self, symbol: str, data_loader_func, processing_func=None, **kwargs) -> Optional[AssetData]:
        """Load and process a single symbol."""
        try:
            # Load data
            data = data_loader_func(symbol, **kwargs)
            
            if data is None or data.empty:
                logger.warning(f"No data loaded for {symbol}")
                return None
            
            # Create AssetData
            asset_data = AssetData(
                symbol=symbol,
                data=data,
                last_updated=datetime.now()
            )
            
            # Apply processing function if provided
            if processing_func is not None:
                processed_data = processing_func(asset_data.data, **kwargs)
                asset_data.data = processed_data
                asset_data.last_updated = datetime.now()
            
            return asset_data
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None


class MultiAssetDataCoordinator:
    """
    Main coordinator for multi-asset data management.
    Handles batch processing, synchronization, and storage for up to 50 symbols.
    """
    
    def __init__(self, 
                 storage_path: str = "data/multi_asset_cache",
                 max_assets: int = 50,
                 alignment_method: str = 'inner',
                 max_workers: int = 4,
                 batch_size: int = 10):
        """
        Initialize multi-asset data coordinator.
        
        Args:
            storage_path: Path for data storage
            max_assets: Maximum number of assets to handle
            alignment_method: Method for data alignment
            max_workers: Maximum worker threads
            batch_size: Batch size for processing
        """
        self.max_assets = max_assets
        self.storage = DataStorage(storage_path)
        self.synchronizer = DataSynchronizer(alignment_method)
        self.batch_processor = BatchProcessor(max_workers, batch_size)
        
        self.assets: Dict[str, AssetData] = {}
        self.sync_status: Optional[DataSyncStatus] = None
        self._lock = threading.Lock()
        
        logger.info(f"Initialized MultiAssetDataCoordinator for up to {max_assets} assets")
    
    def add_assets(self, symbols: List[str], data_loader_func, **kwargs) -> Dict[str, bool]:
        """
        Add multiple assets to the coordinator.
        
        Args:
            symbols: List of asset symbols
            data_loader_func: Function to load data for each symbol
            **kwargs: Additional arguments for data loader
            
        Returns:
            Dictionary indicating success/failure for each symbol
        """
        if len(symbols) > self.max_assets:
            logger.warning(f"Requested {len(symbols)} assets exceeds maximum {self.max_assets}")
            symbols = symbols[:self.max_assets]
        
        with self._lock:
            # Check if we're exceeding capacity
            current_count = len(self.assets)
            available_slots = self.max_assets - current_count
            
            if len(symbols) > available_slots:
                logger.warning(f"Only {available_slots} slots available, processing first {available_slots} symbols")
                symbols = symbols[:available_slots]
            
            # Process assets in batches
            results = {}
            
            try:
                asset_data_dict = self.batch_processor.process_assets_batch(
                    symbols, data_loader_func, **kwargs
                )
                
                # Add to coordinator and storage
                for symbol, asset_data in asset_data_dict.items():
                    success = self._add_single_asset(asset_data)
                    results[symbol] = success
                
                logger.info(f"Added {sum(results.values())}/{len(symbols)} assets successfully")
                
            except Exception as e:
                logger.error(f"Failed to add assets: {e}")
                results = {symbol: False for symbol in symbols}
            
            return results
    
    def _add_single_asset(self, asset_data: AssetData) -> bool:
        """Add a single asset to the coordinator."""
        try:
            # Store to disk
            if self.storage.store_asset_data(asset_data):
                # Add to memory
                self.assets[asset_data.symbol] = asset_data
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to add asset {asset_data.symbol}: {e}")
            return False
    
    def remove_assets(self, symbols: List[str]) -> Dict[str, bool]:
        """Remove assets from the coordinator."""
        results = {}
        
        with self._lock:
            for symbol in symbols:
                try:
                    if symbol in self.assets:
                        del self.assets[symbol]
                        results[symbol] = True
                    else:
                        results[symbol] = False
                except Exception as e:
                    logger.error(f"Failed to remove asset {symbol}: {e}")
                    results[symbol] = False
        
        return results
    
    def synchronize_all_assets(self, fill_method: str = 'forward') -> DataSyncStatus:
        """Synchronize all assets in the coordinator."""
        with self._lock:
            if not self.assets:
                return DataSyncStatus(0, 0, None, None, self.synchronizer.alignment_method, False)
            
            synchronized_assets, sync_status = self.synchronizer.synchronize_assets(
                self.assets, fill_method
            )
            
            # Update assets
            self.assets = synchronized_assets
            self.sync_status = sync_status
            
            # Update storage
            for asset_data in synchronized_assets.values():
                self.storage.store_asset_data(asset_data)
            
            return sync_status
    
    def get_asset_data(self, symbol: str) -> Optional[AssetData]:
        """Get data for a specific asset."""
        with self._lock:
            if symbol in self.assets:
                return self.assets[symbol]
            
            # Try loading from storage
            asset_data = self.storage.load_asset_data(symbol)
            if asset_data:
                self.assets[symbol] = asset_data
            
            return asset_data
    
    def get_all_assets(self) -> Dict[str, AssetData]:
        """Get all asset data."""
        with self._lock:
            return self.assets.copy()
    
    def get_synchronized_dataframe(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get synchronized data as a single DataFrame.
        
        Args:
            columns: Specific columns to include (default: all)
            
        Returns:
            Combined DataFrame with multi-level columns
        """
        with self._lock:
            if not self.assets:
                return pd.DataFrame()
            
            dataframes = []
            
            for symbol, asset_data in self.assets.items():
                df = asset_data.data.copy()
                
                if columns:
                    # Filter columns if specified
                    available_cols = [col for col in columns if col in df.columns]
                    if available_cols:
                        df = df[available_cols]
                    else:
                        continue
                
                # Add symbol level to columns
                df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
                dataframes.append(df)
            
            if dataframes:
                combined_df = pd.concat(dataframes, axis=1)
                return combined_df
            else:
                return pd.DataFrame()
    
    def get_asset_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all assets."""
        stats = {}
        
        with self._lock:
            for symbol, asset_data in self.assets.items():
                data = asset_data.data
                
                if not data.empty:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    
                    asset_stats = {
                        'symbol': symbol,
                        'data_points': len(data),
                        'date_range': {
                            'start': data.index[0].isoformat(),
                            'end': data.index[-1].isoformat()
                        },
                        'columns': list(data.columns),
                        'data_quality_score': asset_data.data_quality_score,
                        'missing_data_pct': asset_data.missing_data_pct,
                        'last_updated': asset_data.last_updated.isoformat()
                    }
                    
                    # Add numeric statistics
                    if len(numeric_cols) > 0:
                        numeric_data = data[numeric_cols]
                        asset_stats['statistics'] = {
                            'mean': numeric_data.mean().to_dict(),
                            'std': numeric_data.std().to_dict(),
                            'min': numeric_data.min().to_dict(),
                            'max': numeric_data.max().to_dict()
                        }
                    
                    stats[symbol] = asset_stats
        
        return stats
    
    def cleanup_coordinator(self, days_old: int = 30):
        """Clean up old data and optimize storage."""
        with self._lock:
            # Clean up storage
            self.storage.cleanup_old_data(days_old)
            
            # Remove assets that are no longer in storage
            stored_symbols = set(self.storage.get_stored_symbols())
            current_symbols = set(self.assets.keys())
            
            orphaned_symbols = current_symbols - stored_symbols
            for symbol in orphaned_symbols:
                del self.assets[symbol]
            
            logger.info(f"Cleaned up coordinator, removed {len(orphaned_symbols)} orphaned assets")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        import sys
        
        total_memory = 0
        asset_memory = {}
        
        with self._lock:
            for symbol, asset_data in self.assets.items():
                asset_size = sys.getsizeof(asset_data.data)
                asset_memory[symbol] = asset_size
                total_memory += asset_size
        
        return {
            'total_memory_bytes': total_memory,
            'total_memory_mb': total_memory / (1024 * 1024),
            'asset_count': len(self.assets),
            'avg_memory_per_asset_mb': (total_memory / len(self.assets) / (1024 * 1024)) if self.assets else 0,
            'asset_memory_breakdown': {k: v / (1024 * 1024) for k, v in asset_memory.items()}
        }
    
    def export_coordinator_state(self, filepath: str):
        """Export coordinator state to file."""
        state = {
            'assets_count': len(self.assets),
            'asset_symbols': list(self.assets.keys()),
            'sync_status': self.sync_status.__dict__ if self.sync_status else None,
            'asset_statistics': self.get_asset_statistics(),
            'memory_usage': self.get_memory_usage(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Coordinator state exported to {filepath}")