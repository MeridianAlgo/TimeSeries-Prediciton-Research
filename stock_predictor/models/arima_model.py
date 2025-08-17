"""ARIMA model implementation for stock price prediction."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

from stock_predictor.models.base import BaseModel
from stock_predictor.utils.exceptions import ModelTrainingError, ModelPredictionError


class ARIMAModel(BaseModel):
    """ARIMA model for time series forecasting."""
    
    def __init__(self, name: str = "arima"):
        super().__init__(name)
        self.order = None
        self.seasonal_order = None
        self.fitted_model = None
        self.differencing_applied = 0
        
        # Default hyperparameters
        self.hyperparameters = {
            'max_p': 5,
            'max_d': 2,
            'max_q': 5,
            'seasonal': True,
            'seasonal_periods': 252,  # Trading days in a year
            'information_criterion': 'aic',
            'trend': 'c'  # constant trend
        }
    
    def _build_model(self) -> ARIMA:
        """Build ARIMA model with current hyperparameters."""
        # For ARIMA, we don't pre-build the model since it needs data
        # This will be handled in _fit_model
        return None
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit ARIMA model to training data."""
        # ARIMA works with time series, so we use y_train as the series
        # X_train is ignored for pure ARIMA (could be used for ARIMAX)
        
        try:
            # Convert to pandas Series for statsmodels
            if isinstance(y_train, np.ndarray):
                ts_data = pd.Series(y_train)
            else:
                ts_data = y_train
            
            # Auto-select ARIMA order if not already done
            if self.order is None:
                self.logger.info("Auto-selecting ARIMA order...")
                # Use simpler parameters for faster convergence
                self.hyperparameters.update({
                    'max_p': 3,
                    'max_q': 3,
                    'seasonal': False  # Disable seasonal for faster training
                })
                self.order, self.seasonal_order = self.auto_arima_selection(ts_data)
            
            # Create and fit ARIMA model
            arima_model = ARIMA(
                endog=ts_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.hyperparameters.get('trend', 'c')
            )
            
            self.fitted_model = arima_model.fit()
            self.model = arima_model  # Store the model object for base class compatibility
            
            # Store training history
            self.training_history = {
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'aic': float(self.fitted_model.aic),
                'bic': float(self.fitted_model.bic),
                'log_likelihood': float(self.fitted_model.llf),
                'n_observations': len(ts_data)
            }
            
            self.logger.info(f"ARIMA{self.order} fitted with AIC: {self.fitted_model.aic:.2f}")
            
        except Exception as e:
            raise ModelTrainingError(f"ARIMA model fitting failed: {str(e)}")
    
    def _predict_model(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions using fitted ARIMA model."""
        if self.fitted_model is None:
            raise ModelPredictionError("ARIMA model not fitted")
        
        try:
            # For ARIMA, we predict the next n_steps
            n_steps = len(X_test) if X_test is not None else 1
            
            # Get forecast
            forecast_result = self.fitted_model.forecast(steps=n_steps)
            
            if isinstance(forecast_result, pd.Series):
                predictions = forecast_result.values
            else:
                predictions = np.array(forecast_result)
            
            return predictions
            
        except Exception as e:
            raise ModelPredictionError(f"ARIMA prediction failed: {str(e)}")
    
    def predict_with_uncertainty(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals."""
        if self.fitted_model is None:
            raise ModelPredictionError("ARIMA model not fitted")
        
        try:
            n_steps = len(X_test) if X_test is not None else 1
            
            # Get forecast with confidence intervals
            forecast_result = self.fitted_model.get_forecast(steps=n_steps)
            
            predictions = forecast_result.predicted_mean.values
            conf_int = forecast_result.conf_int()
            
            # Calculate uncertainty as half the confidence interval width
            uncertainties = (conf_int.iloc[:, 1] - conf_int.iloc[:, 0]).values / 2
            
            return predictions, uncertainties
            
        except Exception as e:
            raise ModelPredictionError(f"ARIMA prediction with uncertainty failed: {str(e)}")
    
    def auto_arima_selection(self, data: pd.Series) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int, int]]]:
        """
        Automatically select optimal ARIMA order using grid search.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (order, seasonal_order)
        """
        self.logger.info("Starting auto ARIMA order selection")
        
        # Check stationarity and determine differencing
        d = self._determine_differencing(data)
        
        # Grid search parameters
        max_p = self.hyperparameters.get('max_p', 5)
        max_q = self.hyperparameters.get('max_q', 5)
        ic = self.hyperparameters.get('information_criterion', 'aic')
        
        best_ic = np.inf
        best_order = (0, d, 0)
        best_seasonal_order = None
        
        # Grid search for non-seasonal parameters
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    order = (p, d, q)
                    seasonal_order = None
                    
                    # Add seasonal component if enabled
                    if self.hyperparameters.get('seasonal', False):
                        seasonal_periods = self.hyperparameters.get('seasonal_periods', 252)
                        # Simple seasonal order (1,1,1,s) - can be made more sophisticated
                        seasonal_order = (1, 1, 1, seasonal_periods)
                    
                    # Fit model
                    model = ARIMA(data, order=order, seasonal_order=seasonal_order)
                    fitted = model.fit()
                    
                    # Get information criterion
                    current_ic = fitted.aic if ic == 'aic' else fitted.bic
                    
                    if current_ic < best_ic:
                        best_ic = current_ic
                        best_order = order
                        best_seasonal_order = seasonal_order
                        
                except Exception as e:
                    # Skip this combination if it fails
                    continue
        
        self.logger.info(f"Best ARIMA order: {best_order}, seasonal: {best_seasonal_order}, {ic.upper()}: {best_ic:.2f}")
        
        return best_order, best_seasonal_order
    
    def _determine_differencing(self, data: pd.Series) -> int:
        """
        Determine the degree of differencing needed for stationarity.
        
        Args:
            data: Time series data
            
        Returns:
            Degree of differencing (d parameter)
        """
        max_d = self.hyperparameters.get('max_d', 2)
        
        # Test original series
        if self._is_stationary(data):
            return 0
        
        # Test differenced series
        current_data = data.copy()
        for d in range(1, max_d + 1):
            current_data = current_data.diff().dropna()
            if self._is_stationary(current_data):
                return d
        
        # If still not stationary, return max_d
        self.logger.warning(f"Series may not be stationary even after {max_d} differences")
        return max_d
    
    def _is_stationary(self, data: pd.Series, significance_level: float = 0.05) -> bool:
        """
        Test if time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            data: Time series data
            significance_level: Significance level for the test
            
        Returns:
            True if series is stationary
        """
        try:
            # Perform ADF test
            adf_result = adfuller(data.dropna())
            p_value = adf_result[1]
            
            # Null hypothesis: series has unit root (non-stationary)
            # If p-value < significance_level, reject null (series is stationary)
            return p_value < significance_level
            
        except Exception:
            # If test fails, assume non-stationary
            return False
    
    def analyze_residuals(self) -> Dict[str, Any]:
        """
        Analyze model residuals for diagnostic purposes.
        
        Returns:
            Dictionary with residual analysis results
        """
        if self.fitted_model is None:
            raise ModelTrainingError("Model must be fitted before analyzing residuals")
        
        residuals = self.fitted_model.resid
        
        analysis = {
            'mean': float(residuals.mean()),
            'std': float(residuals.std()),
            'skewness': float(residuals.skew()),
            'kurtosis': float(residuals.kurtosis()),
            'ljung_box_p_value': None,
            'jarque_bera_p_value': None
        }
        
        try:
            # Ljung-Box test for autocorrelation in residuals
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
            analysis['ljung_box_p_value'] = float(lb_result['lb_pvalue'].iloc[-1])
        except Exception:
            pass
        
        try:
            # Jarque-Bera test for normality
            from scipy.stats import jarque_bera
            jb_stat, jb_p_value = jarque_bera(residuals)
            analysis['jarque_bera_p_value'] = float(jb_p_value)
        except Exception:
            pass
        
        return analysis
    
    def get_model_summary(self) -> str:
        """Get detailed model summary."""
        if self.fitted_model is None:
            return "Model not fitted"
        
        return str(self.fitted_model.summary())
    
    def forecast_with_history(self, steps: int, include_history: bool = True) -> pd.DataFrame:
        """
        Generate forecast with historical context.
        
        Args:
            steps: Number of steps to forecast
            include_history: Whether to include historical fitted values
            
        Returns:
            DataFrame with forecasts and confidence intervals
        """
        if self.fitted_model is None:
            raise ModelPredictionError("Model must be fitted before forecasting")
        
        # Get forecast
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        
        forecast_df = pd.DataFrame({
            'forecast': forecast_result.predicted_mean,
            'lower_ci': forecast_result.conf_int().iloc[:, 0],
            'upper_ci': forecast_result.conf_int().iloc[:, 1]
        })
        
        if include_history:
            # Add fitted values (in-sample predictions)
            fitted_df = pd.DataFrame({
                'forecast': self.fitted_model.fittedvalues,
                'lower_ci': np.nan,
                'upper_ci': np.nan
            })
            
            forecast_df = pd.concat([fitted_df, forecast_df])
        
        return forecast_df