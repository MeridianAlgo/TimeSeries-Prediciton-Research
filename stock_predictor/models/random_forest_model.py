"""Random Forest model implementation for stock price prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

from stock_predictor.models.base import BaseModel
from stock_predictor.utils.exceptions import ModelTrainingError, ModelPredictionError


class RandomForestModel(BaseModel):
    """Random Forest model for stock price prediction."""
    
    def __init__(self, name: str = "random_forest"):
        super().__init__(name)
        self.feature_importances_ = None
        
        # Default hyperparameters
        self.hyperparameters = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1,
            'oob_score': True
        }
    
    def _build_model(self) -> RandomForestRegressor:
        """Build Random Forest model with current hyperparameters."""
        return RandomForestRegressor(**self.hyperparameters)
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit Random Forest model to training data."""
        try:
            # Fit the model
            self.model.fit(X_train, y_train)
            
            # Store feature importances
            self.feature_importances_ = self.model.feature_importances_
            
            # Calculate training metrics
            train_predictions = self.model.predict(X_train)
            train_mse = mean_squared_error(y_train, train_predictions)
            train_mae = mean_absolute_error(y_train, train_predictions)
            
            # Calculate validation metrics if validation data provided
            val_mse = None
            val_mae = None
            if X_val is not None and y_val is not None:
                val_predictions = self.model.predict(X_val)
                val_mse = mean_squared_error(y_val, val_predictions)
                val_mae = mean_absolute_error(y_val, val_predictions)
            
            # Store training history
            self.training_history = {
                'train_mse': float(train_mse),
                'train_mae': float(train_mae),
                'train_rmse': float(np.sqrt(train_mse)),
                'val_mse': float(val_mse) if val_mse is not None else None,
                'val_mae': float(val_mae) if val_mae is not None else None,
                'val_rmse': float(np.sqrt(val_mse)) if val_mse is not None else None,
                'oob_score': float(self.model.oob_score_) if hasattr(self.model, 'oob_score_') else None,
                'n_features': X_train.shape[1],
                'n_samples': X_train.shape[0]
            }
            
            self.logger.info(f"Random Forest training completed. Train RMSE: {self.training_history['train_rmse']:.6f}")
            if val_mse is not None:
                self.logger.info(f"Validation RMSE: {self.training_history['val_rmse']:.6f}")
            
        except Exception as e:
            raise ModelTrainingError(f"Random Forest model fitting failed: {str(e)}")
    
    def _predict_model(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions using trained Random Forest model."""
        if self.model is None:
            raise ModelPredictionError("Random Forest model not trained")
        
        try:
            predictions = self.model.predict(X_test)
            return predictions
            
        except Exception as e:
            raise ModelPredictionError(f"Random Forest prediction failed: {str(e)}")
    
    def predict_with_uncertainty(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates using tree variance.
        
        Args:
            X_test: Test features
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if self.model is None:
            raise ModelPredictionError("Random Forest model not trained")
        
        try:
            # Get predictions from all trees
            tree_predictions = np.array([
                tree.predict(X_test) for tree in self.model.estimators_
            ])
            
            # Calculate mean and standard deviation across trees
            predictions = np.mean(tree_predictions, axis=0)
            uncertainties = np.std(tree_predictions, axis=0)
            
            return predictions, uncertainties
            
        except Exception as e:
            raise ModelPredictionError(f"Random Forest prediction with uncertainty failed: {str(e)}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.feature_importances_ is None:
            return {}
        
        if len(self.feature_names) == len(self.feature_importances_):
            return dict(zip(self.feature_names, self.feature_importances_))
        else:
            # Fallback to generic names
            return {
                f'feature_{i}': importance 
                for i, importance in enumerate(self.feature_importances_)
            }
    
    def feature_importance_analysis(self, X: np.ndarray, feature_names: list = None) -> Dict[str, Any]:
        """
        Perform comprehensive feature importance analysis.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            
        Returns:
            Dictionary with importance analysis results
        """
        if self.model is None:
            raise ModelTrainingError("Model must be trained before feature importance analysis")
        
        if feature_names is None:
            feature_names = self.feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Calculate standard deviation of importances across trees
        std_importances = np.std([
            tree.feature_importances_ for tree in self.model.estimators_
        ], axis=0)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'std': std_importances
        }).sort_values('importance', ascending=False)
        
        # Calculate cumulative importance
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        
        # Find features that contribute to 95% of importance
        features_95 = importance_df[importance_df['cumulative_importance'] <= 0.95]
        
        analysis = {
            'importance_scores': dict(zip(feature_names, importances)),
            'importance_ranking': importance_df.to_dict('records'),
            'top_10_features': importance_df.head(10)['feature'].tolist(),
            'features_for_95_percent': features_95['feature'].tolist(),
            'n_features_for_95_percent': len(features_95),
            'mean_importance': float(np.mean(importances)),
            'std_importance': float(np.std(importances))
        }
        
        return analysis
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                             method: str = 'grid', cv: int = 5, n_iter: int = 50) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            method: Tuning method ('grid' or 'random')
            cv: Number of cross-validation folds
            n_iter: Number of iterations for random search
            
        Returns:
            Dictionary with tuning results
        """
        self.logger.info(f"Starting hyperparameter tuning using {method} search")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        try:
            # Create base model
            base_model = RandomForestRegressor(
                random_state=self.hyperparameters.get('random_state', 42),
                n_jobs=self.hyperparameters.get('n_jobs', -1)
            )
            
            # Perform search
            if method == 'grid':
                search = GridSearchCV(
                    base_model, param_grid, cv=cv, 
                    scoring='neg_mean_squared_error', n_jobs=-1
                )
            elif method == 'random':
                search = RandomizedSearchCV(
                    base_model, param_grid, cv=cv, n_iter=n_iter,
                    scoring='neg_mean_squared_error', n_jobs=-1,
                    random_state=self.hyperparameters.get('random_state', 42)
                )
            else:
                raise ValueError(f"Unknown tuning method: {method}")
            
            # Fit search
            search.fit(X_train, y_train)
            
            # Update hyperparameters with best parameters
            self.set_hyperparameters(search.best_params_)
            
            # Rebuild model with best parameters
            self.model = self._build_model()
            
            tuning_results = {
                'best_params': search.best_params_,
                'best_score': float(search.best_score_),
                'best_rmse': float(np.sqrt(-search.best_score_)),
                'cv_results': {
                    'mean_test_scores': search.cv_results_['mean_test_score'].tolist(),
                    'std_test_scores': search.cv_results_['std_test_score'].tolist(),
                    'params': search.cv_results_['params']
                }
            }
            
            self.logger.info(f"Hyperparameter tuning completed. Best RMSE: {tuning_results['best_rmse']:.6f}")
            
            return tuning_results
            
        except Exception as e:
            raise ModelTrainingError(f"Hyperparameter tuning failed: {str(e)}")
    
    def get_tree_predictions(self, X_test: np.ndarray) -> np.ndarray:
        """
        Get predictions from all individual trees.
        
        Args:
            X_test: Test features
            
        Returns:
            Array of shape (n_estimators, n_samples) with predictions from each tree
        """
        if self.model is None:
            raise ModelPredictionError("Model not trained")
        
        tree_predictions = np.array([
            tree.predict(X_test) for tree in self.model.estimators_
        ])
        
        return tree_predictions
    
    def analyze_prediction_stability(self, X_test: np.ndarray) -> Dict[str, Any]:
        """
        Analyze prediction stability across trees.
        
        Args:
            X_test: Test features
            
        Returns:
            Dictionary with stability analysis
        """
        tree_predictions = self.get_tree_predictions(X_test)
        
        # Calculate statistics across trees for each sample
        mean_predictions = np.mean(tree_predictions, axis=0)
        std_predictions = np.std(tree_predictions, axis=0)
        min_predictions = np.min(tree_predictions, axis=0)
        max_predictions = np.max(tree_predictions, axis=0)
        
        # Calculate coefficient of variation (std/mean)
        cv_predictions = std_predictions / (np.abs(mean_predictions) + 1e-8)
        
        analysis = {
            'mean_std': float(np.mean(std_predictions)),
            'mean_cv': float(np.mean(cv_predictions)),
            'max_std': float(np.max(std_predictions)),
            'max_cv': float(np.max(cv_predictions)),
            'prediction_ranges': (max_predictions - min_predictions).tolist(),
            'stability_scores': (1 / (1 + cv_predictions)).tolist()  # Higher is more stable
        }
        
        return analysis
    
    def partial_dependence_analysis(self, X: np.ndarray, feature_indices: list = None) -> Dict[str, Any]:
        """
        Perform partial dependence analysis for selected features.
        
        Args:
            X: Feature matrix
            feature_indices: Indices of features to analyze (default: top 5 important)
            
        Returns:
            Dictionary with partial dependence results
        """
        if self.model is None:
            raise ModelTrainingError("Model must be trained before partial dependence analysis")
        
        try:
            from sklearn.inspection import partial_dependence
            
            if feature_indices is None:
                # Use top 5 most important features
                importances = self.get_feature_importance()
                if importances:
                    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                    feature_indices = [i for i, (name, _) in enumerate(sorted_features[:5])]
                else:
                    feature_indices = list(range(min(5, X.shape[1])))
            
            pd_results = {}
            
            for idx in feature_indices:
                pd_result = partial_dependence(
                    self.model, X, features=[idx], kind='average'
                )
                
                feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f'feature_{idx}'
                
                pd_results[feature_name] = {
                    'values': pd_result['values'][0].tolist(),
                    'grid_values': pd_result['grid_values'][0].tolist()
                }
            
            return pd_results
            
        except ImportError:
            self.logger.warning("Scikit-learn version doesn't support partial_dependence")
            return {}
        except Exception as e:
            self.logger.warning(f"Partial dependence analysis failed: {str(e)}")
            return {}