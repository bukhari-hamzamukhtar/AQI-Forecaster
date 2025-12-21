"""
Model Explainability Module for AQI Predictor
Implements SHAP and LIME explanations for time-series forecasting models.

This module provides:
1. SHAP explanations (global and local)
2. LIME explanations (local)
3. Time-series aware feature explanations
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Try importing SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

# Try importing LIME
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    LimeTabularExplainer = None


class ModelExplainer:
    """
    Comprehensive model explainer for AQI forecasting models.
    Handles SHAP and LIME explanations for time-series forecasting.
    """
    
    def __init__(self, model, scaler, feature_names: List[str], model_type: str = "auto", 
                 display_names: Optional[List[str]] = None):
        """
        Initialize the explainer.
        
        Args:
            model: Trained model (XGBoost, Random Forest, SVM, or Neural Network)
            scaler: Fitted StandardScaler
            feature_names: List of feature names (actual column names)
            model_type: Type of model ("xgb", "rf", "svm", "mlp", or "auto" to detect)
            display_names: Optional list of display names for visualization (defaults to feature_names)
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.display_names = display_names if display_names is not None else feature_names
        self.model_type = self._detect_model_type(model_type)
        self.shap_explainer = None
        self.lime_explainer = None
        
    def _detect_model_type(self, model_type: str) -> str:
        """Detect model type automatically if not specified."""
        if model_type != "auto":
            return model_type
        
        model_str = str(type(self.model)).lower()
        if 'xgboost' in model_str or 'xgb' in model_str:
            return "xgb"
        elif 'randomforest' in model_str or 'random_forest' in model_str:
            return "rf"
        elif 'svm' in model_str or 'svr' in model_str:
            return "svm"
        elif 'sequential' in model_str or 'keras' in model_str or 'tensorflow' in model_str:
            return "mlp"
        else:
            return "unknown"
    
    def _is_tree_based(self) -> bool:
        """Check if model is tree-based (supports TreeExplainer)."""
        return self.model_type in ["xgb", "rf"]
    
    def _prepare_model_predict(self):
        """Prepare model prediction function that works with SHAP/LIME."""
        def predict_wrapper(X):
            """Wrapper for model prediction that handles scaling."""
            if isinstance(X, pd.DataFrame):
                X = X.values
            # Scale the input
            X_scaled = self.scaler.transform(X)
            # Predict
            pred = self.model.predict(X_scaled)
            # Handle different output formats
            if hasattr(pred, 'flatten'):
                return pred.flatten()
            return pred
        return predict_wrapper
    
    def create_shap_explainer(self, training_data: Optional[pd.DataFrame] = None, 
                              sample_size: int = 100) -> bool:
        """
        Create SHAP explainer based on model type.
        
        Args:
            training_data: Training data for background (needed for non-tree models)
            sample_size: Number of samples to use for background data
            
        Returns:
            True if explainer was created successfully, False otherwise
        """
        if not SHAP_AVAILABLE:
            return False
        
        try:
            if self._is_tree_based():
                # Tree-based models: Use TreeExplainer (fast and exact)
                self.shap_explainer = shap.TreeExplainer(self.model)
                return True
            else:
                # Non-tree models: Use KernelExplainer (slower but works for any model)
                if training_data is None:
                    raise ValueError("Training data is required for non-tree models to create SHAP explainer")
                
                # Prepare background data
                if isinstance(training_data, pd.DataFrame):
                    # Select relevant columns
                    cols = [col for col in self.feature_names if col in training_data.columns]
                    if len(cols) != len(self.feature_names):
                        missing = set(self.feature_names) - set(cols)
                        raise ValueError(f"Missing features in training data: {missing}")
                    background = training_data[cols].values
                else:
                    background = training_data
                
                if len(background) == 0:
                    raise ValueError("Training data is empty")
                
                # Sample background data
                if len(background) > sample_size:
                    background = background[:sample_size]
                
                # Scale background data
                background_scaled = self.scaler.transform(background)
                
                # Create explainer with prediction wrapper
                predict_fn = self._prepare_model_predict()
                self.shap_explainer = shap.KernelExplainer(
                    predict_fn,
                    background_scaled,
                    feature_names=self.feature_names
                )
                return True
        except Exception as e:
            import traceback
            error_msg = f"Error creating SHAP explainer: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return False
    
    def explain_shap_local(self, input_data: np.ndarray, 
                          plot: bool = True) -> Tuple[Optional[np.ndarray], Optional[go.Figure]]:
        """
        Generate local SHAP explanation for a single prediction.
        
        Args:
            input_data: Single input sample (1D array or 2D array with 1 row)
            plot: Whether to create a plotly visualization
            
        Returns:
            Tuple of (shap_values, plotly_figure)
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return None, None
        
        try:
            # Ensure input is 2D
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            
            # Scale input
            input_scaled = self.scaler.transform(input_data)
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(input_scaled)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For multi-output, take first
            if shap_values.ndim > 1:
                shap_values = shap_values[0]  # For single prediction, take first row
            
            # Create visualization
            fig = None
            if plot:
                fig = self._plot_shap_local(shap_values, input_scaled[0])
            
            return shap_values, fig
        except Exception as e:
            print(f"Error calculating local SHAP values: {e}")
            return None, None
    
    def explain_shap_global(self, training_data: pd.DataFrame, 
                           sample_size: int = 100,
                           plot: bool = True) -> Tuple[Optional[np.ndarray], Optional[go.Figure]]:
        """
        Generate global SHAP explanation (summary plot) for the model.
        
        Args:
            training_data: Training data to explain
            sample_size: Number of samples to use (for speed)
            plot: Whether to create a plotly visualization
            
        Returns:
            Tuple of (shap_values_array, plotly_figure)
        """
        if not SHAP_AVAILABLE:
            raise ValueError("SHAP is not available. Please install with: pip install shap")
        
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized. Call create_shap_explainer() first.")
        
        try:
            # Prepare data
            if isinstance(training_data, pd.DataFrame):
                cols = [col for col in self.feature_names if col in training_data.columns]
                if len(cols) != len(self.feature_names):
                    missing = set(self.feature_names) - set(cols)
                    raise ValueError(f"Missing features in training data: {missing}")
                X = training_data[cols].values
            else:
                X = training_data
            
            if len(X) == 0:
                raise ValueError("Training data is empty")
            
            # Sample data for speed
            if len(X) > sample_size:
                indices = np.random.choice(len(X), sample_size, replace=False)
                X = X[indices]
            
            # Scale data
            X_scaled = self.scaler.transform(X)
            
            # Calculate SHAP values for all samples
            shap_values = self.shap_explainer.shap_values(X_scaled)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Ensure shap_values is 2D array
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)
            
            # Create visualization
            fig = None
            if plot:
                fig = self._plot_shap_global(shap_values, X_scaled)
            
            return shap_values, fig
        except Exception as e:
            import traceback
            error_msg = f"Error calculating global SHAP values: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            raise RuntimeError(f"Failed to generate global SHAP explanation: {str(e)}") from e
    
    def _plot_shap_local(self, shap_values: np.ndarray, 
                        input_values: np.ndarray) -> go.Figure:
        """Create a local SHAP force plot visualization."""
        # Calculate base value (expected value)
        base_value = self.shap_explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]
        
        # Create waterfall-style plot
        fig = go.Figure()
        
        # Sort features by absolute SHAP value
        feature_importance = np.abs(shap_values)
        sorted_indices = np.argsort(feature_importance)[::-1]
        
        # Prepare data for plotting (use display names)
        sorted_features = [self.display_names[i] for i in sorted_indices]
        sorted_shap = shap_values[sorted_indices]
        sorted_input = input_values[sorted_indices]
        
        # Color bars: red for positive (increases prediction), blue for negative
        colors = ['#ff4444' if v > 0 else '#4444ff' for v in sorted_shap]
        
        # Create bar chart
        fig.add_trace(go.Bar(
            x=sorted_features,
            y=sorted_shap,
            marker_color=colors,
            text=[f"{v:+.3f}<br>(val: {sorted_input[i]:.2f})" 
                  for i, v in enumerate(sorted_shap)],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                         'SHAP Value: %{y:.3f}<br>' +
                         'Feature Value: %{text}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="SHAP Local Explanation - Feature Impact on This Prediction",
            xaxis_title="Features (sorted by importance)",
            yaxis_title="SHAP Value (Impact on Prediction)",
            height=500,
            showlegend=False,
            xaxis={'tickangle': -45}
        )
        
        # Add annotation for base value
        fig.add_annotation(
            x=0.5,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"Base Value (Expected): {base_value:.2f}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig
    
    def _plot_shap_global(self, shap_values: np.ndarray, 
                         input_data: np.ndarray) -> go.Figure:
        """Create a global SHAP summary plot visualization."""
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Sort features by importance (use display names)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        sorted_features = [self.display_names[i] for i in sorted_indices]
        sorted_importance = mean_abs_shap[sorted_indices]
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=sorted_features,
            y=sorted_importance,
            marker_color='steelblue',
            text=[f"{v:.3f}" for v in sorted_importance],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="SHAP Global Summary - Overall Feature Importance",
            xaxis_title="Features (sorted by importance)",
            yaxis_title="Mean |SHAP Value|",
            height=500,
            showlegend=False,
            xaxis={'tickangle': -45}
        )
        
        return fig
    
    def create_lime_explainer(self, training_data: pd.DataFrame, 
                             sample_size: int = 1000) -> bool:
        """
        Create LIME explainer for local explanations.
        
        Args:
            training_data: Training data for LIME
            sample_size: Maximum number of samples to use
            
        Returns:
            True if explainer was created successfully, False otherwise
        """
        if not LIME_AVAILABLE:
            raise ValueError("LIME is not available. Please install with: pip install lime")
        
        if training_data is None:
            raise ValueError("Training data is required to create LIME explainer")
        
        try:
            # Prepare training data
            if isinstance(training_data, pd.DataFrame):
                cols = [col for col in self.feature_names if col in training_data.columns]
                if len(cols) != len(self.feature_names):
                    missing = set(self.feature_names) - set(cols)
                    raise ValueError(f"Missing features in training data: {missing}")
                X_train = training_data[cols].values
            else:
                X_train = training_data
            
            if len(X_train) == 0:
                raise ValueError("Training data is empty")
            
            # Sample data if too large
            if len(X_train) > sample_size:
                indices = np.random.choice(len(X_train), sample_size, replace=False)
                X_train = X_train[indices]
            
            # Scale training data
            X_train_scaled = self.scaler.transform(X_train)
            
            # Ensure feature names are all strings (LIME can have issues with None or non-string names)
            feature_names_clean = [str(f) if f is not None else f"feature_{i}" 
                                  for i, f in enumerate(self.feature_names)]
            
            # Create LIME explainer
            # Disable discretization to avoid scipy truncnorm errors with low-variance features
            # discretize_continuous=False prevents the domain error in scipy.stats.truncnorm
            self.lime_explainer = LimeTabularExplainer(
                X_train_scaled,
                feature_names=feature_names_clean,
                mode='regression',
                discretize_continuous=False  # Disable to avoid scipy truncnorm domain errors
            )
            return True
        except Exception as e:
            import traceback
            error_msg = f"Error creating LIME explainer: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return False
    
    def explain_lime_local(self, input_data: np.ndarray, 
                          num_features: int = 10,
                          plot: bool = True) -> Tuple[Optional[Dict], Optional[go.Figure]]:
        """
        Generate local LIME explanation for a single prediction.
        
        Args:
            input_data: Single input sample
            num_features: Number of top features to show
            plot: Whether to create a plotly visualization
            
        Returns:
            Tuple of (explanation_dict, plotly_figure)
        """
        if not LIME_AVAILABLE:
            raise ValueError("LIME is not available. Please install with: pip install lime")
        
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized. Call create_lime_explainer() first.")
        
        try:
            # Ensure input is 2D
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            
            # Validate input shape
            if input_data.shape[1] != len(self.feature_names):
                raise ValueError(f"Input shape mismatch: expected {len(self.feature_names)} features, got {input_data.shape[1]}")
            
            # Scale input
            input_scaled = self.scaler.transform(input_data)
            
            # Prepare prediction function
            predict_fn = self._prepare_model_predict()
            
            # Generate explanation with error handling
            try:
                explanation = self.lime_explainer.explain_instance(
                    input_scaled[0],
                    predict_fn,
                    num_features=min(num_features, len(self.feature_names)),
                    top_labels=1
                )
            except Exception as explain_error:
                # LIME can fail with certain model types or data
                raise RuntimeError(f"LIME explain_instance failed: {str(explain_error)}. "
                                 f"This might happen with certain model types or if the prediction function fails.") from explain_error
            
            # Extract explanation data safely
            try:
                explanation_list = explanation.as_list()
                if not explanation_list or len(explanation_list) == 0:
                    raise ValueError("LIME explanation returned empty list")
                
                # Ensure all feature names and values are valid
                explanation_dict = {
                    'features': [str(item[0]) if item[0] is not None else f"feature_{i}" 
                                for i, item in enumerate(explanation_list)],
                    'values': [float(item[1]) if item[1] is not None else 0.0 
                              for item in explanation_list],
                    'intercept': None,
                    'prediction': None
                }
                
                # Safely extract intercept and prediction
                if hasattr(explanation, 'intercept'):
                    try:
                        if isinstance(explanation.intercept, (list, np.ndarray)) and len(explanation.intercept) > 0:
                            explanation_dict['intercept'] = float(explanation.intercept[0])
                        elif not isinstance(explanation.intercept, (list, np.ndarray)):
                            explanation_dict['intercept'] = float(explanation.intercept)
                    except (ValueError, TypeError, IndexError):
                        pass  # Intercept extraction failed, keep as None
                
                if hasattr(explanation, 'prediction'):
                    try:
                        if isinstance(explanation.prediction, (list, np.ndarray)) and len(explanation.prediction) > 0:
                            explanation_dict['prediction'] = float(explanation.prediction[0])
                        elif not isinstance(explanation.prediction, (list, np.ndarray)):
                            explanation_dict['prediction'] = float(explanation.prediction)
                    except (ValueError, TypeError, IndexError):
                        pass  # Prediction extraction failed, keep as None
                        
            except Exception as extract_error:
                raise RuntimeError(f"Failed to extract LIME explanation data: {str(extract_error)}") from extract_error
            
            # Create visualization
            fig = None
            if plot:
                fig = self._plot_lime_local(explanation_dict)
            
            return explanation_dict, fig
        except Exception as e:
            import traceback
            error_msg = f"Error calculating LIME explanation: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            raise RuntimeError(f"Failed to generate LIME explanation: {str(e)}") from e
    
    def _plot_lime_local(self, explanation_dict: Dict) -> go.Figure:
        """Create a local LIME explanation visualization."""
        features = explanation_dict['features']
        values = explanation_dict['values']
        
        # Ensure we have valid data
        if not features or not values or len(features) != len(values):
            raise ValueError("Invalid explanation data for plotting")
        
        # Sort by absolute value
        # Map feature names to display names
        # LIME might return feature names as strings, so we need to match them
        feature_to_display = dict(zip(self.feature_names, self.display_names))
        # Also create a mapping from string versions
        feature_to_display_str = {str(k): v for k, v in feature_to_display.items()}
        
        # Filter out None values and ensure all values are numeric
        valid_data = [(f, v) for f, v in zip(features, values) 
                     if f is not None and v is not None]
        
        if not valid_data:
            raise ValueError("No valid feature data for plotting")
        
        sorted_data = sorted(valid_data, key=lambda x: abs(float(x[1])), reverse=True)
        sorted_features = []
        sorted_values = []
        
        for feat_name, feat_value in sorted_data:
            # Try to map to display name
            feat_str = str(feat_name)
            if feat_name in feature_to_display:
                sorted_features.append(feature_to_display[feat_name])
            elif feat_str in feature_to_display_str:
                sorted_features.append(feature_to_display_str[feat_str])
            else:
                sorted_features.append(feat_str)
            sorted_values.append(float(feat_value))
        
        # Color bars: red for positive, blue for negative
        colors = ['#ff4444' if v > 0 else '#4444ff' for v in sorted_values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=sorted_features,
            y=sorted_values,
            marker_color=colors,
            text=[f"{v:+.3f}" for v in sorted_values],
            textposition='outside'
        ))
        
        intercept = explanation_dict.get('intercept')
        prediction = explanation_dict.get('prediction')
        
        title = "LIME Local Explanation - Feature Impact on This Prediction"
        if intercept is not None and prediction is not None:
            try:
                title += f"<br><sub>Base: {float(intercept):.2f} | Prediction: {float(prediction):.2f}</sub>"
            except (ValueError, TypeError):
                pass  # Skip title addition if formatting fails
        
        fig.update_layout(
            title=title,
            xaxis_title="Features (sorted by importance)",
            yaxis_title="LIME Value (Impact on Prediction)",
            height=500,
            showlegend=False,
            xaxis={'tickangle': -45}
        )
        
        return fig
    
    def get_feature_importance_summary(self, shap_values: np.ndarray) -> pd.DataFrame:
        """
        Create a summary DataFrame of feature importance from SHAP values.
        
        Args:
            shap_values: SHAP values array (can be local or global)
            
        Returns:
            DataFrame with feature importance metrics
        """
        if shap_values is None:
            return pd.DataFrame()
        
        # Handle both 1D (local) and 2D (global) arrays
        if shap_values.ndim == 1:
            mean_abs = np.abs(shap_values)
            mean_shap = shap_values
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)
            mean_shap = shap_values.mean(axis=0)
        
        # Create summary DataFrame (use display names)
        summary = pd.DataFrame({
            'Feature': self.display_names,
            'Mean |SHAP|': mean_abs,
            'Mean SHAP': mean_shap,
            'Rank': np.argsort(mean_abs)[::-1] + 1
        }).sort_values('Mean |SHAP|', ascending=False)
        
        return summary


def create_explainer(model, scaler, feature_names: List[str], 
                    model_type: str = "auto", display_names: Optional[List[str]] = None) -> ModelExplainer:
    """
    Convenience function to create a ModelExplainer instance.
    
    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        feature_names: List of feature names (actual column names)
        model_type: Type of model (optional, will auto-detect)
        display_names: Optional list of display names for visualization
        
    Returns:
        ModelExplainer instance
    """
    return ModelExplainer(model, scaler, feature_names, model_type, display_names)

