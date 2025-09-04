import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import jarque_bera, normaltest
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

from ..models.qc_models import QCResult, QCLot, WestgardViolation, WestgardRuleEnum, QCStatusEnum

logger = logging.getLogger(__name__)

@dataclass
class QCDataPoint:
    """Enhanced QC data point with additional metadata"""
    value: float
    timestamp: datetime
    lot_number: str
    operator: str
    run_id: str
    shift: str
    duplicate_number: int = 1
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'lot_number': self.lot_number,
            'operator': self.operator,
            'run_id': self.run_id,
            'shift': self.shift,
            'duplicate_number': self.duplicate_number,
            'temperature': self.temperature,
            'humidity': self.humidity
        }

@dataclass
class WestgardViolationResult:
    """Enhanced violation result with detailed information"""
    rule_violated: WestgardRuleEnum
    severity: str
    message: str
    affected_points: List[int]  # Indices of affected data points
    pattern_description: str
    recommended_action: str
    confidence: float = 1.0  # Confidence in the violation (0-1)
    additional_context: Optional[Dict[str, Any]] = None

@dataclass
class QCStatistics:
    """Statistical summary of QC data"""
    n_points: int
    mean: float
    std_dev: float
    cv_percent: float
    min_value: float
    max_value: float
    q1: float
    median: float
    q3: float
    iqr: float
    outlier_count: int
    normality_p_value: float
    trend_slope: Optional[float] = None
    trend_p_value: Optional[float] = None

class EnhancedWestgardEngine:
    """Enhanced Westgard multi-rule QC system with ML-powered analysis"""
    
    def __init__(self, target_mean: float, target_sd: float, 
                 enable_ml_analysis: bool = True, confidence_threshold: float = 0.8):
        self.target_mean = target_mean
        self.target_sd = target_sd
        self.enable_ml_analysis = enable_ml_analysis
        self.confidence_threshold = confidence_threshold
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # ML models for anomaly detection
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self._initialize_ml_models()
        
    def _initialize_ml_models(self):
        """Initialize ML models for anomaly detection"""
        if self.enable_ml_analysis:
            self.isolation_forest = IsolationForest(
                contamination=0.1,  # Expected proportion of outliers
                random_state=42,
                n_estimators=100
            )
    
    async def evaluate_comprehensive(self, data_points: List[QCDataPoint]) -> Dict[str, Any]:
        """Comprehensive evaluation including traditional rules and ML analysis"""
        try:
            # Convert to values for statistical analysis
            values = [dp.value for dp in data_points]
            
            if len(values) == 0:
                return {
                    "status": "insufficient_data",
                    "message": "No data points provided",
                    "violations": [],
                    "statistics": None,
                    "ml_analysis": None
                }
            
            # Traditional Westgard rules evaluation
            violations = await self._evaluate_westgard_rules(data_points)
            
            # Statistical analysis
            statistics = await self._calculate_statistics(values)
            
            # ML-powered anomaly detection
            ml_analysis = None
            if self.enable_ml_analysis and len(values) >= 10:  # Minimum points for ML
                ml_analysis = await self._ml_anomaly_detection(data_points)
            
            # Trend analysis
            trend_analysis = await self._trend_analysis(data_points)
            
            # Overall status determination
            status = self._determine_overall_status(violations, ml_analysis)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                violations, statistics, ml_analysis, trend_analysis
            )
            
            return {
                "status": status,
                "violations": [self._violation_to_dict(v) for v in violations],
                "statistics": self._statistics_to_dict(statistics),
                "ml_analysis": ml_analysis,
                "trend_analysis": trend_analysis,
                "recommendations": recommendations,
                "evaluated_at": datetime.utcnow().isoformat(),
                "total_points": len(data_points)
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive QC evaluation: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "evaluated_at": datetime.utcnow().isoformat()
            }
    
    async def _evaluate_westgard_rules(self, data_points: List[QCDataPoint]) -> List[WestgardViolationResult]:
        """Evaluate all Westgard rules against data points"""
        violations = []
        
        # Rule 1-3s: One control exceeds ±3SD
        if violation := await self._check_13s(data_points):
            violations.append(violation)
        
        # Rule 2-2s: Two consecutive controls exceed ±2SD on same side
        if violation := await self._check_22s(data_points):
            violations.append(violation)
        
        # Rule R-4s: Range between controls exceeds 4SD
        if violation := await self._check_r4s(data_points):
            violations.append(violation)
        
        # Rule 4-1s: Four consecutive controls exceed ±1SD on same side
        if violation := await self._check_41s(data_points):
            violations.append(violation)
        
        # Rule 10x: Ten consecutive controls on same side of mean
        if violation := await self._check_10x(data_points):
            violations.append(violation)
        
        # Rule 7T: Seven consecutive controls with increasing/decreasing trend
        if violation := await self._check_7t(data_points):
            violations.append(violation)
        
        # Rule 9x: Nine consecutive controls on same side of mean
        if violation := await self._check_9x(data_points):
            violations.append(violation)
        
        # Rule 12x: Twelve consecutive controls on same side of mean
        if violation := await self._check_12x(data_points):
            violations.append(violation)
        
        return violations
    
    async def _check_13s(self, data_points: List[QCDataPoint]) -> Optional[WestgardViolationResult]:
        """Check 1-3s rule: One control exceeds ±3SD"""
        for i, dp in enumerate(data_points):
            z_score = (dp.value - self.target_mean) / self.target_sd
            if abs(z_score) > 3:
                return WestgardViolationResult(
                    rule_violated=WestgardRuleEnum.RULE_13S,
                    severity="reject",
                    message=f"Control exceeds 3 standard deviations (Z-score: {z_score:.2f})",
                    affected_points=[i],
                    pattern_description=f"Single point at {z_score:.2f}SD from target mean",
                    recommended_action="Stop testing, investigate and correct before resuming",
                    confidence=1.0,
                    additional_context={
                        "z_score": z_score,
                        "value": dp.value,
                        "target_mean": self.target_mean,
                        "target_sd": self.target_sd
                    }
                )
        return None
    
    async def _check_22s(self, data_points: List[QCDataPoint]) -> Optional[WestgardViolationResult]:
        """Check 2-2s rule: Two consecutive controls exceed ±2SD on same side"""
        if len(data_points) < 2:
            return None
            
        for i in range(len(data_points) - 1):
            z1 = (data_points[i].value - self.target_mean) / self.target_sd
            z2 = (data_points[i+1].value - self.target_mean) / self.target_sd
            
            if (z1 > 2 and z2 > 2) or (z1 < -2 and z2 < -2):
                return WestgardViolationResult(
                    rule_violated=WestgardRuleEnum.RULE_22S,
                    severity="reject",
                    message=f"Two consecutive controls exceed 2SD on same side (Z-scores: {z1:.2f}, {z2:.2f})",
                    affected_points=[i, i+1],
                    pattern_description=f"Consecutive points at {z1:.2f}SD and {z2:.2f}SD",
                    recommended_action="Stop testing, investigate systematic error",
                    confidence=0.95,
                    additional_context={
                        "z_scores": [z1, z2],
                        "values": [data_points[i].value, data_points[i+1].value]
                    }
                )
        return None
    
    async def _check_r4s(self, data_points: List[QCDataPoint]) -> Optional[WestgardViolationResult]:
        """Check R-4s rule: Range between consecutive controls exceeds 4SD"""
        if len(data_points) < 2:
            return None
        
        for i in range(len(data_points) - 1):
            range_val = abs(data_points[i].value - data_points[i+1].value)
            if range_val > 4 * self.target_sd:
                return WestgardViolationResult(
                    rule_violated=WestgardRuleEnum.RULE_R4S,
                    severity="reject",
                    message=f"Range between controls exceeds 4SD (Range: {range_val:.2f})",
                    affected_points=[i, i+1],
                    pattern_description=f"Range of {range_val:.2f} exceeds 4SD threshold",
                    recommended_action="Check for random error, repeat analysis",
                    confidence=0.9,
                    additional_context={
                        "range_value": range_val,
                        "threshold": 4 * self.target_sd,
                        "values": [data_points[i].value, data_points[i+1].value]
                    }
                )
        return None
    
    async def _check_41s(self, data_points: List[QCDataPoint]) -> Optional[WestgardViolationResult]:
        """Check 4-1s rule: Four consecutive controls exceed ±1SD on same side"""
        if len(data_points) < 4:
            return None
        
        for i in range(len(data_points) - 3):
            subset = data_points[i:i+4]
            z_scores = [(dp.value - self.target_mean) / self.target_sd for dp in subset]
            
            if all(z > 1 for z in z_scores) or all(z < -1 for z in z_scores):
                return WestgardViolationResult(
                    rule_violated=WestgardRuleEnum.RULE_41S,
                    severity="reject",
                    message=f"Four consecutive controls exceed 1SD on same side",
                    affected_points=list(range(i, i+4)),
                    pattern_description=f"Four consecutive points all beyond 1SD",
                    recommended_action="Investigate systematic shift or trend",
                    confidence=0.85,
                    additional_context={
                        "z_scores": z_scores,
                        "values": [dp.value for dp in subset]
                    }
                )
        return None
    
    async def _check_10x(self, data_points: List[QCDataPoint]) -> Optional[WestgardViolationResult]:
        """Check 10x rule: Ten consecutive controls on same side of mean"""
        if len(data_points) < 10:
            return None
        
        for i in range(len(data_points) - 9):
            subset = data_points[i:i+10]
            
            all_above = all(dp.value > self.target_mean for dp in subset)
            all_below = all(dp.value < self.target_mean for dp in subset)
            
            if all_above or all_below:
                side = "above" if all_above else "below"
                return WestgardViolationResult(
                    rule_violated=WestgardRuleEnum.RULE_10X,
                    severity="warning",
                    message=f"Ten consecutive controls {side} target mean",
                    affected_points=list(range(i, i+10)),
                    pattern_description=f"Ten consecutive points all {side} mean",
                    recommended_action="Check for systematic bias, consider recalibration",
                    confidence=0.8,
                    additional_context={
                        "side": side,
                        "values": [dp.value for dp in subset]
                    }
                )
        return None
    
    async def _check_7t(self, data_points: List[QCDataPoint]) -> Optional[WestgardViolationResult]:
        """Check 7T rule: Seven consecutive controls with increasing/decreasing trend"""
        if len(data_points) < 7:
            return None
        
        for i in range(len(data_points) - 6):
            subset = [dp.value for dp in data_points[i:i+7]]
            
            # Check for consistently increasing trend
            increasing = all(subset[j] < subset[j+1] for j in range(6))
            # Check for consistently decreasing trend  
            decreasing = all(subset[j] > subset[j+1] for j in range(6))
            
            if increasing or decreasing:
                trend = "increasing" if increasing else "decreasing"
                return WestgardViolationResult(
                    rule_violated=WestgardRuleEnum.RULE_7T,
                    severity="warning",
                    message=f"Seven consecutive controls with {trend} trend",
                    affected_points=list(range(i, i+7)),
                    pattern_description=f"Seven consecutive points showing {trend} trend",
                    recommended_action="Investigate trend cause, check reagent stability",
                    confidence=0.75,
                    additional_context={
                        "trend_type": trend,
                        "values": subset
                    }
                )
        return None
    
    async def _check_9x(self, data_points: List[QCDataPoint]) -> Optional[WestgardViolationResult]:
        """Check 9x rule: Nine consecutive controls on same side of mean"""
        if len(data_points) < 9:
            return None
        
        for i in range(len(data_points) - 8):
            subset = data_points[i:i+9]
            
            all_above = all(dp.value > self.target_mean for dp in subset)
            all_below = all(dp.value < self.target_mean for dp in subset)
            
            if all_above or all_below:
                side = "above" if all_above else "below"
                return WestgardViolationResult(
                    rule_violated=WestgardRuleEnum.RULE_9X,
                    severity="warning",
                    message=f"Nine consecutive controls {side} target mean",
                    affected_points=list(range(i, i+9)),
                    pattern_description=f"Nine consecutive points all {side} mean",
                    recommended_action="Investigate systematic shift",
                    confidence=0.82
                )
        return None
    
    async def _check_12x(self, data_points: List[QCDataPoint]) -> Optional[WestgardViolationResult]:
        """Check 12x rule: Twelve consecutive controls on same side of mean"""
        if len(data_points) < 12:
            return None
        
        for i in range(len(data_points) - 11):
            subset = data_points[i:i+12]
            
            all_above = all(dp.value > self.target_mean for dp in subset)
            all_below = all(dp.value < self.target_mean for dp in subset)
            
            if all_above or all_below:
                side = "above" if all_above else "below"
                return WestgardViolationResult(
                    rule_violated=WestgardRuleEnum.RULE_12X,
                    severity="reject",
                    message=f"Twelve consecutive controls {side} target mean",
                    affected_points=list(range(i, i+12)),
                    pattern_description=f"Twelve consecutive points all {side} mean",
                    recommended_action="Stop testing, investigate and correct systematic bias",
                    confidence=0.9
                )
        return None
    
    async def _calculate_statistics(self, values: List[float]) -> QCStatistics:
        """Calculate comprehensive statistics for QC data"""
        values_array = np.array(values)
        
        # Basic statistics
        n_points = len(values)
        mean = np.mean(values_array)
        std_dev = np.std(values_array, ddof=1)  # Sample standard deviation
        cv_percent = (std_dev / mean) * 100 if mean != 0 else 0
        
        # Quartiles and outliers
        q1, median, q3 = np.percentile(values_array, [25, 50, 75])
        iqr = q3 - q1
        
        # Outliers using IQR method
        outlier_threshold_low = q1 - 1.5 * iqr
        outlier_threshold_high = q3 + 1.5 * iqr
        outlier_count = np.sum((values_array < outlier_threshold_low) | 
                              (values_array > outlier_threshold_high))
        
        # Normality test
        try:
            if len(values) >= 8:  # Minimum for normality test
                _, normality_p_value = normaltest(values_array)
            else:
                normality_p_value = 1.0  # Assume normal for small samples
        except:
            normality_p_value = 1.0
        
        # Trend analysis
        trend_slope = None
        trend_p_value = None
        if len(values) >= 5:
            try:
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values_array)
                trend_slope = slope
                trend_p_value = p_value
            except:
                pass
        
        return QCStatistics(
            n_points=n_points,
            mean=mean,
            std_dev=std_dev,
            cv_percent=cv_percent,
            min_value=float(np.min(values_array)),
            max_value=float(np.max(values_array)),
            q1=q1,
            median=median,
            q3=q3,
            iqr=iqr,
            outlier_count=outlier_count,
            normality_p_value=normality_p_value,
            trend_slope=trend_slope,
            trend_p_value=trend_p_value
        )
    
    async def _ml_anomaly_detection(self, data_points: List[QCDataPoint]) -> Dict[str, Any]:
        """ML-powered anomaly detection using Isolation Forest"""
        try:
            # Prepare features for ML analysis
            features = []
            for dp in data_points:
                feature_vector = [
                    dp.value,
                    (dp.value - self.target_mean) / self.target_sd,  # Z-score
                    dp.timestamp.hour,  # Hour of day
                    dp.timestamp.weekday(),  # Day of week
                    dp.duplicate_number,
                    dp.temperature or 25.0,  # Default temperature if not provided
                    dp.humidity or 50.0  # Default humidity if not provided
                ]
                features.append(feature_vector)
            
            features_array = np.array(features)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Fit isolation forest
            anomaly_scores = self.isolation_forest.fit_predict(features_scaled)
            decision_scores = self.isolation_forest.decision_function(features_scaled)
            
            # Identify anomalies
            anomaly_indices = np.where(anomaly_scores == -1)[0].tolist()
            
            # Calculate confidence scores
            confidence_scores = []
            for score in decision_scores:
                # Convert decision function output to confidence (0-1)
                confidence = max(0, min(1, (score + 0.5) * 2))
                confidence_scores.append(confidence)
            
            return {
                "anomalies_detected": len(anomaly_indices),
                "anomaly_indices": anomaly_indices,
                "anomaly_scores": anomaly_scores.tolist(),
                "decision_scores": decision_scores.tolist(),
                "confidence_scores": confidence_scores,
                "contamination_rate": len(anomaly_indices) / len(data_points) * 100,
                "model_type": "isolation_forest",
                "features_used": ["value", "z_score", "hour", "weekday", "duplicate", "temperature", "humidity"]
            }
            
        except Exception as e:
            logger.error(f"Error in ML anomaly detection: {str(e)}")
            return {
                "error": str(e),
                "anomalies_detected": 0,
                "model_type": "isolation_forest"
            }
    
    async def _trend_analysis(self, data_points: List[QCDataPoint]) -> Dict[str, Any]:
        """Perform comprehensive trend analysis"""
        if len(data_points) < 5:
            return {"insufficient_data": True}
        
        values = [dp.value for dp in data_points]
        timestamps = [dp.timestamp for dp in data_points]
        
        # Convert timestamps to hours from first measurement
        time_hours = [(ts - timestamps[0]).total_seconds() / 3600 for ts in timestamps]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_hours, values)
        
        # Trend significance
        trend_significant = p_value < 0.05
        trend_strength = abs(r_value)
        
        # Trend direction
        if slope > 0 and trend_significant:
            trend_direction = "increasing"
        elif slope < 0 and trend_significant:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        # Seasonal analysis (if enough data points)
        seasonal_pattern = None
        if len(data_points) >= 20:
            seasonal_pattern = await self._detect_seasonal_pattern(data_points)
        
        return {
            "trend_direction": trend_direction,
            "slope": slope,
            "intercept": intercept,
            "correlation_coefficient": r_value,
            "p_value": p_value,
            "standard_error": std_err,
            "trend_significant": trend_significant,
            "trend_strength": trend_strength,
            "seasonal_pattern": seasonal_pattern,
            "time_period_hours": max(time_hours)
        }
    
    async def _detect_seasonal_pattern(self, data_points: List[QCDataPoint]) -> Optional[Dict[str, Any]]:
        """Detect seasonal patterns in QC data"""
        try:
            # Group by hour of day
            hourly_means = {}
            for dp in data_points:
                hour = dp.timestamp.hour
                if hour not in hourly_means:
                    hourly_means[hour] = []
                hourly_means[hour].append(dp.value)
            
            # Calculate mean for each hour
            hour_stats = {}
            for hour, values in hourly_means.items():
                if len(values) >= 2:
                    hour_stats[hour] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "count": len(values)
                    }
            
            if len(hour_stats) < 3:
                return None
            
            # Check for significant variation by hour
            hour_means = [stats["mean"] for stats in hour_stats.values()]
            variation_cv = np.std(hour_means) / np.mean(hour_means) * 100
            
            return {
                "hourly_variation_cv": variation_cv,
                "significant_hourly_pattern": variation_cv > 5,  # More than 5% CV
                "hour_statistics": hour_stats
            }
            
        except Exception as e:
            logger.error(f"Error in seasonal pattern detection: {str(e)}")
            return None
    
    def _determine_overall_status(self, violations: List[WestgardViolationResult], 
                                 ml_analysis: Optional[Dict[str, Any]]) -> str:
        """Determine overall QC status based on all analyses"""
        # Check for reject-level violations
        reject_violations = [v for v in violations if v.severity == "reject"]
        if reject_violations:
            return "out_of_control"
        
        # Check for warning-level violations
        warning_violations = [v for v in violations if v.severity == "warning"]
        if warning_violations:
            return "warning"
        
        # Check ML analysis for high anomaly rate
        if ml_analysis and ml_analysis.get("contamination_rate", 0) > 15:
            return "warning"
        
        return "in_control"
    
    async def _generate_recommendations(self, violations: List[WestgardViolationResult],
                                       statistics: QCStatistics, ml_analysis: Optional[Dict[str, Any]],
                                       trend_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on all analyses"""
        recommendations = []
        
        # Westgard rule violations
        if violations:
            for violation in violations:
                recommendations.append(violation.recommended_action)
        
        # Statistical recommendations
        if statistics.cv_percent > 10:
            recommendations.append("High CV detected - investigate precision issues")
        
        if statistics.outlier_count > 0:
            recommendations.append(f"Remove {statistics.outlier_count} outlier(s) and investigate causes")
        
        if statistics.normality_p_value < 0.05:
            recommendations.append("Non-normal distribution detected - review QC process")
        
        # Trend recommendations
        if trend_analysis.get("trend_significant"):
            direction = trend_analysis.get("trend_direction", "unknown")
            recommendations.append(f"Significant {direction} trend detected - investigate systematic drift")
        
        # ML recommendations
        if ml_analysis and ml_analysis.get("anomalies_detected", 0) > 0:
            count = ml_analysis["anomalies_detected"]
            recommendations.append(f"ML model detected {count} anomalous result(s) - review flagged points")
        
        # Default recommendation if no issues
        if not recommendations:
            recommendations.append("QC results are within acceptable limits - continue routine monitoring")
        
        return recommendations
    
    def _violation_to_dict(self, violation: WestgardViolationResult) -> Dict[str, Any]:
        """Convert violation result to dictionary"""
        return {
            "rule_violated": violation.rule_violated.value,
            "severity": violation.severity,
            "message": violation.message,
            "affected_points": violation.affected_points,
            "pattern_description": violation.pattern_description,
            "recommended_action": violation.recommended_action,
            "confidence": violation.confidence,
            "additional_context": violation.additional_context
        }
    
    def _statistics_to_dict(self, statistics: QCStatistics) -> Dict[str, Any]:
        """Convert statistics to dictionary"""
        return {
            "n_points": statistics.n_points,
            "mean": statistics.mean,
            "std_dev": statistics.std_dev,
            "cv_percent": statistics.cv_percent,
            "min_value": statistics.min_value,
            "max_value": statistics.max_value,
            "q1": statistics.q1,
            "median": statistics.median,
            "q3": statistics.q3,
            "iqr": statistics.iqr,
            "outlier_count": statistics.outlier_count,
            "normality_p_value": statistics.normality_p_value,
            "trend_slope": statistics.trend_slope,
            "trend_p_value": statistics.trend_p_value
        }
    
    async def validate_qc_lot_setup(self, qc_data: List[QCDataPoint], min_points: int = 20) -> Dict[str, Any]:
        """Validate QC lot setup and calculate initial statistics"""
        if len(qc_data) < min_points:
            return {
                "valid": False,
                "message": f"Insufficient data points. Need at least {min_points}, got {len(qc_data)}",
                "recommendations": [
                    f"Collect at least {min_points} QC measurements before establishing limits",
                    "Ensure QC measurements span multiple days and operators",
                    "Verify consistent storage and handling conditions"
                ]
            }
        
        values = [dp.value for dp in qc_data]
        statistics = await self._calculate_statistics(values)
        
        # Validation checks
        validation_results = {
            "sufficient_data": len(qc_data) >= min_points,
            "acceptable_cv": statistics.cv_percent <= 15,  # Typical QC CV limit
            "normal_distribution": statistics.normality_p_value > 0.05,
            "no_excessive_outliers": statistics.outlier_count / len(qc_data) <= 0.05
        }
        
        all_valid = all(validation_results.values())
        
        recommendations = []
        if not validation_results["acceptable_cv"]:
            recommendations.append(f"CV too high ({statistics.cv_percent:.1f}%) - investigate precision")
        if not validation_results["normal_distribution"]:
            recommendations.append("Non-normal distribution - review methodology")
        if not validation_results["no_excessive_outliers"]:
            recommendations.append("Too many outliers - investigate and remove invalid results")
        
        return {
            "valid": all_valid,
            "validation_results": validation_results,
            "calculated_mean": statistics.mean,
            "calculated_sd": statistics.std_dev,
            "calculated_cv": statistics.cv_percent,
            "statistics": self._statistics_to_dict(statistics),
            "recommendations": recommendations
        }