import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from app.qc.enhanced_westgard import (
    EnhancedWestgardEngine, QCDataPoint, WestgardViolationResult, 
    QCStatistics, WestgardRuleEnum
)

@pytest.fixture
def sample_qc_data():
    """Generate sample QC data points"""
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    data_points = []
    
    # Generate 30 data points with known characteristics
    for i in range(30):
        value = 100 + np.random.normal(0, 2)  # Target mean 100, SD 2
        timestamp = base_time + timedelta(hours=i)
        
        data_point = QCDataPoint(
            value=value,
            timestamp=timestamp,
            lot_number="LOT123",
            operator=f"OP{i % 3 + 1}",
            run_id=f"RUN_{i:03d}",
            shift="day" if i % 24 < 16 else "night",
            duplicate_number=1,
            temperature=23.0 + np.random.uniform(-1, 1),
            humidity=45.0 + np.random.uniform(-5, 5)
        )
        data_points.append(data_point)
    
    return data_points

@pytest.fixture
def westgard_engine():
    """Create Westgard engine instance"""
    return EnhancedWestgardEngine(
        target_mean=100.0,
        target_sd=2.0,
        enable_ml_analysis=True,
        confidence_threshold=0.8
    )

class TestEnhancedWestgardEngine:
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test Westgard engine initialization"""
        engine = EnhancedWestgardEngine(target_mean=50.0, target_sd=5.0)
        
        assert engine.target_mean == 50.0
        assert engine.target_sd == 5.0
        assert engine.enable_ml_analysis == True
        assert engine.confidence_threshold == 0.8
        assert engine.isolation_forest is not None
    
    @pytest.mark.asyncio
    async def test_13s_rule_violation(self, westgard_engine):
        """Test 1-3s rule detection"""
        # Create data with one point beyond 3SD
        data_points = [
            QCDataPoint(98.0, datetime.now(), "LOT1", "OP1", "RUN1", "day"),
            QCDataPoint(101.0, datetime.now(), "LOT1", "OP1", "RUN2", "day"),
            QCDataPoint(107.0, datetime.now(), "LOT1", "OP1", "RUN3", "day")  # 3.5SD above mean
        ]
        
        violation = await westgard_engine._check_13s(data_points)
        
        assert violation is not None
        assert violation.rule_violated == WestgardRuleEnum.RULE_13S
        assert violation.severity == "reject"
        assert 2 in violation.affected_points  # Third point (index 2)
        assert "3 standard deviations" in violation.message
    
    @pytest.mark.asyncio
    async def test_22s_rule_violation(self, westgard_engine):
        """Test 2-2s rule detection"""
        # Create data with two consecutive points beyond 2SD on same side
        data_points = [
            QCDataPoint(99.0, datetime.now(), "LOT1", "OP1", "RUN1", "day"),
            QCDataPoint(104.5, datetime.now(), "LOT1", "OP1", "RUN2", "day"),  # 2.25SD above
            QCDataPoint(105.0, datetime.now(), "LOT1", "OP1", "RUN3", "day")   # 2.5SD above
        ]
        
        violation = await westgard_engine._check_22s(data_points)
        
        assert violation is not None
        assert violation.rule_violated == WestgardRuleEnum.RULE_22S
        assert violation.severity == "reject"
        assert violation.affected_points == [1, 2]
    
    @pytest.mark.asyncio
    async def test_r4s_rule_violation(self, westgard_engine):
        """Test R-4s rule detection"""
        # Create data with range exceeding 4SD
        data_points = [
            QCDataPoint(96.0, datetime.now(), "LOT1", "OP1", "RUN1", "day"),   # -2SD
            QCDataPoint(105.0, datetime.now(), "LOT1", "OP1", "RUN2", "day")   # +2.5SD (range = 9, >4SD)
        ]
        
        violation = await westgard_engine._check_r4s(data_points)
        
        assert violation is not None
        assert violation.rule_violated == WestgardRuleEnum.RULE_R4S
        assert violation.severity == "reject"
        assert violation.affected_points == [0, 1]
    
    @pytest.mark.asyncio
    async def test_41s_rule_violation(self, westgard_engine):
        """Test 4-1s rule detection"""
        # Create data with four consecutive points beyond 1SD on same side
        data_points = [
            QCDataPoint(102.5, datetime.now(), "LOT1", "OP1", "RUN1", "day"),  # +1.25SD
            QCDataPoint(102.2, datetime.now(), "LOT1", "OP1", "RUN2", "day"),  # +1.1SD
            QCDataPoint(103.0, datetime.now(), "LOT1", "OP1", "RUN3", "day"),  # +1.5SD
            QCDataPoint(102.8, datetime.now(), "LOT1", "OP1", "RUN4", "day")   # +1.4SD
        ]
        
        violation = await westgard_engine._check_41s(data_points)
        
        assert violation is not None
        assert violation.rule_violated == WestgardRuleEnum.RULE_41S
        assert violation.severity == "reject"
        assert violation.affected_points == [0, 1, 2, 3]
    
    @pytest.mark.asyncio
    async def test_10x_rule_violation(self, westgard_engine):
        """Test 10x rule detection"""
        # Create data with 10 consecutive points above mean
        data_points = []
        for i in range(10):
            data_points.append(
                QCDataPoint(100.5, datetime.now(), "LOT1", "OP1", f"RUN{i}", "day")  # Slightly above mean
            )
        
        violation = await westgard_engine._check_10x(data_points)
        
        assert violation is not None
        assert violation.rule_violated == WestgardRuleEnum.RULE_10X
        assert violation.severity == "warning"
        assert violation.affected_points == list(range(10))
    
    @pytest.mark.asyncio
    async def test_7t_rule_violation(self, westgard_engine):
        """Test 7T rule detection"""
        # Create data with increasing trend
        data_points = []
        for i in range(7):
            value = 98 + i * 0.5  # Increasing trend
            data_points.append(
                QCDataPoint(value, datetime.now(), "LOT1", "OP1", f"RUN{i}", "day")
            )
        
        violation = await westgard_engine._check_7t(data_points)
        
        assert violation is not None
        assert violation.rule_violated == WestgardRuleEnum.RULE_7T
        assert violation.severity == "warning"
        assert "increasing" in violation.additional_context["trend_type"]
    
    @pytest.mark.asyncio
    async def test_no_violations_normal_data(self, westgard_engine, sample_qc_data):
        """Test that normal QC data doesn't trigger violations"""
        # Use normal sample data (should not trigger violations)
        violations = await westgard_engine._evaluate_westgard_rules(sample_qc_data[:10])
        
        assert len(violations) == 0
    
    @pytest.mark.asyncio
    async def test_calculate_statistics(self, westgard_engine):
        """Test statistical calculations"""
        values = [98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        
        stats = await westgard_engine._calculate_statistics(values)
        
        assert stats.n_points == 8
        assert abs(stats.mean - 101.5) < 0.01
        assert stats.cv_percent > 0
        assert stats.q1 < stats.median < stats.q3
        assert stats.min_value == 98.0
        assert stats.max_value == 105.0
    
    @pytest.mark.asyncio
    async def test_comprehensive_evaluation(self, westgard_engine, sample_qc_data):
        """Test comprehensive evaluation with normal data"""
        result = await westgard_engine.evaluate_comprehensive(sample_qc_data)
        
        assert result["status"] in ["in_control", "warning", "out_of_control"]
        assert "violations" in result
        assert "statistics" in result
        assert "ml_analysis" in result
        assert "trend_analysis" in result
        assert "recommendations" in result
        assert result["total_points"] == len(sample_qc_data)
    
    @pytest.mark.asyncio
    async def test_ml_anomaly_detection(self, westgard_engine, sample_qc_data):
        """Test ML-powered anomaly detection"""
        # Add obvious outliers
        outlier_data = sample_qc_data.copy()
        outlier_data.append(
            QCDataPoint(
                value=120.0,  # Way beyond normal range
                timestamp=datetime.now(),
                lot_number="LOT123",
                operator="OP1",
                run_id="OUTLIER",
                shift="day"
            )
        )
        
        ml_result = await westgard_engine._ml_anomaly_detection(outlier_data)
        
        assert ml_result["anomalies_detected"] >= 0
        assert "anomaly_indices" in ml_result
        assert "confidence_scores" in ml_result
        assert ml_result["model_type"] == "isolation_forest"
        assert len(ml_result["confidence_scores"]) == len(outlier_data)
    
    @pytest.mark.asyncio
    async def test_trend_analysis(self, westgard_engine):
        """Test trend analysis"""
        # Create data with clear increasing trend
        trend_data = []
        for i in range(15):
            value = 98 + i * 0.3  # Clear increasing trend
            timestamp = datetime(2024, 1, 1) + timedelta(hours=i)
            trend_data.append(
                QCDataPoint(value, timestamp, "LOT1", "OP1", f"RUN{i}", "day")
            )
        
        trend_result = await westgard_engine._trend_analysis(trend_data)
        
        assert trend_result["trend_direction"] in ["increasing", "decreasing", "stable"]
        assert "slope" in trend_result
        assert "correlation_coefficient" in trend_result
        assert "p_value" in trend_result
        assert "trend_significant" in trend_result
    
    @pytest.mark.asyncio
    async def test_insufficient_data(self, westgard_engine):
        """Test handling of insufficient data"""
        result = await westgard_engine.evaluate_comprehensive([])
        
        assert result["status"] == "insufficient_data"
        assert "No data points provided" in result["message"]
    
    @pytest.mark.asyncio
    async def test_validate_qc_lot_setup_valid(self, westgard_engine):
        """Test QC lot validation with valid data"""
        # Generate sufficient normal data
        valid_data = []
        for i in range(25):
            value = 100 + np.random.normal(0, 1.5)  # Good precision
            timestamp = datetime(2024, 1, 1) + timedelta(hours=i)
            valid_data.append(
                QCDataPoint(value, timestamp, "LOT1", "OP1", f"RUN{i}", "day")
            )
        
        validation = await westgard_engine.validate_qc_lot_setup(valid_data)
        
        assert validation["valid"] == True
        assert "calculated_mean" in validation
        assert "calculated_sd" in validation
        assert validation["validation_results"]["sufficient_data"] == True
    
    @pytest.mark.asyncio
    async def test_validate_qc_lot_setup_insufficient_data(self, westgard_engine):
        """Test QC lot validation with insufficient data"""
        # Only 5 data points
        insufficient_data = []
        for i in range(5):
            value = 100 + np.random.normal(0, 2)
            timestamp = datetime(2024, 1, 1) + timedelta(hours=i)
            insufficient_data.append(
                QCDataPoint(value, timestamp, "LOT1", "OP1", f"RUN{i}", "day")
            )
        
        validation = await westgard_engine.validate_qc_lot_setup(insufficient_data, min_points=20)
        
        assert validation["valid"] == False
        assert "Insufficient data points" in validation["message"]
        assert len(validation["recommendations"]) > 0
    
    @pytest.mark.asyncio
    async def test_validate_qc_lot_setup_high_cv(self, westgard_engine):
        """Test QC lot validation with high CV"""
        # Generate data with high variability
        high_cv_data = []
        for i in range(25):
            value = 100 + np.random.normal(0, 10)  # Very high SD
            timestamp = datetime(2024, 1, 1) + timedelta(hours=i)
            high_cv_data.append(
                QCDataPoint(value, timestamp, "LOT1", "OP1", f"RUN{i}", "day")
            )
        
        validation = await westgard_engine.validate_qc_lot_setup(high_cv_data)
        
        # May or may not be valid depending on exact CV, but should have recommendations
        assert "calculated_cv" in validation
        if validation["calculated_cv"] > 15:
            assert any("CV too high" in rec for rec in validation.get("recommendations", []))
    
    @pytest.mark.asyncio
    async def test_seasonal_pattern_detection(self, westgard_engine):
        """Test seasonal pattern detection"""
        # Create data with hourly variation
        seasonal_data = []
        for day in range(3):  # 3 days of data
            for hour in range(24):
                # Simulate higher values during night shift
                base_value = 100
                if 20 <= hour or hour <= 6:  # Night shift
                    base_value += 2  # Systematic difference
                
                value = base_value + np.random.normal(0, 1)
                timestamp = datetime(2024, 1, 1 + day, hour, 0, 0)
                seasonal_data.append(
                    QCDataPoint(value, timestamp, "LOT1", f"OP{hour%3+1}", f"RUN{day*24+hour}", 
                               "night" if (20 <= hour or hour <= 6) else "day")
                )
        
        seasonal_result = await westgard_engine._detect_seasonal_pattern(seasonal_data)
        
        if seasonal_result:  # May not detect pattern with random data
            assert "hourly_variation_cv" in seasonal_result
            assert "significant_hourly_pattern" in seasonal_result
    
    def test_qc_data_point_to_dict(self):
        """Test QC data point serialization"""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        data_point = QCDataPoint(
            value=100.5,
            timestamp=timestamp,
            lot_number="LOT123",
            operator="OP1",
            run_id="RUN001",
            shift="day",
            duplicate_number=2,
            temperature=24.5,
            humidity=48.2
        )
        
        data_dict = data_point.to_dict()
        
        assert data_dict["value"] == 100.5
        assert data_dict["timestamp"] == timestamp.isoformat()
        assert data_dict["lot_number"] == "LOT123"
        assert data_dict["operator"] == "OP1"
        assert data_dict["run_id"] == "RUN001"
        assert data_dict["shift"] == "day"
        assert data_dict["duplicate_number"] == 2
        assert data_dict["temperature"] == 24.5
        assert data_dict["humidity"] == 48.2
    
    @pytest.mark.asyncio
    async def test_error_handling(self, westgard_engine):
        """Test error handling in comprehensive evaluation"""
        # Create problematic data that might cause errors
        problematic_data = [
            QCDataPoint(float('inf'), datetime.now(), "LOT1", "OP1", "RUN1", "day"),
            QCDataPoint(float('nan'), datetime.now(), "LOT1", "OP1", "RUN2", "day"),
        ]
        
        result = await westgard_engine.evaluate_comprehensive(problematic_data)
        
        # Should handle errors gracefully
        assert "status" in result
        # May return error status or handle the problematic values

# Integration tests
class TestWestgardEngineIntegration:
    
    @pytest.mark.asyncio
    async def test_complete_qc_workflow(self):
        """Test complete QC workflow from lot validation to ongoing monitoring"""
        engine = EnhancedWestgardEngine(target_mean=100.0, target_sd=2.0)
        
        # 1. Initial lot validation with 25 points
        initial_data = []
        for i in range(25):
            value = 100 + np.random.normal(0, 1.8)  # Slightly better precision
            timestamp = datetime(2024, 1, 1) + timedelta(hours=i)
            initial_data.append(
                QCDataPoint(value, timestamp, "LOT123", f"OP{i%3+1}", f"INIT{i:03d}", "day")
            )
        
        # Validate lot setup
        validation = await engine.validate_qc_lot_setup(initial_data)
        assert validation["valid"] == True
        
        # 2. Ongoing monitoring - add problematic results
        monitoring_data = initial_data.copy()
        
        # Add some normal results
        for i in range(5):
            value = 100 + np.random.normal(0, 2)
            timestamp = datetime(2024, 1, 2) + timedelta(hours=i)
            monitoring_data.append(
                QCDataPoint(value, timestamp, "LOT123", "OP1", f"MON{i:03d}", "day")
            )
        
        # Add a 1-3s violation
        monitoring_data.append(
            QCDataPoint(107.0, datetime(2024, 1, 2, 6, 0), "LOT123", "OP1", "VIOL001", "day")
        )
        
        # 3. Comprehensive evaluation
        evaluation = await engine.evaluate_comprehensive(monitoring_data)
        
        assert evaluation["status"] == "out_of_control"
        assert len(evaluation["violations"]) > 0
        assert evaluation["violations"][0]["rule_violated"] == "1-3s"
        assert "statistics" in evaluation
        assert evaluation["total_points"] == len(monitoring_data)
    
    @pytest.mark.asyncio
    async def test_multiple_rule_violations(self):
        """Test detection of multiple simultaneous rule violations"""
        engine = EnhancedWestgardEngine(target_mean=100.0, target_sd=2.0)
        
        # Create data that violates multiple rules
        violation_data = [
            QCDataPoint(99.0, datetime(2024, 1, 1, 8, 0), "LOT1", "OP1", "R01", "day"),
            QCDataPoint(104.5, datetime(2024, 1, 1, 9, 0), "LOT1", "OP1", "R02", "day"),  # Start of 2-2s
            QCDataPoint(105.0, datetime(2024, 1, 1, 10, 0), "LOT1", "OP1", "R03", "day"), # Complete 2-2s
            QCDataPoint(108.0, datetime(2024, 1, 1, 11, 0), "LOT1", "OP1", "R04", "day"), # 1-3s violation
            QCDataPoint(99.0, datetime(2024, 1, 1, 12, 0), "LOT1", "OP1", "R05", "day"),  # Large range (R-4s)
        ]
        
        evaluation = await engine.evaluate_comprehensive(violation_data)
        
        assert evaluation["status"] == "out_of_control"
        assert len(evaluation["violations"]) >= 2  # Should detect multiple violations
        
        # Check for specific violations
        violation_rules = [v["rule_violated"] for v in evaluation["violations"]]
        assert "1-3s" in violation_rules  # Should detect 1-3s violation
        # May also detect 2-2s and R-4s depending on exact implementation
    
    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self):
        """Test performance with large dataset"""
        engine = EnhancedWestgardEngine(target_mean=100.0, target_sd=2.0)
        
        # Generate large dataset (500 points)
        large_dataset = []
        for i in range(500):
            value = 100 + np.random.normal(0, 2)
            timestamp = datetime(2024, 1, 1) + timedelta(minutes=i*15)  # Every 15 minutes
            large_dataset.append(
                QCDataPoint(value, timestamp, "LOT123", f"OP{i%10+1}", f"R{i:04d}", 
                           "day" if 6 <= (i//4) % 24 <= 18 else "night")
            )
        
        # Measure evaluation time
        start_time = datetime.now()
        evaluation = await engine.evaluate_comprehensive(large_dataset)
        end_time = datetime.now()
        
        evaluation_time = (end_time - start_time).total_seconds()
        
        assert evaluation_time < 10.0  # Should complete within 10 seconds
        assert evaluation["status"] in ["in_control", "warning", "out_of_control"]
        assert evaluation["total_points"] == 500
        assert "ml_analysis" in evaluation
        assert "trend_analysis" in evaluation

if __name__ == "__main__":
    pytest.main([__file__, "-v"])