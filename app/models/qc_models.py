from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey, Index, Enum, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import uuid
import enum
from .base import Base, TimeStampedModel, AuditMixin

class QCLevelEnum(enum.Enum):
    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2" 
    LEVEL_3 = "level_3"
    MULTI_LEVEL = "multi_level"

class QCStatusEnum(enum.Enum):
    PENDING = "pending"
    IN_CONTROL = "in_control"
    OUT_OF_CONTROL = "out_of_control"
    WARNING = "warning"
    REJECTED = "rejected"

class WestgardRuleEnum(enum.Enum):
    RULE_13S = "1-3s"
    RULE_22S = "2-2s" 
    RULE_R4S = "R-4s"
    RULE_41S = "4-1s"
    RULE_10X = "10x"
    RULE_7T = "7T"
    RULE_9X = "9x"
    RULE_12X = "12x"

class CAPAStatusEnum(enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CLOSED = "closed"
    CANCELLED = "cancelled"

class Instrument(TimeStampedModel, AuditMixin):
    __tablename__ = "instruments"
    
    instrument_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    manufacturer = Column(String(100))
    model = Column(String(100))
    serial_number = Column(String(100), unique=True)
    location = Column(String(100))
    installation_date = Column(DateTime)
    service_date = Column(DateTime)
    calibration_due = Column(DateTime)
    maintenance_schedule = Column(JSONB)  # Store maintenance schedule as JSON
    
    # Relationships
    analytes = relationship("Analyte", back_populates="instrument")
    qc_lots = relationship("QCLot", back_populates="instrument")
    
    def __repr__(self):
        return f"<Instrument(id='{self.instrument_id}', name='{self.name}')>"

class Analyte(TimeStampedModel, AuditMixin):
    __tablename__ = "analytes"
    
    analyte_code = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    units = Column(String(20))
    method = Column(String(100))
    specimen_type = Column(String(50))
    
    # Reference ranges and limits
    reference_range_low = Column(Float)
    reference_range_high = Column(Float)
    critical_low = Column(Float)
    critical_high = Column(Float)
    
    # QC specifications
    allowable_total_error = Column(Float)  # Percentage
    coefficient_variation = Column(Float)  # Expected CV%
    
    # Associations
    instrument_id = Column(Integer, ForeignKey('instruments.id'), nullable=False)
    instrument = relationship("Instrument", back_populates="analytes")
    
    # Relationships
    qc_lots = relationship("QCLot", back_populates="analyte")
    qc_results = relationship("QCResult", back_populates="analyte")
    
    def __repr__(self):
        return f"<Analyte(code='{self.analyte_code}', name='{self.name}')>"

class QCLot(TimeStampedModel, AuditMixin):
    __tablename__ = "qc_lots"
    
    lot_number = Column(String(50), unique=True, nullable=False, index=True)
    manufacturer = Column(String(100))
    product_name = Column(String(100))
    qc_level = Column(Enum(QCLevelEnum), nullable=False)
    
    # Lot information
    expiration_date = Column(DateTime, nullable=False)
    received_date = Column(DateTime)
    opened_date = Column(DateTime)
    first_use_date = Column(DateTime)
    
    # Target values and statistics
    target_mean = Column(Float)
    target_sd = Column(Float)
    calculated_mean = Column(Float)
    calculated_sd = Column(Float)
    calculated_cv = Column(Float)  # Coefficient of Variation
    
    # Statistical parameters
    n_results = Column(Integer, default=0)
    last_calculated = Column(DateTime)
    
    # Storage conditions
    storage_temperature = Column(String(50))
    storage_conditions = Column(Text)
    
    # Associations
    instrument_id = Column(Integer, ForeignKey('instruments.id'), nullable=False)
    instrument = relationship("Instrument", back_populates="qc_lots")
    
    analyte_id = Column(Integer, ForeignKey('analytes.id'), nullable=False)
    analyte = relationship("Analyte", back_populates="qc_lots")
    
    # Relationships
    qc_results = relationship("QCResult", back_populates="qc_lot")
    
    # Indexes
    __table_args__ = (
        Index('idx_lot_analyte_instrument', 'analyte_id', 'instrument_id'),
        Index('idx_lot_expiration', 'expiration_date'),
    )
    
    def is_expired(self) -> bool:
        """Check if QC lot is expired"""
        return self.expiration_date < datetime.utcnow()
    
    def days_until_expiration(self) -> int:
        """Calculate days until expiration"""
        delta = self.expiration_date - datetime.utcnow()
        return delta.days
    
    def __repr__(self):
        return f"<QCLot(lot='{self.lot_number}', level='{self.qc_level.value}')>"

class QCResult(TimeStampedModel, AuditMixin):
    __tablename__ = "qc_results"
    
    result_id = Column(String(50), unique=True, nullable=False, index=True)
    
    # Result data
    value = Column(Float, nullable=False)
    units = Column(String(20))
    run_number = Column(Integer)
    duplicate_number = Column(Integer, default=1)
    
    # Timing
    run_datetime = Column(DateTime, nullable=False, default=func.now())
    shift = Column(String(20))  # morning, afternoon, night
    operator = Column(String(100))
    
    # QC evaluation
    status = Column(Enum(QCStatusEnum), nullable=False, default=QCStatusEnum.PENDING)
    z_score = Column(Float)  # (value - mean) / SD
    deviation_from_target = Column(Float)  # Percentage deviation
    
    # Flags and comments
    repeat_analysis = Column(Boolean, default=False)
    outlier_flag = Column(Boolean, default=False)
    comments = Column(Text)
    
    # Associations
    analyte_id = Column(Integer, ForeignKey('analytes.id'), nullable=False)
    analyte = relationship("Analyte", back_populates="qc_results")
    
    qc_lot_id = Column(Integer, ForeignKey('qc_lots.id'), nullable=False)
    qc_lot = relationship("QCLot", back_populates="qc_results")
    
    # Relationships
    rule_violations = relationship("WestgardViolation", back_populates="qc_result")
    
    # Indexes
    __table_args__ = (
        Index('idx_result_lot_datetime', 'qc_lot_id', 'run_datetime'),
        Index('idx_result_analyte_datetime', 'analyte_id', 'run_datetime'),
        Index('idx_result_status', 'status'),
    )
    
    def calculate_z_score(self, target_mean: float, target_sd: float) -> float:
        """Calculate Z-score for the result"""
        if target_sd == 0:
            return 0.0
        self.z_score = (self.value - target_mean) / target_sd
        return self.z_score
    
    def calculate_deviation(self, target_mean: float) -> float:
        """Calculate percentage deviation from target"""
        if target_mean == 0:
            return 0.0
        self.deviation_from_target = ((self.value - target_mean) / target_mean) * 100
        return self.deviation_from_target
    
    def __repr__(self):
        return f"<QCResult(id='{self.result_id}', value={self.value}, status='{self.status.value}')>"

class WestgardViolation(TimeStampedModel, AuditMixin):
    __tablename__ = "westgard_violations"
    
    violation_id = Column(String(50), unique=True, nullable=False, index=True)
    
    # Violation details
    rule_violated = Column(Enum(WestgardRuleEnum), nullable=False)
    severity = Column(String(20))  # warning, reject, critical
    violation_datetime = Column(DateTime, nullable=False, default=func.now())
    
    # Context information
    consecutive_violations = Column(Integer, default=1)
    pattern_description = Column(Text)
    recommended_action = Column(Text)
    
    # Resolution tracking
    resolved = Column(Boolean, default=False)
    resolved_datetime = Column(DateTime)
    resolution_method = Column(String(100))
    resolution_comments = Column(Text)
    
    # Associations
    qc_result_id = Column(Integer, ForeignKey('qc_results.id'), nullable=False)
    qc_result = relationship("QCResult", back_populates="rule_violations")
    
    # Related results that contributed to violation
    related_result_ids = Column(ARRAY(Integer))
    
    # Relationships
    capa_actions = relationship("CAPAAction", back_populates="westgard_violation")
    
    def __repr__(self):
        return f"<WestgardViolation(rule='{self.rule_violated.value}', severity='{self.severity}')>"

class QCRun(TimeStampedModel, AuditMixin):
    __tablename__ = "qc_runs"
    
    run_id = Column(String(50), unique=True, nullable=False, index=True)
    run_datetime = Column(DateTime, nullable=False, default=func.now())
    
    # Run information
    run_type = Column(String(50))  # daily, weekly, maintenance, calibration
    shift = Column(String(20))
    operator = Column(String(100))
    
    # Run status
    status = Column(Enum(QCStatusEnum), nullable=False, default=QCStatusEnum.PENDING)
    total_results = Column(Integer, default=0)
    passed_results = Column(Integer, default=0)
    failed_results = Column(Integer, default=0)
    
    # Performance metrics
    completion_time_minutes = Column(Integer)
    coefficient_variation = Column(Float)
    
    # Comments and notes
    pre_run_comments = Column(Text)
    post_run_comments = Column(Text)
    
    # Associations - this run can span multiple instruments/analytes
    instrument_id = Column(Integer, ForeignKey('instruments.id'))
    instrument = relationship("Instrument")
    
    # Results included in this run (via foreign key in QCResult)
    
    def calculate_pass_rate(self) -> float:
        """Calculate pass rate for the run"""
        if self.total_results == 0:
            return 0.0
        return (self.passed_results / self.total_results) * 100
    
    def __repr__(self):
        return f"<QCRun(id='{self.run_id}', status='{self.status.value}')>"

class QCAlert(TimeStampedModel, AuditMixin):
    __tablename__ = "qc_alerts"
    
    alert_id = Column(String(50), unique=True, nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)  # rule_violation, trend, expiration, etc.
    severity = Column(String(20))  # low, medium, high, critical
    
    # Alert content
    title = Column(String(200))
    message = Column(Text)
    alert_datetime = Column(DateTime, nullable=False, default=func.now())
    
    # Status tracking
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100))
    acknowledged_datetime = Column(DateTime)
    
    resolved = Column(Boolean, default=False)
    resolved_by = Column(String(100))
    resolved_datetime = Column(DateTime)
    
    # Alert context (JSON for flexible structure)
    context_data = Column(JSONB)
    
    # Escalation
    escalation_level = Column(Integer, default=0)
    escalated_to = Column(ARRAY(String))
    
    def __repr__(self):
        return f"<QCAlert(type='{self.alert_type}', severity='{self.severity}')>"

class CAPAAction(TimeStampedModel, AuditMixin):
    __tablename__ = "capa_actions"
    
    capa_id = Column(String(50), unique=True, nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    
    # CAPA categorization
    capa_type = Column(String(50))  # corrective, preventive, both
    root_cause = Column(Text)
    impact_assessment = Column(Text)
    
    # Status and timeline
    status = Column(Enum(CAPAStatusEnum), nullable=False, default=CAPAStatusEnum.OPEN)
    priority = Column(String(20))  # low, medium, high, critical
    
    target_completion_date = Column(DateTime)
    actual_completion_date = Column(DateTime)
    
    # Assignment
    assigned_to = Column(String(100))
    assigned_by = Column(String(100))
    assigned_datetime = Column(DateTime)
    
    # Verification and closure
    verification_required = Column(Boolean, default=True)
    verified_by = Column(String(100))
    verification_datetime = Column(DateTime)
    verification_comments = Column(Text)
    
    # Effectiveness monitoring
    effectiveness_check_due = Column(DateTime)
    effectiveness_verified = Column(Boolean, default=False)
    
    # Associations
    westgard_violation_id = Column(Integer, ForeignKey('westgard_violations.id'))
    westgard_violation = relationship("WestgardViolation", back_populates="capa_actions")
    
    def is_overdue(self) -> bool:
        """Check if CAPA action is overdue"""
        if not self.target_completion_date:
            return False
        return datetime.utcnow() > self.target_completion_date and self.status not in [CAPAStatusEnum.COMPLETED, CAPAStatusEnum.CLOSED]
    
    def days_until_due(self) -> int:
        """Calculate days until due date"""
        if not self.target_completion_date:
            return 0
        delta = self.target_completion_date - datetime.utcnow()
        return delta.days
    
    def __repr__(self):
        return f"<CAPAAction(id='{self.capa_id}', status='{self.status.value}')>"

class QCTrend(TimeStampedModel, AuditMixin):
    __tablename__ = "qc_trends"
    
    trend_id = Column(String(50), unique=True, nullable=False, index=True)
    
    # Trend analysis parameters
    analyte_id = Column(Integer, ForeignKey('analytes.id'), nullable=False)
    qc_lot_id = Column(Integer, ForeignKey('qc_lots.id'), nullable=False)
    
    # Time period for trend
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Statistical analysis
    trend_type = Column(String(50))  # increasing, decreasing, cyclic, random
    slope = Column(Float)  # Linear trend slope
    correlation_coefficient = Column(Float)  # R value
    p_value = Column(Float)  # Statistical significance
    
    # Trend strength
    strength = Column(String(20))  # weak, moderate, strong
    confidence_level = Column(Float)  # 0.95, 0.99, etc.
    
    # Alert thresholds
    alert_threshold_reached = Column(Boolean, default=False)
    alert_generated = Column(Boolean, default=False)
    
    # Analysis results
    data_points = Column(Integer)
    trend_description = Column(Text)
    recommendations = Column(Text)
    
    analyte = relationship("Analyte")
    qc_lot = relationship("QCLot")
    
    def __repr__(self):
        return f"<QCTrend(type='{self.trend_type}', strength='{self.strength}')>"

class ComplianceReport(TimeStampedModel, AuditMixin):
    __tablename__ = "compliance_reports"
    
    report_id = Column(String(50), unique=True, nullable=False, index=True)
    report_type = Column(String(50))  # daily, weekly, monthly, quarterly, annual
    
    # Report period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    generated_datetime = Column(DateTime, nullable=False, default=func.now())
    
    # Report scope
    instruments_included = Column(ARRAY(String))
    analytes_included = Column(ARRAY(String))
    
    # Summary statistics
    total_qc_results = Column(Integer)
    passed_results = Column(Integer)
    failed_results = Column(Integer)
    warning_results = Column(Integer)
    
    # Compliance metrics
    overall_compliance_rate = Column(Float)  # Percentage
    westgard_violations = Column(Integer)
    capa_actions_opened = Column(Integer)
    capa_actions_closed = Column(Integer)
    
    # Trend analysis summary
    significant_trends = Column(Integer)
    instruments_out_of_control = Column(Integer)
    
    # Report data (detailed JSON)
    report_data = Column(JSONB)
    
    # File information if exported
    exported_file_path = Column(String(500))
    exported_by = Column(String(100))
    
    def __repr__(self):
        return f"<ComplianceReport(id='{self.report_id}', type='{self.report_type}')>"

# Create additional indexes for performance
Index('idx_qc_result_datetime_status', QCResult.run_datetime, QCResult.status)
Index('idx_violation_datetime_rule', WestgardViolation.violation_datetime, WestgardViolation.rule_violated)
Index('idx_capa_status_due_date', CAPAAction.status, CAPAAction.target_completion_date)
Index('idx_alert_datetime_severity', QCAlert.alert_datetime, QCAlert.severity)
Index('idx_trend_dates', QCTrend.start_date, QCTrend.end_date)