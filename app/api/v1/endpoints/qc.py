from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import io
import asyncio
from ....qc.enhanced_westgard import EnhancedWestgardEngine, QCDataPoint
from ....models.qc_models import (
    QCResult, QCLot, Instrument, Analyte, WestgardViolation, 
    QCAlert, CAPAAction, QCRun, ComplianceReport,
    QCStatusEnum, WestgardRuleEnum, CAPAStatusEnum
)
from ....database import get_db
from ....utils.monitoring import track_performance, BusinessMetricsCollector
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qc", tags=["Quality Control"])

# Pydantic models for request/response
class QCResultSubmission(BaseModel):
    analyte_code: str
    lot_number: str
    value: float
    run_datetime: datetime
    operator: str
    shift: str
    duplicate_number: int = 1
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    comments: Optional[str] = None

class BulkQCSubmission(BaseModel):
    results: List[QCResultSubmission]
    run_id: str
    instrument_id: str

class QCLotRequest(BaseModel):
    lot_number: str
    analyte_code: str
    instrument_id: str
    target_mean: Optional[float] = None
    target_sd: Optional[float] = None
    expiration_date: datetime
    qc_level: str = "level_1"

class WestgardEvaluationRequest(BaseModel):
    analyte_code: str
    lot_number: str
    instrument_id: str
    days_back: int = 30
    include_ml_analysis: bool = True

@router.post("/results/submit", response_model=Dict)
@track_performance
async def submit_qc_result(
    submission: QCResultSubmission,
    db: Session = Depends(get_db)
):
    """Submit a single QC result for evaluation"""
    try:
        # Validate analyte exists
        analyte = db.query(Analyte).filter(
            Analyte.analyte_code == submission.analyte_code
        ).first()
        
        if not analyte:
            raise HTTPException(status_code=404, detail=f"Analyte '{submission.analyte_code}' not found")
        
        # Validate QC lot exists
        qc_lot = db.query(QCLot).filter(
            QCLot.lot_number == submission.lot_number,
            QCLot.analyte_id == analyte.id
        ).first()
        
        if not qc_lot:
            raise HTTPException(status_code=404, detail=f"QC lot '{submission.lot_number}' not found")
        
        # Create QC result record
        qc_result = QCResult(
            result_id=f"QC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{submission.analyte_code}",
            value=submission.value,
            units=analyte.units,
            run_datetime=submission.run_datetime,
            operator=submission.operator,
            shift=submission.shift,
            duplicate_number=submission.duplicate_number,
            comments=submission.comments,
            analyte_id=analyte.id,
            qc_lot_id=qc_lot.id,
            status=QCStatusEnum.PENDING
        )
        
        # Calculate Z-score and deviation
        if qc_lot.target_mean and qc_lot.target_sd:
            qc_result.calculate_z_score(qc_lot.target_mean, qc_lot.target_sd)
            qc_result.calculate_deviation(qc_lot.target_mean)
        
        db.add(qc_result)
        db.commit()
        db.refresh(qc_result)
        
        # Perform Westgard evaluation in background
        background_tasks = BackgroundTasks()
        background_tasks.add_task(
            _evaluate_westgard_rules,
            qc_result.id,
            qc_lot.id,
            db
        )
        
        # Track business metrics
        BusinessMetricsCollector.track_lab_order_processed(
            department=analyte.instrument.location or "Unknown",
            status="qc_submitted"
        )
        
        return {
            "success": True,
            "result_id": qc_result.result_id,
            "message": "QC result submitted successfully",
            "z_score": qc_result.z_score,
            "deviation_percent": qc_result.deviation_from_target,
            "status": qc_result.status.value
        }
        
    except Exception as e:
        logger.error(f"Error submitting QC result: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/results/bulk-submit", response_model=Dict)
@track_performance
async def bulk_submit_qc_results(
    submission: BulkQCSubmission,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Submit multiple QC results in bulk"""
    try:
        created_results = []
        
        # Validate instrument exists
        instrument = db.query(Instrument).filter(
            Instrument.instrument_id == submission.instrument_id
        ).first()
        
        if not instrument:
            raise HTTPException(status_code=404, detail=f"Instrument '{submission.instrument_id}' not found")
        
        # Create QC run record
        qc_run = QCRun(
            run_id=submission.run_id,
            run_datetime=datetime.utcnow(),
            run_type="routine",
            instrument_id=instrument.id,
            total_results=len(submission.results),
            status=QCStatusEnum.PENDING
        )
        db.add(qc_run)
        
        # Process each result
        for result_data in submission.results:
            # Validate analyte and lot
            analyte = db.query(Analyte).filter(
                Analyte.analyte_code == result_data.analyte_code,
                Analyte.instrument_id == instrument.id
            ).first()
            
            if not analyte:
                logger.warning(f"Analyte '{result_data.analyte_code}' not found for instrument '{submission.instrument_id}'")
                continue
            
            qc_lot = db.query(QCLot).filter(
                QCLot.lot_number == result_data.lot_number,
                QCLot.analyte_id == analyte.id
            ).first()
            
            if not qc_lot:
                logger.warning(f"QC lot '{result_data.lot_number}' not found")
                continue
            
            # Create result record
            qc_result = QCResult(
                result_id=f"{submission.run_id}_{result_data.analyte_code}_{result_data.duplicate_number}",
                value=result_data.value,
                units=analyte.units,
                run_datetime=result_data.run_datetime,
                operator=result_data.operator,
                shift=result_data.shift,
                duplicate_number=result_data.duplicate_number,
                comments=result_data.comments,
                analyte_id=analyte.id,
                qc_lot_id=qc_lot.id,
                status=QCStatusEnum.PENDING
            )
            
            # Calculate statistics
            if qc_lot.target_mean and qc_lot.target_sd:
                qc_result.calculate_z_score(qc_lot.target_mean, qc_lot.target_sd)
                qc_result.calculate_deviation(qc_lot.target_mean)
            
            db.add(qc_result)
            created_results.append(qc_result)
        
        db.commit()
        
        # Update QC run status
        qc_run.total_results = len(created_results)
        db.commit()
        
        # Schedule Westgard evaluations
        for qc_result in created_results:
            background_tasks.add_task(
                _evaluate_westgard_rules,
                qc_result.id,
                qc_result.qc_lot_id,
                db
            )
        
        return {
            "success": True,
            "run_id": submission.run_id,
            "results_processed": len(created_results),
            "total_submitted": len(submission.results),
            "message": f"Processed {len(created_results)} QC results"
        }
        
    except Exception as e:
        logger.error(f"Error in bulk QC submission: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate/westgard", response_model=Dict)
@track_performance
async def evaluate_westgard_rules(
    request: WestgardEvaluationRequest,
    db: Session = Depends(get_db)
):
    """Evaluate Westgard rules for specified analyte and lot"""
    try:
        # Get analyte and lot
        analyte = db.query(Analyte).filter(
            Analyte.analyte_code == request.analyte_code
        ).first()
        
        if not analyte:
            raise HTTPException(status_code=404, detail="Analyte not found")
        
        qc_lot = db.query(QCLot).filter(
            QCLot.lot_number == request.lot_number,
            QCLot.analyte_id == analyte.id
        ).first()
        
        if not qc_lot:
            raise HTTPException(status_code=404, detail="QC lot not found")
        
        # Get recent QC results
        cutoff_date = datetime.utcnow() - timedelta(days=request.days_back)
        qc_results = db.query(QCResult).filter(
            QCResult.qc_lot_id == qc_lot.id,
            QCResult.run_datetime >= cutoff_date
        ).order_by(QCResult.run_datetime).all()
        
        if not qc_results:
            return {
                "success": True,
                "message": "No QC results found for evaluation period",
                "status": "insufficient_data"
            }
        
        # Convert to QC data points
        data_points = []
        for result in qc_results:
            data_point = QCDataPoint(
                value=result.value,
                timestamp=result.run_datetime,
                lot_number=qc_lot.lot_number,
                operator=result.operator or "Unknown",
                run_id=f"run_{result.id}",
                shift=result.shift or "Unknown",
                duplicate_number=result.duplicate_number
            )
            data_points.append(data_point)
        
        # Initialize Westgard engine
        if not qc_lot.target_mean or not qc_lot.target_sd:
            raise HTTPException(
                status_code=400, 
                detail="QC lot missing target mean/SD values"
            )
        
        westgard_engine = EnhancedWestgardEngine(
            target_mean=qc_lot.target_mean,
            target_sd=qc_lot.target_sd,
            enable_ml_analysis=request.include_ml_analysis
        )
        
        # Perform comprehensive evaluation
        evaluation_result = await westgard_engine.evaluate_comprehensive(data_points)
        
        # Update QC result statuses based on evaluation
        if evaluation_result["status"] == "out_of_control":
            # Mark recent results as out of control
            for result in qc_results[-5:]:  # Last 5 results
                result.status = QCStatusEnum.OUT_OF_CONTROL
            
            # Create alert
            await _create_qc_alert(
                qc_lot_id=qc_lot.id,
                alert_type="westgard_violation",
                severity="high",
                title=f"Westgard rule violation - {analyte.name}",
                message=f"QC out of control for {analyte.name}, Lot {qc_lot.lot_number}",
                db=db
            )
            
            BusinessMetricsCollector.track_anomaly(
                department=analyte.instrument.location or "Unknown"
            )
        
        elif evaluation_result["status"] == "warning":
            for result in qc_results[-3:]:  # Last 3 results
                result.status = QCStatusEnum.WARNING
        
        else:
            for result in qc_results:
                if result.status == QCStatusEnum.PENDING:
                    result.status = QCStatusEnum.IN_CONTROL
        
        db.commit()
        
        return {
            "success": True,
            "evaluation": evaluation_result,
            "analyte": analyte.name,
            "lot_number": qc_lot.lot_number,
            "results_evaluated": len(qc_results),
            "evaluation_period_days": request.days_back
        }
        
    except Exception as e:
        logger.error(f"Error in Westgard evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/lots/create", response_model=Dict)
@track_performance
async def create_qc_lot(
    lot_request: QCLotRequest,
    db: Session = Depends(get_db)
):
    """Create a new QC lot"""
    try:
        # Validate analyte and instrument
        analyte = db.query(Analyte).filter(
            Analyte.analyte_code == lot_request.analyte_code
        ).first()
        
        if not analyte:
            raise HTTPException(status_code=404, detail="Analyte not found")
        
        instrument = db.query(Instrument).filter(
            Instrument.instrument_id == lot_request.instrument_id
        ).first()
        
        if not instrument:
            raise HTTPException(status_code=404, detail="Instrument not found")
        
        # Check if lot already exists
        existing_lot = db.query(QCLot).filter(
            QCLot.lot_number == lot_request.lot_number,
            QCLot.analyte_id == analyte.id
        ).first()
        
        if existing_lot:
            raise HTTPException(status_code=400, detail="QC lot already exists")
        
        # Create QC lot
        qc_lot = QCLot(
            lot_number=lot_request.lot_number,
            analyte_id=analyte.id,
            instrument_id=instrument.id,
            target_mean=lot_request.target_mean,
            target_sd=lot_request.target_sd,
            expiration_date=lot_request.expiration_date,
            qc_level=lot_request.qc_level,
            received_date=datetime.utcnow()
        )
        
        db.add(qc_lot)
        db.commit()
        db.refresh(qc_lot)
        
        return {
            "success": True,
            "lot_id": qc_lot.id,
            "lot_number": qc_lot.lot_number,
            "message": "QC lot created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating QC lot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lots/{lot_number}/statistics", response_model=Dict)
@track_performance
async def get_lot_statistics(
    lot_number: str,
    analyte_code: str,
    days_back: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get statistical summary for a QC lot"""
    try:
        # Find QC lot
        analyte = db.query(Analyte).filter(
            Analyte.analyte_code == analyte_code
        ).first()
        
        if not analyte:
            raise HTTPException(status_code=404, detail="Analyte not found")
        
        qc_lot = db.query(QCLot).filter(
            QCLot.lot_number == lot_number,
            QCLot.analyte_id == analyte.id
        ).first()
        
        if not qc_lot:
            raise HTTPException(status_code=404, detail="QC lot not found")
        
        # Get QC results
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        results = db.query(QCResult).filter(
            QCResult.qc_lot_id == qc_lot.id,
            QCResult.run_datetime >= cutoff_date
        ).all()
        
        if not results:
            return {
                "success": True,
                "message": "No results found for the specified period",
                "statistics": None
            }
        
        values = [r.value for r in results]
        values_array = np.array(values)
        
        # Calculate statistics
        statistics = {
            "n_results": len(values),
            "mean": float(np.mean(values_array)),
            "std_dev": float(np.std(values_array, ddof=1)),
            "cv_percent": float(np.std(values_array, ddof=1) / np.mean(values_array) * 100),
            "min_value": float(np.min(values_array)),
            "max_value": float(np.max(values_array)),
            "median": float(np.median(values_array)),
            "q1": float(np.percentile(values_array, 25)),
            "q3": float(np.percentile(values_array, 75)),
            "target_mean": qc_lot.target_mean,
            "target_sd": qc_lot.target_sd,
            "bias_percent": ((np.mean(values_array) - qc_lot.target_mean) / qc_lot.target_mean * 100) if qc_lot.target_mean else 0,
            "precision_ratio": (np.std(values_array, ddof=1) / qc_lot.target_sd) if qc_lot.target_sd else 1
        }
        
        # Status summary
        status_counts = {}
        for result in results:
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "success": True,
            "lot_number": lot_number,
            "analyte": analyte.name,
            "period_days": days_back,
            "statistics": statistics,
            "status_summary": status_counts,
            "lot_expiration": qc_lot.expiration_date.isoformat(),
            "days_until_expiration": qc_lot.days_until_expiration()
        }
        
    except Exception as e:
        logger.error(f"Error getting lot statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/summary", response_model=Dict)
@track_performance
async def get_qc_dashboard_summary(
    days_back: int = Query(default=7, ge=1, le=30),
    db: Session = Depends(get_db)
):
    """Get QC dashboard summary"""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        # Total results
        total_results = db.query(QCResult).filter(
            QCResult.run_datetime >= cutoff_date
        ).count()
        
        # Results by status
        status_counts = {}
        for status in QCStatusEnum:
            count = db.query(QCResult).filter(
                QCResult.run_datetime >= cutoff_date,
                QCResult.status == status
            ).count()
            status_counts[status.value] = count
        
        # Active violations
        active_violations = db.query(WestgardViolation).filter(
            WestgardViolation.resolved == False,
            WestgardViolation.violation_datetime >= cutoff_date
        ).count()
        
        # Open CAPA actions
        open_capas = db.query(CAPAAction).filter(
            CAPAAction.status.in_([CAPAStatusEnum.OPEN, CAPAStatusEnum.IN_PROGRESS])
        ).count()
        
        # Instruments with issues
        instruments_with_issues = db.query(QCResult).filter(
            QCResult.run_datetime >= cutoff_date,
            QCResult.status.in_([QCStatusEnum.OUT_OF_CONTROL, QCStatusEnum.WARNING])
        ).join(Analyte).join(Instrument).distinct(Instrument.id).count()
        
        # Recent alerts
        recent_alerts = db.query(QCAlert).filter(
            QCAlert.alert_datetime >= cutoff_date,
            QCAlert.resolved == False
        ).order_by(QCAlert.alert_datetime.desc()).limit(10).all()
        
        # Compliance rate
        total_evaluated = status_counts.get("in_control", 0) + status_counts.get("out_of_control", 0)
        compliance_rate = (status_counts.get("in_control", 0) / total_evaluated * 100) if total_evaluated > 0 else 0
        
        return {
            "success": True,
            "period_days": days_back,
            "summary": {
                "total_results": total_results,
                "compliance_rate": round(compliance_rate, 1),
                "active_violations": active_violations,
                "open_capa_actions": open_capas,
                "instruments_with_issues": instruments_with_issues,
                "status_distribution": status_counts
            },
            "recent_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "type": alert.alert_type,
                    "severity": alert.severity,
                    "title": alert.title,
                    "datetime": alert.alert_datetime.isoformat()
                }
                for alert in recent_alerts
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/compliance", response_model=Dict)
@track_performance
async def generate_compliance_report(
    start_date: datetime,
    end_date: datetime,
    instrument_ids: Optional[List[str]] = Query(None),
    analyte_codes: Optional[List[str]] = Query(None),
    export_format: str = Query(default="json", regex="^(json|csv|excel)$"),
    db: Session = Depends(get_db)
):
    """Generate comprehensive compliance report"""
    try:
        # Build query
        query = db.query(QCResult).filter(
            QCResult.run_datetime >= start_date,
            QCResult.run_datetime <= end_date
        ).join(QCLot).join(Analyte)
        
        if instrument_ids:
            query = query.join(Instrument).filter(
                Instrument.instrument_id.in_(instrument_ids)
            )
        
        if analyte_codes:
            query = query.filter(
                Analyte.analyte_code.in_(analyte_codes)
            )
        
        results = query.all()
        
        if not results:
            return {
                "success": True,
                "message": "No data found for specified criteria",
                "report": None
            }
        
        # Generate report data
        report_data = await _generate_compliance_report_data(results, start_date, end_date, db)
        
        # Create compliance report record
        report = ComplianceReport(
            report_id=f"COMP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type="custom",
            period_start=start_date,
            period_end=end_date,
            generated_datetime=datetime.utcnow(),
            instruments_included=instrument_ids or [],
            analytes_included=analyte_codes or [],
            **report_data["summary"],
            report_data=report_data
        )
        
        db.add(report)
        db.commit()
        
        if export_format == "json":
            return {
                "success": True,
                "report_id": report.report_id,
                "report": report_data
            }
        
        # Generate file export
        file_content = await _export_compliance_report(report_data, export_format)
        
        if export_format == "csv":
            media_type = "text/csv"
            filename = f"compliance_report_{report.report_id}.csv"
        else:  # excel
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"compliance_report_{report.report_id}.xlsx"
        
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error generating compliance report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def _evaluate_westgard_rules(qc_result_id: int, qc_lot_id: int, db: Session):
    """Background task to evaluate Westgard rules"""
    try:
        # Get QC lot and recent results
        qc_lot = db.query(QCLot).get(qc_lot_id)
        if not qc_lot or not qc_lot.target_mean or not qc_lot.target_sd:
            return
        
        # Get last 30 results for evaluation context
        recent_results = db.query(QCResult).filter(
            QCResult.qc_lot_id == qc_lot_id
        ).order_by(QCResult.run_datetime.desc()).limit(30).all()
        
        if len(recent_results) < 1:
            return
        
        # Convert to data points
        data_points = []
        for result in reversed(recent_results):  # Chronological order
            data_point = QCDataPoint(
                value=result.value,
                timestamp=result.run_datetime,
                lot_number=qc_lot.lot_number,
                operator=result.operator or "Unknown",
                run_id=f"run_{result.id}",
                shift=result.shift or "Unknown",
                duplicate_number=result.duplicate_number
            )
            data_points.append(data_point)
        
        # Initialize and run Westgard engine
        westgard_engine = EnhancedWestgardEngine(
            target_mean=qc_lot.target_mean,
            target_sd=qc_lot.target_sd,
            enable_ml_analysis=True
        )
        
        evaluation = await westgard_engine.evaluate_comprehensive(data_points)
        
        # Process violations
        if evaluation["violations"]:
            for violation_data in evaluation["violations"]:
                # Create violation record
                violation = WestgardViolation(
                    violation_id=f"VIO_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{violation_data['rule_violated']}",
                    rule_violated=WestgardRuleEnum(violation_data["rule_violated"]),
                    severity=violation_data["severity"],
                    pattern_description=violation_data["pattern_description"],
                    recommended_action=violation_data["recommended_action"],
                    qc_result_id=qc_result_id,
                    related_result_ids=[recent_results[i].id for i in violation_data.get("affected_points", [])]
                )
                
                db.add(violation)
                
                # Create alert for critical violations
                if violation_data["severity"] == "reject":
                    await _create_qc_alert(
                        qc_lot_id=qc_lot_id,
                        alert_type="critical_violation",
                        severity="critical",
                        title=f"Critical QC Violation - {violation_data['rule_violated']}",
                        message=violation_data["message"],
                        db=db
                    )
        
        db.commit()
        
    except Exception as e:
        logger.error(f"Error in background Westgard evaluation: {str(e)}")

async def _create_qc_alert(qc_lot_id: int, alert_type: str, severity: str, 
                          title: str, message: str, db: Session):
    """Create QC alert"""
    try:
        alert = QCAlert(
            alert_id=f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            context_data={"qc_lot_id": qc_lot_id}
        )
        
        db.add(alert)
        db.commit()
        
    except Exception as e:
        logger.error(f"Error creating QC alert: {str(e)}")

async def _generate_compliance_report_data(results: List[QCResult], 
                                         start_date: datetime, 
                                         end_date: datetime, 
                                         db: Session) -> Dict[str, Any]:
    """Generate compliance report data"""
    
    total_results = len(results)
    passed_results = len([r for r in results if r.status == QCStatusEnum.IN_CONTROL])
    failed_results = len([r for r in results if r.status == QCStatusEnum.OUT_OF_CONTROL])
    warning_results = len([r for r in results if r.status == QCStatusEnum.WARNING])
    
    # Compliance rate
    evaluated_results = passed_results + failed_results
    compliance_rate = (passed_results / evaluated_results * 100) if evaluated_results > 0 else 0
    
    # Violations in period
    violations = db.query(WestgardViolation).filter(
        WestgardViolation.violation_datetime >= start_date,
        WestgardViolation.violation_datetime <= end_date
    ).all()
    
    # CAPA actions
    capa_opened = db.query(CAPAAction).filter(
        CAPAAction.created_at >= start_date,
        CAPAAction.created_at <= end_date
    ).count()
    
    capa_closed = db.query(CAPAAction).filter(
        CAPAAction.actual_completion_date >= start_date,
        CAPAAction.actual_completion_date <= end_date
    ).count()
    
    return {
        "summary": {
            "total_qc_results": total_results,
            "passed_results": passed_results,
            "failed_results": failed_results,
            "warning_results": warning_results,
            "overall_compliance_rate": round(compliance_rate, 2),
            "westgard_violations": len(violations),
            "capa_actions_opened": capa_opened,
            "capa_actions_closed": capa_closed
        },
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days": (end_date - start_date).days
        },
        "violations": [
            {
                "rule": v.rule_violated.value,
                "severity": v.severity,
                "datetime": v.violation_datetime.isoformat(),
                "resolved": v.resolved
            }
            for v in violations
        ],
        "detailed_results": [
            {
                "result_id": r.result_id,
                "analyte": r.analyte.name,
                "value": r.value,
                "status": r.status.value,
                "z_score": r.z_score,
                "datetime": r.run_datetime.isoformat(),
                "operator": r.operator
            }
            for r in results[:100]  # Limit detailed results
        ]
    }

async def _export_compliance_report(report_data: Dict[str, Any], format_type: str) -> bytes:
    """Export compliance report to specified format"""
    
    if format_type == "csv":
        # Create CSV content
        df = pd.DataFrame(report_data["detailed_results"])
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue().encode('utf-8')
    
    elif format_type == "excel":
        # Create Excel content
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([report_data["summary"]])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed results sheet
            results_df = pd.DataFrame(report_data["detailed_results"])
            results_df.to_excel(writer, sheet_name='Results', index=False)
            
            # Violations sheet
            violations_df = pd.DataFrame(report_data["violations"])
            violations_df.to_excel(writer, sheet_name='Violations', index=False)
        
        return excel_buffer.getvalue()
    
    else:
        raise ValueError(f"Unsupported export format: {format_type}")