import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class QCResult:
    value: float
    timestamp: datetime
    lot_number: str
    operator: str

class WestgardRuleEngine:
    """Implements Westgard multi-rule QC system"""
    
    def __init__(self, mean: float, sd: float):
        self.mean = mean
        self.sd = sd
        self.history = []
        
    def evaluate(self, results: List[QCResult]) -> Dict:
        """Evaluate QC results against all Westgard rules"""
        violations = []
        
        # Rule 1-3s: One control exceeds ±3SD
        if self._check_13s(results):
            violations.append({
                "rule": "1-3s",
                "severity": "reject",
                "message": "Control exceeds 3 standard deviations"
            })
        
        # Rule 2-2s: Two consecutive controls exceed ±2SD
        if self._check_22s(results):
            violations.append({
                "rule": "2-2s",
                "severity": "reject",
                "message": "Two consecutive controls exceed 2SD on same side"
            })
        
        # Rule R-4s: Range between controls exceeds 4SD
        if self._check_r4s(results):
            violations.append({
                "rule": "R-4s",
                "severity": "reject",
                "message": "Range between controls exceeds 4SD"
            })
        
        # Rule 4-1s: Four consecutive controls exceed ±1SD
        if self._check_41s(results):
            violations.append({
                "rule": "4-1s",
                "severity": "reject",
                "message": "Four consecutive controls exceed 1SD on same side"
            })
        
        # Rule 10x: Ten consecutive controls on same side of mean
        if self._check_10x(results):
            violations.append({
                "rule": "10x",
                "severity": "warning",
                "message": "Ten consecutive controls on same side of mean"
            })
        
        return {
            "status": "reject" if violations else "accept",
            "violations": violations,
            "evaluated_at": datetime.now()
        }
    
    def _check_13s(self, results: List[QCResult]) -> bool:
        """Check 1-3s rule"""
        for r in results:
            if abs(r.value - self.mean) > 3 * self.sd:
                return True
        return False
    
    def _check_22s(self, results: List[QCResult]) -> bool:
        """Check 2-2s rule"""
        if len(results) < 2:
            return False
        for i in range(len(results) - 1):
            if (results[i].value - self.mean > 2 * self.sd and 
                results[i+1].value - self.mean > 2 * self.sd):
                return True
            if (self.mean - results[i].value > 2 * self.sd and 
                self.mean - results[i+1].value > 2 * self.sd):
                return True
        return False
    
    def _check_r4s(self, results: List[QCResult]) -> bool:
        """Check R-4s rule"""
        if len(results) < 2:
            return False
        values = [r.value for r in results[-2:]]
        return max(values) - min(values) > 4 * self.sd
    
    def _check_41s(self, results: List[QCResult]) -> bool:
        """Check 4-1s rule"""
        if len(results) < 4:
            return False
        for i in range(len(results) - 3):
            subset = results[i:i+4]
            if all(r.value > self.mean + self.sd for r in subset):
                return True
            if all(r.value < self.mean - self.sd for r in subset):
                return True
        return False
    
    def _check_10x(self, results: List[QCResult]) -> bool:
        """Check 10x rule"""
        if len(results) < 10:
            return False
        last_10 = results[-10:]
        if all(r.value > self.mean for r in last_10):
            return True
        if all(r.value < self.mean for r in last_10):
            return True
        return False
