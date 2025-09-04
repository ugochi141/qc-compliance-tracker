class WestgardRules:
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd
    
    def check_13s(self, value):
        """Check if value exceeds 3 SD"""
        return abs(value - self.mean) > 3 * self.sd
    
    def check_22s(self, values):
        """Check if 2 consecutive values exceed 2 SD on same side"""
        if len(values) < 2:
            return False
        for i in range(len(values) - 1):
            if (values[i] - self.mean > 2 * self.sd and 
                values[i+1] - self.mean > 2 * self.sd):
                return True
        return False
    
    def check_r4s(self, values):
        """Check if range exceeds 4 SD"""
        if len(values) < 2:
            return False
        return max(values) - min(values) > 4 * self.sd
