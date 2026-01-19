"""
Medical Calculator Tool.

Local calculations - no external API needed.
"""

import logging
from typing import Optional
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class MedicalCalcTool(BaseTool):
    """
    Medical calculations and scoring tools.
    
    Performs various medical calculations locally:
    - Unit conversions
    - Clinical scores
    - Basic calculations
    
    SKELETON - Core structure, calculations can be added.
    """
    
    name = "medical_calc"
    description = "Perform medical calculations. Available: unit_convert, bmi, gcs (Glasgow Coma Scale)."
    parameters = {
        "type": "object",
        "properties": {
            "calculation": {
                "type": "string",
                "description": "Type of calculation: 'unit_convert', 'bmi', 'gcs'"
            },
            "values": {
                "type": "object",
                "description": "Input values for the calculation"
            }
        },
        "required": ["calculation", "values"]
    }
    
    def execute(
        self,
        calculation: str,
        values: dict,
        **kwargs
    ) -> ToolResult:
        """
        Perform a medical calculation.
        
        Args:
            calculation: Type of calculation
            values: Input values
            
        Returns:
            ToolResult with calculation result
        """
        calc_methods = {
            "bmi": self._calculate_bmi,
            "gcs": self._calculate_gcs,
            "unit_convert": self._unit_convert,
        }
        
        if calculation not in calc_methods:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown calculation: {calculation}. Available: {list(calc_methods.keys())}"
            )
        
        try:
            result = calc_methods[calculation](values)
            return ToolResult(
                success=True,
                data=result,
                metadata={"calculation": calculation, "inputs": values}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    def _calculate_bmi(self, values: dict) -> str:
        """Calculate BMI."""
        weight_kg = values.get("weight_kg")
        height_m = values.get("height_m")
        
        if not weight_kg or not height_m:
            raise ValueError("Need weight_kg and height_m")
        
        bmi = weight_kg / (height_m ** 2)
        
        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25:
            category = "Normal weight"
        elif bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"
        
        return f"BMI: {bmi:.1f} ({category})"
    
    def _calculate_gcs(self, values: dict) -> str:
        """Calculate Glasgow Coma Scale."""
        eye = values.get("eye", 0)  # 1-4
        verbal = values.get("verbal", 0)  # 1-5
        motor = values.get("motor", 0)  # 1-6
        
        total = eye + verbal + motor
        
        if total <= 8:
            severity = "Severe"
        elif total <= 12:
            severity = "Moderate"
        else:
            severity = "Mild"
        
        return f"GCS: {total}/15 (E{eye}V{verbal}M{motor}) - {severity}"
    
    def _unit_convert(self, values: dict) -> str:
        """Convert between units."""
        value = values.get("value")
        from_unit = values.get("from")
        to_unit = values.get("to")
        
        # TODO: Implement more conversions
        conversions = {
            ("kg", "lb"): lambda x: x * 2.205,
            ("lb", "kg"): lambda x: x / 2.205,
            ("cm", "inch"): lambda x: x / 2.54,
            ("inch", "cm"): lambda x: x * 2.54,
            ("c", "f"): lambda x: (x * 9/5) + 32,
            ("f", "c"): lambda x: (x - 32) * 5/9,
        }
        
        key = (from_unit.lower(), to_unit.lower())
        if key not in conversions:
            raise ValueError(f"Unknown conversion: {from_unit} to {to_unit}")
        
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.2f} {to_unit}"

