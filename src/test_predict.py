"""
Test script for the predict module.
Runs 3 diverse examples through the pipeline.
"""

import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_report, load_artifacts, set_artifacts

def main():
    examples = [
        {
            "id": "1 - Clearly Dangerous (SIF)",
            "text": "Operator was working on the scaffolding at level 3. The safety harness lanyard snapped, causing the worker to fall 15 meters to the concrete floor below. Victim suffered severe head trauma and multiple fractures.",
            "industry_sector": "Mining",
            "employee_type": "Employee"
        },
        {
            "id": "2 - Minor Incident (Non-SIF)",
            "text": "While walking through the office corridor, the employee slipped on a small puddle of spilled coffee. He twisted his ankle slightly but was able to walk it off. Applied an ice pack at the nursing station.",
            "industry_sector": "Corporate",
            "employee_type": "Third Party"
        },
        {
            "id": "3 - Ambiguous / Potential Near Miss",
            "text": "During routine maintenance of the conveyor belt, the lockout/tagout (LOTO) procedure was initiated. However, a residual charge remained in the system. When the technician touched the panel, he received a mild electrical shock but no burns.",
            "industry_sector": "Metals",
            "employee_type": "Employee"
        }
    ]
    
    print("=" * 60)
    print("  Testing predict_report() Pipeline")
    print("=" * 60)
    
    set_artifacts(load_artifacts())
    
    for ex in examples:
        print(f"\n[{ex['id']}]")
        print(f"Input Text: \"{ex['text']}\"")
        print("-" * 40)
        
        result = predict_report(
            text=ex['text'],
            industry_sector=ex['industry_sector'],
            employee_type=ex['employee_type']
        )
        
        # Truncate NN texts for clean printing
        for nn in result['nearest_neighbors']:
            if len(nn['text']) > 100:
                nn['text'] = nn['text'][:97] + "..."
                
        print(json.dumps(result, indent=2))
        print("=" * 60)

if __name__ == "__main__":
    main()
