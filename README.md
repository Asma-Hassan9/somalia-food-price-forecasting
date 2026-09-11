# Somalia Food Price Alert System - Deployment Package
Generated on: 2025-09-26 06:56:36

## Files Included:

### Core Alert Data:
- somalia_food_price_alerts.csv - Complete alerts dataset with regional mapping
- somalia_crisis_alerts.csv - Crisis-level alerts requiring immediate action
- somalia_regional_alert_summary.csv - Regional risk analysis summary
- somalia_commodity_risk_analysis.csv - Commodity risk breakdown

### Threshold Data:
- somalia_primary_thresholds.csv - Commodity-season specific thresholds
- somalia_secondary_thresholds.csv - Commodity-only fallback thresholds

### Mapping Files:
- commodity_mappings.csv - Commodity code to name mapping
- market_mappings.csv - Market code to name mapping  
- district_region_mappings.csv - District to region administrative mapping

### System Reports:
- system_health_report.csv - Alert system performance metrics
- model_performance_summary.csv - ML model accuracy metrics

## Alert System Summary:
- Total Predictions: 5356
- Total Alerts: 1595
- Crisis Alerts: 400
- Regional Coverage: 12 regions
- Model Accuracy: R² = 0.9588

## Usage:
1. Load alerts data for operational response
2. Use regional summary for policy decisions  
3. Apply thresholds for continued monitoring
4. Reference mappings for readable reports
