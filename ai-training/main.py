"""
Main orchestrator script for brain tumor classification.
1. Trains model on training data (85% train / 15% validation split)
2. Tests model on separate testing dataset
3. Compares and analyzes results from both evaluations
4. Generates comprehensive report
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from config.config import config
from utils.helpers import setup_logger
from main_train import main as train_main
from main_test import main as test_main


def compare_results(val_metrics: dict, test_metrics: dict, output_dir: str, logger):
    """
    Compare validation and test results, detect any issues or discrepancies.
    
    Args:
        val_metrics: Metrics from validation set (15% split)
        test_metrics: Metrics from test set (separate testing locations)
        output_dir: Directory to save comparison report
        logger: Logger instance
    """
    logger.info("="*60)
    logger.info("COMPARING VALIDATION AND TEST RESULTS")
    logger.info("="*60)
    
    # Key metrics to compare
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
    
    comparison = {
        'validation_metrics': {k: val_metrics[k] for k in metrics_to_compare},
        'test_metrics': {k: test_metrics[k] for k in metrics_to_compare},
        'differences': {},
        'warnings': [],
        'analysis': {}
    }
    
    # Calculate differences
    for metric in metrics_to_compare:
        val_value = val_metrics[metric]
        test_value = test_metrics[metric]
        diff = test_value - val_value
        diff_percent = (diff / val_value * 100) if val_value > 0 else 0
        
        comparison['differences'][metric] = {
            'absolute': float(diff),
            'percent': float(diff_percent)
        }
    
    # Detect potential issues
    logger.info("\nMetric Comparison:")
    logger.info(f"{'Metric':<15} {'Validation (15%)':<20} {'Test Set':<20} {'Difference':<20} {'% Change':<15}")
    logger.info("-" * 90)
    
    for metric in metrics_to_compare:
        val_val = comparison['validation_metrics'][metric]
        test_val = comparison['test_metrics'][metric]
        diff = comparison['differences'][metric]['absolute']
        diff_pct = comparison['differences'][metric]['percent']
        
        logger.info(f"{metric.capitalize():<15} {val_val:<20.4f} {test_val:<20.4f} {diff:+.4f} ({diff_pct:+.2f}%)")
        
        # Check for significant discrepancies (>10% drop)
        if diff_pct < -10:
            warning = f"Significant drop in {metric}: {diff_pct:.2f}% decrease on test set"
            comparison['warnings'].append(warning)
            logger.warning(f"  ⚠ {warning}")
        
        # Check for significant improvements (might indicate overfitting on validation)
        elif diff_pct > 10:
            info = f"Significant improvement in {metric}: {diff_pct:.2f}% increase on test set"
            comparison['analysis'][f'{metric}_improvement'] = info
            logger.info(f"  ℹ {info}")
    
    # Overall analysis
    accuracy_diff_pct = comparison['differences']['accuracy']['percent']
    f1_diff_pct = comparison['differences']['f1']['percent']
    
    logger.info("\nOverall Analysis:")
    
    # Model performance consistency
    if abs(accuracy_diff_pct) < 5:
        analysis = "Model shows consistent performance between validation and test sets (good generalization)"
        comparison['analysis']['consistency'] = analysis
        logger.info(f"  ✓ {analysis}")
    elif accuracy_diff_pct < -10:
        analysis = "Model performance drops significantly on test set - possible overfitting or data distribution mismatch"
        comparison['analysis']['overfitting_warning'] = analysis
        comparison['warnings'].append(analysis)
        logger.warning(f"  ⚠ {analysis}")
    else:
        analysis = f"Model shows {abs(accuracy_diff_pct):.2f}% difference between validation and test sets"
        comparison['analysis']['performance_variance'] = analysis
        logger.info(f"  ℹ {analysis}")
    
    # Per-class comparison
    logger.info("\nPer-Class Comparison:")
    logger.info(f"{'Class':<15} {'Val Precision':<18} {'Test Precision':<18} {'Val Recall':<18} {'Test Recall':<18}")
    logger.info("-" * 90)
    
    val_precisions = val_metrics.get('per_class_precision', [])
    val_recalls = val_metrics.get('per_class_recall', [])
    test_precisions = test_metrics.get('per_class_precision', [])
    test_recalls = test_metrics.get('per_class_recall', [])
    
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    class_comparison = {}
    for i, class_name in enumerate(class_names):
        if i < len(val_precisions) and i < len(test_precisions):
            val_prec = val_precisions[i]
            test_prec = test_precisions[i]
            val_rec = val_recalls[i] if i < len(val_recalls) else 0
            test_rec = test_recalls[i] if i < len(test_recalls) else 0
            
            prec_diff = test_prec - val_prec
            rec_diff = test_rec - val_rec
            
            logger.info(f"{class_name:<15} {val_prec:<18.4f} {test_prec:<18.4f} {val_rec:<18.4f} {test_rec:<18.4f}")
            
            class_comparison[class_name] = {
                'precision': {'val': float(val_prec), 'test': float(test_prec), 'diff': float(prec_diff)},
                'recall': {'val': float(val_rec), 'test': float(test_rec), 'diff': float(rec_diff)}
            }
            
            # Check for problematic classes
            if prec_diff < -0.15 or rec_diff < -0.15:
                warning = f"Class '{class_name}' shows significant performance drop on test set"
                comparison['warnings'].append(warning)
                logger.warning(f"  ⚠ {warning}")
    
    comparison['per_class_comparison'] = class_comparison
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    if len(comparison['warnings']) > 0:
        logger.warning(f"\n⚠ Found {len(comparison['warnings'])} warning(s):")
        for warning in comparison['warnings']:
            logger.warning(f"  - {warning}")
    else:
        logger.info("\n✓ No major issues detected. Model shows good generalization.")
    
    logger.info(f"\nValidation Set Performance (15% split):")
    logger.info(f"  Accuracy: {comparison['validation_metrics']['accuracy']:.4f}")
    logger.info(f"  F1-Score: {comparison['validation_metrics']['f1']:.4f}")
    
    logger.info(f"\nTest Set Performance (separate testing locations):")
    logger.info(f"  Accuracy: {comparison['test_metrics']['accuracy']:.4f}")
    logger.info(f"  F1-Score: {comparison['test_metrics']['f1']:.4f}")
    
    # Save comparison report
    comparison_path = os.path.join(output_dir, "results_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"\nComparison report saved to: {comparison_path}")
    
    # Save text report
    report_path = os.path.join(output_dir, "comparison_report.txt")
    with open(report_path, 'w') as f:
        f.write("Brain Tumor Classification - Results Comparison Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERALL METRICS COMPARISON\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Metric':<15} {'Validation (15%)':<20} {'Test Set':<20} {'Difference':<20}\n")
        f.write("-"*60 + "\n")
        for metric in metrics_to_compare:
            val_val = comparison['validation_metrics'][metric]
            test_val = comparison['test_metrics'][metric]
            diff = comparison['differences'][metric]['absolute']
            f.write(f"{metric.capitalize():<15} {val_val:<20.4f} {test_val:<20.4f} {diff:+.4f}\n")
        
        f.write("\nANALYSIS\n")
        f.write("-"*60 + "\n")
        for key, value in comparison['analysis'].items():
            f.write(f"{value}\n")
        
        if len(comparison['warnings']) > 0:
            f.write("\nWARNINGS\n")
            f.write("-"*60 + "\n")
            for warning in comparison['warnings']:
                f.write(f"⚠ {warning}\n")
        
        f.write("\nPER-CLASS COMPARISON\n")
        f.write("-"*60 + "\n")
        for class_name, metrics in class_comparison.items():
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: Val={metrics['precision']['val']:.4f}, Test={metrics['precision']['test']:.4f}, Diff={metrics['precision']['diff']:+.4f}\n")
            f.write(f"  Recall:    Val={metrics['recall']['val']:.4f}, Test={metrics['recall']['test']:.4f}, Diff={metrics['recall']['diff']:+.4f}\n")
    
    logger.info(f"Comparison report (text) saved to: {report_path}")
    
    return comparison


def main():
    """Main orchestrator function."""
    
    # Setup logging
    logger = setup_logger("brain_tumor_main", config.LOG_DIR)
    logger.info("="*60)
    logger.info("BRAIN TUMOR CLASSIFICATION - MAIN PIPELINE")
    logger.info("="*60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Training
    logger.info("="*60)
    logger.info("STEP 1: TRAINING MODEL")
    logger.info("="*60)
    logger.info("Training model on training data with 85/15 split...")
    logger.info("(85% for training, 15% for validation)\n")
    
    try:
        val_metrics, best_model_path = train_main()
        logger.info("\n✓ Training completed successfully")
        logger.info(f"Best model saved at: {best_model_path}\n")
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Step 2: Testing
    logger.info("="*60)
    logger.info("STEP 2: TESTING MODEL")
    logger.info("="*60)
    logger.info("Testing model on separate testing dataset...\n")
    
    try:
        test_metrics = test_main(model_path=best_model_path)
        logger.info("\n✓ Testing completed successfully\n")
    except Exception as e:
        logger.error(f"\n✗ Testing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Step 3: Comparison and Analysis
    logger.info("="*60)
    logger.info("STEP 3: COMPARING RESULTS")
    logger.info("="*60)
    
    try:
        comparison = compare_results(val_metrics, test_metrics, config.OUTPUT_DIR, logger)
        logger.info("\n✓ Comparison completed successfully\n")
    except Exception as e:
        logger.error(f"\n✗ Comparison failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Final summary
    logger.info("="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    logger.info("Results Summary:")
    logger.info(f"  Validation Accuracy (15% split): {val_metrics['accuracy']:.4f}")
    logger.info(f"  Test Accuracy (separate test set): {test_metrics['accuracy']:.4f}")
    logger.info(f"  Accuracy Difference: {comparison['differences']['accuracy']['absolute']:+.4f} ({comparison['differences']['accuracy']['percent']:+.2f}%)")
    
    logger.info(f"\nOutput files saved to: {config.OUTPUT_DIR}")
    logger.info("  - training_validation_results.json: Validation set results (15%)")
    logger.info("  - testing_results.json: Test set results")
    logger.info("  - results_comparison.json: Comparison analysis")
    logger.info("  - comparison_report.txt: Human-readable comparison report")
    
    if len(comparison['warnings']) > 0:
        logger.warning(f"\n⚠ {len(comparison['warnings'])} warning(s) detected - see comparison report for details")
    
    logger.info("\n" + "="*60)


if __name__ == "__main__":
    main()

