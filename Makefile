.PHONY: copy-artifacts

copy-artifacts:
	cp ../shoppers-pipeline/data/processed/features_train_sample.parquet data/
	cp ../shoppers-pipeline/data/processed/acquisition_insights.parquet data/
	cp ../shoppers-pipeline/data/processed/retention_insights.parquet data/
	cp ../shoppers-pipeline/data/processed/feature_importance_acq.parquet data/
	cp ../shoppers-pipeline/data/processed/feature_importance_ret.parquet data/
	cp ../shoppers-pipeline/data/processed/selection_bias_anova.parquet data/
	cp ../shoppers-pipeline/data/processed/calibration_acq.parquet data/
	cp ../shoppers-pipeline/data/processed/calibration_ret.parquet data/
	cp ../shoppers-pipeline/data/processed/category_cycle_summary.parquet data/
	cp ../shoppers-pipeline/data/processed/model_metrics.json data/
	cp ../shoppers-pipeline/data/processed/holdout_predictions_acq.parquet data/
	cp ../shoppers-pipeline/data/processed/holdout_predictions_ret.parquet data/
	cp ../shoppers-pipeline/data/processed/val_predictions_acq.parquet data/
	cp ../shoppers-pipeline/data/processed/val_predictions_ret.parquet data/
	cp ../shoppers-pipeline/data/processed/counterfactual_baseline.json data/
	cp ../shoppers-pipeline/data/processed/optimal_threshold_sim.parquet data/
	cp ../shoppers-pipeline/data/processed/optimal_threshold_metrics.json data/
	cp ../shoppers-pipeline/data/processed/shap_values_acq.parquet data/
	cp ../shoppers-pipeline/data/processed/shap_data_acq.parquet data/
	cp ../shoppers-pipeline/data/processed/shap_values_ret.parquet data/
	cp ../shoppers-pipeline/data/processed/shap_data_ret.parquet data/
