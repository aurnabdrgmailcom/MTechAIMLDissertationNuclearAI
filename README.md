Code, Datasets and their relevance:

1)Chapter_5_Code

Physics - Grounded Surrogate Dataset for Subcritical Thorium - U233 Reactor Governance

Description:

This repository contains the source code and generated datasets for Chapter 5 of the dissertation: "Intelligent Computational System for Constraint - First Governance of Subcritical Thorium - U233 Reactors."

The code implements a "Physics - Grounded" data generation pipeline used to train surrogate models for reactor safety. It utilizes a One - Group Diffusion Theory approximation with Bateman depletion logic to simulate the steady-state physics of a Cylindrical Thorium - U233 core.

Key Features:

Methodology: Rejection sampling on random geometries (core radius R, height H, blanket thickness t, and initial U - 233 fraction).
Safety Filter:
The generator strictly rejects unsafe supercritical designs k_eff  >= 0.98 and physically impossible states k_eff  >= 1.0. It also rejects designs exceeding Power (500 MWth) or average core power density (100 W/cm³) envelopes.
Physics Engine: Custom Python implementation of analytical flux solutions for cylindrical coordinates, coupled with ENDF/ B - VIII.0 nuclear data cross - sections.
Output: A verified dataset of 1,500 subcritical reactor configurations, labelled with k_eff, Total Power, Power Density, and Breeding proxies (bred U-233 atoms after 1 day, plus a normalization by initial Th inventory).

File Structure & Contents:

1. Source Code

Step5_Dataset_Generation.ipynb: The primary Jupyter Notebook that runs the physics simulation. It handles:
Anchoring nuclear constants to ENDF/B-VIII.0 values and saving them to nuclear_data.json for reproducibility (the run uses the in-memory nuc object). Defining the reactor geometry bounds (Radius: 50–150 cm, Height: 100–300 cm).
Evaluating a closed - form one - group diffusion proxy with cylindrical buckling (fundamental mode) for 15,000+ random candidates.
Filtering for the final 1,500 safe samples.

2. Configuration & Metadata

dataset_schema.json: Defines the physical units, bounds, and descriptions for every column in the dataset (e.g., core_radius_cm, keff_est).
material_library.json: Specifies the material compositions (Fuel: Th -U233 mix, Moderator: D2O, Reflector: Graphite) and number densities.
nuclear_data.json: Contains the microscopic cross-sections σ_f, σ_c, ν derived from ENDF/B-VIII.0 for U - 233 and Th -232.
exfor_cache.sqlite: Legacy Artifact. An empty SQLite database file. (Note: The initial design planned to fetch data dynamically from the IAEA EXFOR API, but the final implementation standardized on static ENDF values for reproducibility. This file is retained for code compatibility).

3. The Dataset

surrogate_dataset.parquet: The primary dataset file (Apache Parquet format) containing the 1,500 samples. Recommended for use with Pandas (pd.read_parquet).
surrogate_dataset.csv: A CSV version of the same dataset for human readability.
surrogate_dataset.npz: A NumPy archive containing the data arrays for direct matrix manipulation.

Usage Notes:

To reproduce the dataset, open Step5_Dataset_Generation.ipynb and run all cells.
The safety_margin column in the dataset represents the normalized distance to the nearest safety constraint (keff, power, power-density)
Warning: This dataset is based on a steady-state diffusion approximation ("Gedanken" model) intended for AI governance research, not for licensing real - world reactor designs without high - fidelity validation (e.g., MCNP/OpenMC).

2)Chapter_6_Code

Description:

This repository contains the source code and augmented datasets for Chapter 6 of the dissertation: "Intelligent Computational System for Constraint - First Governance of Subcritical Thorium - U233 Reactors"

To overcome the sparsity of the original physics simulated dataset (Chapter 5), this module implements an Epistemic Uncertainty Aware Data Expansion strategy. It utilizes a Deep Ensemble of 9 neural network regressors (scikit - learn MLPRegressor) to explore the reactor design space and generate synthetic samples. Crucially, it filters these samples based on prediction variance, ensuring that only high - confidence, physically valid designs are added to the training pool.

Key Methodology:

1.      Candidate Generation: Uses Latin Hypercube Sampling (LHS) / Random Uniform Sampling to propose 25000 theoretical reactor geometries within the design bounds.

2.      Deep Ensemble Filter: A committee of 9 independent regressors predicts the physics parameters (k_eff, Power, Breeding Proxy, flux_shape_metric and avg_core_power_density_W_cm3) for each candidate.

3.      Uncertainty Gating: Candidates are accepted only if their predictive variance σ² is within the 90th percentile of the real data's error distribution. This rejects hallucinated data points in out of distribution regions.

4.      Safety Compliance: Strict rejection of any synthetic design violating safety envelopes k_eff >= 0.98, Power > 500 MW, power density >=100 W/cm³).

Outcome:

The process successfully expanded the dataset from 1,500 real samples to 6,841 total samples (adding 5,341 validated synthetic points), significantly smoothing the design manifold for downstream surrogate training.

File Structure & Contents:

1. Source Code

Chapter_6_Synthetic_Data_Augmentation.ipynb: The Jupyter Notebook implementing the Deep Ensemble logic. It trains the ensemble, calculates variance metrics, filters the candidate pool, and merges the datasets.

2. Reports & Metrics

augmentation_report.json: Detailed metadata on the process.
Ensemble Members: 9
Uncertainty Gate: 90th percentile (0.9)
Synthetic Candidates: 25,000
Accepted: 5,341 (21.4% acceptance rate)
ensemble_test_metrics.json: Validation performance of the ensemble on held-out real data (e.g., k_eff R² ≈ 0.9967), confirming the filter's reliability.

3. The Datasets

augmented_dataset.parquet: (The Primary Artifact) The final combined dataset (1,500 Real + 5,341 Synthetically generated samples = 6,841 samples). Contains a Boolean column is_synthetic to distinguish sources.
synthetic_accepted.parquet: Subset containing only the 5,341 accepted synthetic designs.
synthetic_candidates.parquet: The raw pool of 25,000 candidates (useful for studying what the model rejected).

4. Documentation

Chapter_6_Code_Output.pdf: A PDF capture of the execution logs, showing distribution plots (Real vs. Synthetic) and safety verification steps.

Usage Notes:

Use augmented_dataset.parquet for training high - capacity machine learning models (like Transformers or Neural Networks) that require more data than the original physics simulator provided.
The is_synthetic flag allows researchers to use curriculum learning (training on synthetic first, fine - tuning on real).

3)Chapter_7_Code

Description:

This repository contains the complete source code, comparative benchmarks, and final deployed model artifacts for Chapter 7 of the dissertation: "Intelligent Computational System for Constraint - First Governance of Subcritical Thorium - U233 Reactors."

The ICS serves as the a surrogate based digital twin for rapid design space governance for the reactor design process. It replaces computationally expensive Monte Carlo simulations with fast, differentiable surrogate models. This module covers the entire lifecycle of the ICS:

1.      Benchmarking: A rigorous comparison between Deep Learning (FT - Transformer) and Gradient Boosting (XGBoost) architectures.

2.      Selection: The adoption of XGBoost as the superior engine for this tabular physics domain (R^2 > 0.99$).

3.      Safety Calibration: The calculation of Conformal Prediction residuals to guarantee safety envelopes with 90% confidence.

Key Research Findings:

Model Selection: On the 6,841-sample dataset, the XGBoost ensemble achieved near - perfect linearity (R^2=0.995 for k_eff), whereas the FT - Transformer struggled (R^2=0.20), validating the efficiency of tree-based methods for low-dimensional nuclear data.
Safety Assurance: The system calibrated a safety buffer of 0.027 Δk (2737 pcm) for criticality predictions, ensuring that the 90% upper confidence bound never exceeds the limit of k_eff=0.98.

File Structure & Contents:

1. Source Code (The Engine)

Chapter_7_ICS_Training.ipynb: The master notebook that:
Loads the augmented dataset.
Trains both candidate architectures (XGBoost & FT - Transformer).
Performs Conformal Calibration on the validation set.
Serializes the final models and scalers.

2. Final Deployed Models (The Active ICS)

xgb_keff_est.joblib: Predicts effective multiplication factor (k_{eff}).
xgb_total_power_W.joblib: Predicts total thermal power output.
xgb_flux_shape_metric.joblib: Predicts neutron flux peaking factors.
xgb_breeding_proxy_norm.joblib: Predicts fissile fuel breeding potential.
xgb_avg_core_power_density_W_cm3.joblib: Predicts thermal intensity constraints.

3. Safety Calibration Artifacts (The "Constraint-First" Logic)

xgb_conformal_residual_quantiles.json: (Critical File) Contains the calculated prediction intervals q_(α=0.1) for each physics target. These values are added to point predictions during inference to create the Safe Upper Boundary.
Example: k_eff_safe = k_eff_pred + 0.02737

4. Research Artifacts (Deep Learning Benchmarks)

ft_transformer_state.pt: Saved weights of the experimental Feature Tokenizer Transformer (for reproducibility of negative results).
ft_test_metrics.json vs xgb_test_metrics.json: Comparative logs showing the performance gap (e.g., XGBoost RMSE = 0.017 vs Transformer RMSE = 0.229).

4) Chapter_8_Code

Description:

This repository contains the source code, evaluation reports, and final optimized design candidates for Chapter 8 of the dissertation: "Intelligent Computational System for Constraint - First Governance of Subcritical Thorium - U233 Reactors."

This module represents the operational deployment of the ICS. It loads the trained XGBoost surrogates (from Chapter 7) and subjects them to a series of rigorous tests to validate their utility as a Safety - First governance tool. It performs two critical functions:

1.      Safety Verification: It runs specific "Stress Test" scenarios (e.g., maximizing power, pushing breeding limits) to verify that the Conformal Prediction intervals correctly flag unsafe designs.

2.      Bounded Optimization (Discovery): It scans thousands of random candidates to identify optimal reactor configurations and then constructs SAFE scenarios from a large pool by selecting max breeding and max power among candidates that already pass conservative gates, and then adds a near-miss reject case where the upper bound crosses k_eff=0.98

Key Results:

Safety Coverage: The system demonstrated an empirical coverage of 93.3% on test data, exceeding the target 90% confidence level.
Out – of - Distribution (OOD) Detection: Successfully flagged Boundary scenarios as high - risk OOD distance (scaled) > p95 threshold, preventing the model from making overconfident predictions on unvalidated geometries.

File Structure & Contents:

1. Source Code (The Evaluator)

Chapter_8_EvaluatingICS.ipynb: The operational notebook that:
Loads the 5 trained XGBoost models and the Conformal Safety Quantiles.
Defines specific test scenarios ("Baseline", "High Power", "High Breeding").
Runs a large-scale scan of random candidates to find optimal designs.
Generates safety reports and OOD warnings.

2. Evaluation Reports (The Evidence)

chapter8_coverage_report.json: Statistical validation showing the Hit Rate of the safety intervals (e.g., k_eff coverage = 0.933).
chapter8_scenarios.json & chapter8_scenario_predictions.csv: The inputs and outputs for the specific case studies. It shows exactly how the system predicted the safety margins for the "Baseline_SAFE", "Breeding_SAFE", and "Boundary_SAFE" designs.
chapter8_report.json: Master summary of the evaluation parameters (Safety Envelopes, Domain Bounds).

3. Optimization Results (The "Discovery")

chapter8_candidate_rankings.csv: (High Value Artifact) A ranked list of the top reactor designs found by the ICS.
Columns include: rank_type (e.g., top_breeding), keff_est_mean, keff_est_upper (Safety Bound), and physical dimensions ($R, H$).
Usage: These rows represent the "recommended designs" that a human engineer would then verify with high-fidelity physics codes.

4. Documentation

Chapter_8_EvaluatingICS.ipynb - Colab.pdf: A PDF capture of the execution logs, providing a visual proof of the code's successful run.
