# 3D Print Quality Predictor
## Test Results

Testing was performed throughout development to verify the data-processing pipeline, database, machine-learning model, Streamlit interface, decision-support workflow, security controls, error handling, and system monitoring.

All tests listed below passed. Supporting screenshots are stored in the `documentation/testing` directory and use the corresponding test IDs.

| Test ID | Test | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| T01 | Data cleaning validation | The cleaning script should retain all 500 source records, standardize the required fields, and flag the 17 records with zero roughness values without deleting them. | The cleaned dataset contained all 500 source records. Seventeen zero-roughness records were retained and flagged with `roughness_valid = False`. | PASS |
| T02 | SQLite database validation | The database should contain all 500 cleaned records, including 483 records available for roughness analysis/modeling and 17 flagged zero-roughness records. | The `print_records` table contained 500 total records: 483 with `roughness_valid = 1` and 17 with `roughness_valid = 0`. | PASS |
| T03 | Model training | The training script should use only the 483 records with positive roughness values, create an 80/20 split, train the Random Forest model, and save the model and metrics files. | The model used 483 records, producing 386 training records and 97 test records. `roughness_model.joblib` and `model_metrics.json` were created successfully. | PASS |
| T04 | Model evaluation | The saved model should reproduce the expected held-out test metrics without retraining. | Evaluation produced MAE = 0.8208 µm, RMSE = 1.1305 µm, and R² = 0.7234 on 97 held-out test records. | PASS |
| T05 | Overview page | The Overview page should load successfully and display the expected dataset summary information. | The page displayed 500 total print records, 483 positive roughness records, 10 materials, and 14 infill patterns. | PASS |
| T06 | Data Explorer filtering | Changing Data Explorer filters should update the matching records, summary statistics, data table, and visualizations. | A PETG filter was applied and the displayed records, statistics, and charts updated to match the selected data. | PASS |
| T07 | Prediction meets target | A prediction at or below the selected maximum roughness target should be identified as meeting the target. | The predictor generated a roughness estimate below the selected target and displayed the correct “meets target” decision-support message. | PASS |
| T08 | Prediction does not meet target | A prediction above the selected maximum roughness target should be identified as not meeting the target. | The predictor generated a roughness estimate above the selected target and displayed the correct “does not meet target” decision-support message. | PASS |
| T09 | Model Performance page | The Model Performance page should load the saved metrics and evaluation results and display model statistics and visualizations. | The page displayed the Random Forest test metrics, baseline comparison, five-fold cross-validation results, accuracy bands, predicted-versus-measured chart, residual distribution, and largest prediction errors. | PASS |
| T10 | Normal system status | When the database, model, and metrics files are available, the Overview page should report all components as available and the overall system as Operational. | The database, trained model, and metrics file were shown as available and the overall status was Operational. | PASS |
| T11 | Missing-model failure handling | Temporarily removing the trained model should not crash the application. The system should identify the missing component and report a Degraded status. | The model file was temporarily renamed. The application remained available, identified the trained model as unavailable, and changed the overall status to Degraded. | PASS |
| T12 | Predictor material restriction | rPET should not be offered as a prediction material because all of its source roughness records are among the zero-value records excluded from model development. | rPET was not present in the Quality Predictor material options. The interface explains why it is unavailable for prediction. | PASS |
| T13 | Supported predictor inputs | Numeric predictor controls should only allow values represented in the records used for model development. | The predictor restricted layer thickness to 0.1, 0.2, and 0.3 mm; infill density to 20%, 50%, and 80%; and print speed to 30, 50, and 70 mm/s. | PASS |
| T14 | Read-only SQLite access | The application database connection should permit read operations while preventing write operations. | A `SELECT` query successfully returned the 500 database records. An attempted `INSERT` was rejected with a SQLite read-only database error. | PASS |
| T15 | Empty-filter handling | A Data Explorer filter combination with no matching records should display a user-friendly message instead of producing an application error. | The application displayed “No records match the selected filters.” and remained operational. | PASS |

## Model Verification Results

The final Random Forest model was evaluated on a held-out test set that was not used for training.

- Modeling records: **483**
- Training records: **386**
- Test records: **97**
- Random Forest MAE: **0.8208 µm**
- Random Forest RMSE: **1.1305 µm**
- Random Forest R²: **0.7234**
- Five-fold cross-validation MAE: **0.7799 µm**
- Five-fold cross-validation RMSE: **1.0663 µm**
- Five-fold cross-validation R²: **0.7301**
- Predictions within ±1.0 µm: **74.2%**
- Predictions within ±1.5 µm: **84.5%**
- Predictions within ±2.0 µm: **89.7%**

An earlier set of recorded model metrics did not match the separate evaluation results. The training and evaluation scripts were reviewed to verify that the held-out test records were not used during training. The scripts were then rerun, and the final results above were reproduced consistently.

## Revisions Resulting From Testing

Testing and code review resulted in several application revisions:

- Zero roughness records were retained in the cleaned dataset rather than deleted and were flagged so they could be excluded from roughness model development.
- Only process settings available before printing were used as model features to avoid data leakage.
- Predictor material and numeric controls were restricted to values represented in the records used for model development.
- rPET was removed from prediction options because all of its source roughness records were excluded from model development.
- SQLite access used by the application was changed to read-only/query-only mode.
- User-selected SQL values remained parameterized, and dynamic SQL column names were restricted through application-defined allowlists.
- User-facing error messages were simplified while technical details continued to be written to the application log.
- Database connections were closed after queries.
- Static option queries and roughness ranges were cached.
- The trained model was cached and configured to refresh when the model file changes.
- Application startup logging was limited to once per Streamlit session instead of once per rerun.

## Final Result

All 15 documented tests passed. The completed application successfully loaded and queried the prepared FDM dataset, generated surface-roughness predictions, compared predictions with user-selected quality targets, displayed the required interactive visualizations and model-performance information, enforced the tested database and input protections, and handled the tested failure and empty-result conditions without crashing.