# Predicting Vessel Turnaround Time at Brazilian Ports

This README is a short overview. The full work is in
[`vessel_turnaround_prediction.ipynb`](vessel_turnaround_prediction.ipynb), with detailed observations after each step and a summary at the
top.

**Question:** can the information known when a vessel arrives be used to predict how long it will stay in port
(`t_total_port_stay_h`)?

**Short answer:** partly. On the test period a LightGBM model reaches an MAE of 38.0 hours, 16.9% lower than a
"median stay per port" baseline (45.7 hours), with a median absolute error of 15.0 hours. Performance is much weaker
for stays over 7 days: these are 9% of calls but cause about half of the total error.

## Data

| Source | What it gives | Use |
|---|---|---|
| [Brazilian Port Calls & Lead Time 2023–2025](https://www.kaggle.com/datasets/tiagoalberione/brazilian-port-lead-time-2023-2025) (Kaggle, `brazilian_port_calls_model_ready_2023_2025.csv`) | 129,625 port calls (one row = one visit of one vessel to one port), 124 columns: port, calendar, declared operations, port/vessel/route history, congestion, weather | Main dataset |
| Data dictionary (`data_dictionary.csv`), from the dataset's [GitHub repository](https://github.com/tiagoalberione/port-leadtime-prediction) | Description, source, transformation and availability at prediction time for every column | Understanding the columns and checking for leakage |
| [`holidays`](https://pypi.org/project/holidays/) Python library (external) | Brazilian national and state public holidays | Feature `holidays_next_7d`: holidays in the 7 days after arrival, at the port's state |

I used the dataset's own time-based split. Train runs Jan 2023 – Jun 2024 (66,909 calls). Validation runs
Jul 2024 – Jun 2025 (41,714 calls); the dataset's separate calibration period is merged into it. The final test
runs Jul – Dec 2025 (21,002 calls) and was used only once, at the end. All metrics are calculated per record
(row).

## Approach

1. **EDA and data quality (Section 3).**
   - I checked the data dictionary for leakage. Only columns available at arrival are used: past-only history
     (D-1 cutoff), arrival location and calendar, and the declared operations.
   - I checked missing values, constant columns, the target distribution (heavily right-skewed: median 38 h,
     mean 71 h, longest stay 1,280 h or about 53 days) and extreme stays.
   - I identified 1,394 records with different `port_call_id` values but identical port, arrival minute and
     total stay. These may represent multiple operational records for the same physical vessel visit (e.g.
     loading/unloading pairs), although this cannot be confirmed from the available identifiers. Because
     matching records share the arrival time, they fall within the same time split. I checked all 673 groups
     and found no group with records assigned to different splits.
   - I looked at patterns by port, operation, time, vessel history and congestion.
2. **Features (Section 4).** I used the dataset's 119 features, plus the holiday feature and two EDA features
   (`has_vessel_history`, `n_operations`). Missing history was left as missing, since it means "no earlier
   calls". LightGBM handles that directly.
3. **Modelling (Section 5).** Baseline: the median stay per port from training data. Model: LightGBM, with
   early stopping on validation. All choices were made on validation:
   - **Target and loss.** Training with MAE loss gave validation MAE 41.6 h, against 46.5 h with
     squared-error loss. A log target (predictions converted back to hours before
     calculating MAE) gave the same 41.6 h, so the model predicts hours directly.
   - **Feature groups.** Removing the weather, holiday and EDA features changed MAE by 0.07 h or less. That is
     the same as the run-to-run variation between random seeds, so they were dropped (104 features left).
   - **Training-data cleanup.** Removing records with matching port, arrival and stay from the training data,
     as well as stays under 1 h, did not measurably change validation MAE. Since these records have different
     `port_call_id` values and may represent distinct operations, I retained them.
   - **Learning curve and settings grid.** Training MAE was 37.5 h against 41.6 h on validation, a small gap.
     A 3×3 grid of tree size and learning rate stayed within 41.6–41.9 h.
4. **Final evaluation (Section 6).** I retrained on train + validation and evaluated once on the test period:
   overall, by stay length, Santos vs other ports, and feature importance.

## Results (test period, Jul – Dec 2025)

| Model | MAE (h) | RMSE (h) | Median absolute error (h) |
|---|---|---|---|
| Baseline: median stay per port | 45.7 | 92.6 | 20.1 |
| **LightGBM (final)** | **38.0** | **80.0** | **15.0** |

| Actual stay | Calls | Model MAE (h) | Baseline MAE (h) | Model predicts too short | Share of model's total error |
|---|---|---|---|---|---|
| Under 1 day | 6,430 | 14.2 | 18.9 | 19% | 11% |
| 1 – 3 days | 8,581 | 16.7 | 14.1 | 47% | 18% |
| 3 – 7 days | 4,066 | 41.7 | 58.2 | 80% | 21% |
| Over 7 days | 1,925 | 204.1 | 249.1 | 99% | 49% |

- **Where it works:** the model beats the baseline in every stay group except 1–3 days, where the port median
  is already close.
- **Santos (BRSSZ)**, where almost all of the very high congestion in the dataset occurs: the model improves on the baseline by 24%
  (other ports 15%). For stays over 7 days the improvement is 28%, but the MAE is still about 200 h.
- **What drives the predictions:** route history is the strongest group (51% of LightGBM's gain importance,
  measured on training data, so a guide rather than exact). This is the median stay
  of earlier calls on the same previous-port → next-port route. Next come the port itself (17%), vessel
  history (10%), port history (7%) and declared operations (6%). Congestion (3%) and calendar (3%)
  contribute little overall.

## Did the additional features help?

No, not measurably.
- **Holidays (external source).** Calls arriving in Carnival and Christmas weeks stay only about 2% longer
  than usual for their port. Adding the feature changed validation MAE by 0.01 h.
- **Weather and my EDA features.** These changed validation MAE by 0.07 h or less, within run-to-run
  variation.

I report this as a negative result: under this validation setup, the holiday, weather and EDA features gave
no measurable improvement. Their extra information may be small, already represented in the history
features, or hard for the model to use.

## Limitations

- **Very long stays are poorly predicted with the available features.** Stays over 7 days are almost always
  underpredicted, which suggests that important drivers are missing from the dataset, for example vessel
  type and size, cargo volume, berth availability, maintenance or commercial waiting. Part of the
  underprediction also comes from the MAE loss: the model predicts the typical (median) stay for similar
  calls, so rare very long stays are predicted too short by design.
- **No vessel characteristics.** The raw source (Porto Sem Papel, dados.transportes.gov.br) has vessel name and
  IMO (often blank), but no type, size or cargo. Those would need paid registries.
- **Declared operations (`op_*`) are assumed to be known at arrival.** The dictionary treats them as
  arrival-time information. If they are registered later, results would be slightly optimistic. The same
  applies to the route features, which use the reported next port.
- **Point predictions only.** For planning, a range would be more useful than one number, especially given
  the long tail.

## Possible improvements

- **Prediction intervals** (e.g. quantile LightGBM, calibrated on the dataset's calibration period), so a
  planner sees "likely 20–40 h, risk of over 7 days".
- **A separate model for long-stay risk.** Flag calls likely to exceed 7 days rather than predicting their
  exact length.
- **More external data:**
  - vessel type and size via IMO;
  - [Comex Stat](https://comexstat.mdic.gov.br/) monthly export volumes per port (e.g. soy season at Santos);
  - tide data.

## How to run

Python 3.11 with `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `lightgbm` (4.7) and `holidays`:

```bash
pip install pandas numpy matplotlib scikit-learn lightgbm holidays
```

Place `brazilian_port_calls_model_ready_2023_2025.csv` and `data_dictionary.csv` (see Data above) next to
the notebook, then run all cells. A full run takes a few minutes.

