# FMCSA Vehicle Inspection snapshot (CS6140 midterm data)

Data files used by my CS6140 midterm notebook, "Will This Truck Fail Inspection?".

- Source: FMCSA Vehicle Inspection File, U.S. DOT open data portal:
  https://data.transportation.gov/Trucking-and-Motorcoaches/Vehicle-Inspection-File/fx4q-ay7w
- `fmcsa_vehicle_inspections_fy2026_sample.csv.gz`: 200,000 inspections (levels 1, 2, 5, 6, fiscal year 2026), all 63 original columns. Built by `prepare_fmcsa.py`.
- `prep_counts.json`: row counts and rates at each filtering stage.
- `carrier_history.csv.gz` and `carrier_history_meta.json`: each sampled carrier's inspection counts from before Oct 1, 2025. Built by `prepare_carrier_history.py`.
- `SNAPSHOT_DATE.txt`: date the full file was downloaded.

U.S. government open data. Muhammad Marzouk Baig, Northeastern University.
