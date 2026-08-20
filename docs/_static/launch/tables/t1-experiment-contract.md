### T1 · Experiment contract summary

| Contract field | Declared value | Evidence path |
| --- | --- | --- |
| Experiment | Learned QPSK Receiver Calibration | identity.title |
| Result status | completed | identity.result_status |
| Methods and roles | Uncompensated QPSK (baseline); Calibrated I/Q oracle (reference); Learned I/Q receiver (candidate) | design.methods |
| SNR cells (dB) | -2, 0, 2, 4, 6, 8, 10 | design.snr_db |
| Paired seeds | 71001, 72001, 73001 | design.paired_seeds |
| Retained runs | 63 | design.run_count |
| Compared bits per run | 1,048,576 | design.bits_per_run |
| Aggregation | arithmetic_mean_across_paired_seeds | scientific_status.aggregation |
| Statistical unit | paired held-out seed within a predeclared SNR cell | design.statistical_unit |
| Uncertainty display | observed_min_max_not_confidence_interval | scientific_status.uncertainty_display |
| Evidence tier | completed_experimental_benchmark | scientific_status.evidence_level |
| Publication ready | false | scientific_status.publication_ready |
| Distribution clearance | candidate | distribution_clearance.status |
| Reference role | The calibrated I/Q oracle is a diagnostic reference with calibration knowledge, not a deployable competitor using the same information. | scientific_status.reference_role |

Distribution clearance and scientific status are independent.
