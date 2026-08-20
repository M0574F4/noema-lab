# Benchmark Suites

Noema uses suites to group benchmark protocols, recipes, adapter points, metrics, and documentation
around a coherent research direction.

The umbrella is learned-communication evaluation. Semantic communication is the active-development
flagship suite. Experimental suites run end to end with benchmark packs, baselines, metrics, plots,
and adapter points, but their protocols may still change before becoming canonical.

| Suite | Maturity | Scope |
| --- | --- | --- |
| [Semantic Communication](semantic_comm.md) | Active development | meaning/task-aware communication, DeepJSCC, protected digital baselines |
| [Neural Receiver / AI-PHY](neural_receiver.md) | Experimental | QPSK/AWGN receiver adapters, BER/BLER |
| [Channel Estimation](channel_estimation.md) | Experimental | pilot-limited estimation, LS baseline, learned-estimator adapter |
| [MIMO-OFDM](mimo_ofdm.md) | Experimental | OFDM/MIMO-style pilot channel estimation |
| [Beamforming / Precoding](beamforming_precoding.md) | Experimental | MISO beam selection, MRT/codebook baselines, learned policy adapters |
| [Localization / Sensing](localization_sensing.md) | Experimental | range localization, trilateration baseline, sensing adapters |
| [Resource Allocation](resource_allocation.md) | Experimental | OFDM subcarrier power allocation, water-filling oracle, policy adapters |

```{toctree}
:maxdepth: 1
:hidden:

semantic_comm
neural_receiver
channel_estimation
mimo_ofdm
beamforming_precoding
localization_sensing
resource_allocation
```

## Understanding Maturity

**Active development** indicates the maintained flagship surface with runnable protocols,
documented baselines, task-specific metrics, external adapter points, and verification coverage.
Compatibility and publication status are not implied before a protocol is separately frozen and
admitted as canonical.

**Experimental** indicates that the documented workflow is runnable and produces locally verifiable result
bundles, while benchmark details may change between releases. Experimental results are suitable for
evaluation and method development but are not automatically publication-ready or leaderboard
compatible.
