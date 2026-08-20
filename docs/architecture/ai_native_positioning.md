# What Noema Covers

Noema is an experiment-contract and evidence system for learned-communication comparisons. It links
declared protocols, concrete execution plans, communication-resource accounting, optional
returned-model interfaces, and retained result and plot evidence across semantic communication,
learned physical-layer components, channel estimation, resource allocation, sensing, and related
wireless-AI research.

Semantic communication is Noema's flagship suite, but it is not the boundary of the system. The
same recipe, benchmark, capture, training-contract, adapter, and verification interfaces also support
the runnable suites listed in [Benchmark Suites](../suites/index.md).

## Why the Scope Is Broad

Researchers describe closely related work using terms such as neural receivers, AI-native PHY,
learned demapping, MIMO-OFDM, beamforming, channel estimation, interference cancellation,
localization, sensing, and semantic communication. Noema groups these areas under one umbrella
because they share experiment-identity and evidence needs:

```text
experiment contract (typed recipe + benchmark protocol)
  -> declared channel and accounting boundaries
  -> benchmark or dataset capture
  -> external training and artifact return
  -> frozen evaluation
  -> locally verifiable result and plot evidence
```

Each benchmark suite supplies its own protocol, metrics, baselines, and adapter points. The shared
Noema core supplies the execution and evidence contracts.

## What Noema Provides

- typed recipes and operation contracts for communication experiments;
- benchmark execution that produces protocol-bound result bundles;
- dataset capture from declared points in a recipe graph;
- architecture-neutral training contracts for replacement boundaries and differentiable support;
- portable trained-artifact interfaces and explicit adapters for custom runtimes;
- rate, channel-use, seed, environment, and artifact accounting;
- verification for manifests, hashes, plots, and protocol metadata.

## Relationship to Other Tools

Noema integrates with domain and runtime tools such as Sionna, PyTorch, ONNX Runtime, OpenVINO,
external codec repositories, and lab-specific adapters. Those systems provide models, numerical
backends, or simulators. Noema provides the surrounding experiment contract, comparison protocol,
and evidence trail.

For example, Sionna can provide differentiable PHY, OFDM, MIMO, channel coding, and channel models.
Noema can use supported Sionna components without presenting itself as a replacement for Sionna or
as a complete PHY simulator.

## System Boundaries

Noema does not provide:

- every RF impairment, wireless standard, antenna model, scheduler, or network simulator;
- a complete drag-and-drop industrial simulation environment;
- a universal training framework or model architecture;
- evidence that every experimental suite is ready for canonical or leaderboard use.

The Workbench configures and inspects typed recipes, while specialized backends and external training
code remain replaceable. A suite's maturity is stated in its documentation: active-development
suites are maintained and runnable, while experimental suites and protocols may still change before
becoming canonical.
