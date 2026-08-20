# Semantic Communication Suite

The semantic communication suite is Noema's active flagship benchmark family. It focuses on methods
that communicate meaning, task-relevant information, or recoverable content under controlled channel
and rate constraints.

## Current Scope

Implemented and documented Noema surfaces currently cover:

- image reconstruction over clean and noisy channels;
- learned image codecs and protected digital baselines;
- DeepJSCC-style symbol paths and differentiable export;
- text semantic reconstruction presets;
- task-oriented smoke paths such as VQA, retrieval, detection, and segmentation contracts;
- generative receiver workflows such as caption-to-image smoke benchmarks;
- fixed bit/symbol boundaries, transmitted-bit accounting, BER/BLER-style metrics where relevant,
  manifests, verification, and plot export.

## Research Questions

The suite is meant to answer questions such as:

- Does the method preserve content, meaning, or task success under channel pressure?
- How does performance degrade with SNR, channel uses, payload bits, or coding rate?
- Does an AI-native method beat protected digital separation at the same accounting point?
- Can an external method be returned as an adapter and evaluated under a frozen benchmark protocol?

## Typical Metrics

- PSNR, MS-SSIM, MSE, MAE, and rate for reconstruction tasks;
- explicitly labeled token-overlap lexical proxies, edit similarity, and literal exact match for text;
- explicitly named single-reference answer match for task-oriented communication (not consensus VQA accuracy);
- retrieval recall and ranking metrics;
- transmitted bits, payload bits, channel uses, BER, BLER, and outage where the channel path is active;
- runtime, memory, artifact size, and payload/inference timing metrics for system comparison.

## Boundaries

This suite does not attempt to cover every semantic-communication research direction. New tasks enter
through explicit task contracts, metrics, adapter points, and benchmark packs rather than one-off
UI-only demos.
