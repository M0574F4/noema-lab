"""Deterministic base-install suite for the public source repository."""

import unittest


MODULES = (
    "test_public_repository",
    "test_acr_recipe_planner_contracts",
    "test_ai_phy_explicit_pipelines",
    "test_artifact_strict_metadata",
    "test_backend_conformance",
    "test_benchmark_matrix_selection",
    "test_benchmark_metric_provenance",
    "test_benchmark_validity",
    "test_bit_consumer_canonicalization",
    "test_capture_integrity",
    "test_capture_record_layout_propagation",
    "test_capture_seed_semantics",
    "test_catalog_identity_strictness",
    "test_collection_object_contracts",
    "test_communication_condition_semantics",
    "test_core_contract_hardening",
    "test_deepjscc_demo_assets",
    "test_delayed_csi_demo_assets",
    "test_digital_adversarial_contracts",
    "test_docs_editorial_quality",
    "test_docs_interactive_charts",
    "test_docs_theme_shell",
    "test_execution_plan_cache",
    "test_execution_plan_verification",
    "test_execution_planner",
    "test_execution_profiles",
    "test_external_adapter_identity",
    "test_external_asset_provenance",
    "test_hosted_demo_catalog",
    "test_launch_assets",
    "test_matrix",
    "test_metadata_json_object_contract",
    "test_modulation_recognition_demo_assets",
    "test_neural_receiver_suite",
    "test_noise_ops",
    "test_ofdm_allocation_metrics",
    "test_ofdm_power_accounting",
    "test_ofdm_power_ownership",
    "test_packet_contract_v4",
    "test_param_validation",
    "test_payload_length_contracts",
    "test_power_allocator",
    "test_prepare_example_output",
    "test_random_bits",
    "test_recipe_compiler",
    "test_recipe_runtime_readiness",
    "test_recipe_template_catalog",
    "test_recipe_template_identity",
    "test_recipe_variants",
    "test_remote_model_provenance",
    "test_reproducibility_fail_closed",
    "test_resource_guard",
    "test_resource_units",
    "test_runtime_artifact_strict_inputs",
    "test_secure_downloads",
    "test_semantic_accounting_kinds",
    "test_surface_consistency",
    "test_task_operation_integrity",
    "test_training_capture_numeric_inputs",
    "test_training_capture_plan",
    "test_training_contracts",
    "test_training_plans",
    "test_upstream_asset_security",
    "test_verification_task_rate_accounting",
    "test_water_filling",
)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del tests, pattern
    suite = unittest.TestSuite()
    for module in MODULES:
        suite.addTests(loader.loadTestsFromName(f"tests.{module}"))
    return suite
