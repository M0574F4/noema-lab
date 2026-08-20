from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import urllib.request

from noema_lab.core.recipe_templates import (
    instantiate_recipe_template,
    load_recipe_template_catalog,
)
from noema_lab.core.research import research_specs_from_recipe
from noema_lab.ops import build_registry
from noema_lab.ui.server import start_ui_server_in_thread


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_TEMPLATE_IDENTITIES = {
    "semantic_comm.image_reconstruction.default": (
        "image_source_coded_link",
        "Image reconstruction over a source-coded link using Kodak images and a pretrained CompressAI codec.",
    ),
    "semantic_comm.image_reconstruction.protected_digital": (
        "image_protected_digital_link",
        "Kodak image reconstruction over a protected digital AWGN link with JPEG, CRC32, repetition coding, and QPSK.",
    ),
    "semantic_comm.image_reconstruction.deepjscc_awgn": (
        "deepjscc_kodak_awgn_train",
        "DeepJSCC image scenario with typed trainable sender/receiver slots around the canonical power-normalized AWGN path.",
    ),
    "semantic_comm.image_reconstruction.deepjscc_slow_rayleigh": (
        "deepjscc_kodak_slow_rayleigh_train",
        "Nested-bandwidth DeepJSCC image scenario with trainable sender and receiver slots around a blind slow-Rayleigh channel: one unknown complex gain per source image, no receiver equalization, and no CSI or pilots.",
    ),
    "semantic_comm.text_semantic_similarity.default": (
        "text_layered_digital_transport",
        "Text reconstruction over a bit-perfect layered digital link with UTF-8 transport and semantic-similarity metrics.",
    ),
    "semantic_comm.text_semantic_similarity.semantic_state_kb": (
        "text_knowledge_assisted_transport",
        "Knowledge-assisted text reconstruction using a compact SemanticState payload, a receiver knowledge base, and faithfulness metrics.",
    ),
    "semantic_comm.text_semantic_similarity.bart_joint_symbols": (
        "text_continuous_symbol_jscc",
        "Text reconstruction over continuous channel symbols derived from pretrained BART semantic representations.",
    ),
    "semantic_comm.visual_question_answering.default": (
        "visual_question_answering",
        "Visual question answering on a compact downloadable image-question dataset using a pretrained Transformers receiver.",
    ),
    "semantic_comm.object_detection.default": (
        "object_detection",
        "COCO128 object detection with a pretrained Ultralytics YOLO receiver and IoU metrics.",
    ),
    "semantic_comm.segmentation.default": (
        "segmentation",
        "COCO8 instance segmentation with a pretrained Ultralytics YOLO receiver and mask metrics.",
    ),
    "semantic_comm.image_text_retrieval.default": (
        "image_text_retrieval",
        "Flickr8k image-text retrieval with pretrained CLIP embeddings and single-positive Hit@K metrics.",
    ),
    "semantic_comm.image_generation.default": (
        "caption_to_image_generation",
        "Caption-to-image semantic transmission on Flickr8k with a pretrained diffusion receiver and CLIP alignment metrics.",
    ),
    "ai_phy.pilot_channel_estimation.adapter": (
        "siso_pilot_channel_estimation",
        "Flat-fading SISO pilot observations over AWGN evaluated with selectable least-squares and linear-MMSE channel estimators.",
    ),
    "ai_phy.mimo_ofdm_channel_estimation.adapter": (
        "mimo_ofdm_mixed_tdl_channel_estimation",
        "2×2 MIMO-OFDM sparse-pilot channel estimation across mixed Sionna 3GPP TDL-A/C/E profiles with LS, fixed-prior LMMSE, and portable learned estimators.",
    ),
    "ai_phy.csi_compression_feedback.truncated_angular_delay": (
        "limited_feedback_csi",
        "Limited-feedback CSI reconstruction at a 32-real-value, 4-bit budget on correlated Sionna TDL MISO-OFDM channels.",
    ),
    "ai_phy.beamforming_precoding.adapter": (
        "single_user_miso_beamforming",
        "Single-user MISO beamforming with selectable MRT and DFT-codebook reference methods on seeded flat-fading channels.",
    ),
    "ai_phy.wireless_localization.adapter": (
        "range_based_wireless_localization",
        "Four-anchor 2D range localization with SNR-derived measurement noise and selectable trilateration methods.",
    ),
    "ai_phy.aoa_estimation.adapter": (
        "ula_angle_of_arrival_estimation",
        "Single-source narrowband ULA angle estimation with noisy array snapshots and selectable Bartlett or MUSIC methods.",
    ),
    "ai_phy.resource_allocation.equal_power": (
        "ofdm_subcarrier_resource_allocation",
        "OFDM subcarrier power allocation over seeded Sionna TDL channels with selectable equal-power, Shannon water-filling, and learned policies.",
    ),
    "ai_phy.resource_allocation.delayed_csi_finite_blocklength": (
        "ofdm_reliability_allocation_delayed_csi",
        "Reliability-aware wideband OFDM power allocation for short packets using a causal history of delayed, noisy transmitter CSI derived from the same temporally correlated Sionna TDL-C trajectory as the current propagation channel.",
    ),
    "ai_phy.neural_receiver_demapping.adapter": (
        "qpsk_iq_imbalance_receiver_calibration",
        "QPSK over AWGN with a stable receiver I/Q front-end impairment and selectable uncompensated, calibrated-oracle, and learned demappers.",
    ),
    "ai_phy.neural_receiver_demapping.phase_tracking": (
        "qpsk_pilot_phase_tracking",
        "Pilot-aided QPSK packet reception over AWGN with random carrier phase, residual frequency offset, Wiener phase noise, and selectable classical, oracle, or learned phase-tracking receivers.",
    ),
    "ai_phy.automatic_modulation_recognition.awgn": (
        "modulation_recognition_blind_carrier",
        "BPSK, QPSK, and 16-QAM recognition with unknown per-frame carrier phase, residual frequency offset, AWGN, a blind cumulant baseline, and a portable learned-classifier slot.",
    ),
}


def _post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


class RecipeTemplateIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.catalog = load_recipe_template_catalog()

    def test_all_formal_templates_keep_their_identity_for_project_and_packaged_sources(self) -> None:
        self.assertEqual(set(self.catalog.templates), set(EXPECTED_TEMPLATE_IDENTITIES))

        with tempfile.TemporaryDirectory() as empty_project:
            for template in self.catalog.ordered_templates():
                expected_name, expected_description = EXPECTED_TEMPLATE_IDENTITIES[
                    template.id
                ]
                project = instantiate_recipe_template(
                    template.id,
                    ROOT,
                    self.registry,
                )
                packaged = instantiate_recipe_template(
                    template.id,
                    Path(empty_project),
                    self.registry,
                )

                for source, expected_kind, expected_reference in (
                    (project, "project_override", template.recipe_path),
                    (
                        packaged,
                        "packaged_builtin",
                        "noema_lab:%s" % template.starter_resource,
                    ),
                ):
                    with self.subTest(template=template.id, source=expected_kind):
                        task_id = str(
                            (research_specs_from_recipe(source.recipe).get("task") or {}).get(
                                "id"
                            )
                            or ""
                        )
                        self.assertEqual(source.recipe.name, expected_name)
                        self.assertEqual(
                            source.recipe.description,
                            expected_description,
                        )
                        self.assertEqual(task_id, template.task_id)
                        self.assertEqual(
                            source.recipe.execution_profile.to_dict(),
                            template.execution_profile.to_dict(),
                        )
                        self.assertEqual(source.provenance.template_id, template.id)
                        self.assertEqual(source.provenance.source_kind, expected_kind)
                        self.assertEqual(
                            source.provenance.source_reference,
                            expected_reference,
                        )

                self.assertEqual(project.recipe.name, packaged.recipe.name)
                self.assertEqual(
                    project.recipe.description,
                    packaged.recipe.description,
                )

    def test_packaged_starters_are_byte_exact_project_recipe_copies(self) -> None:
        for template in self.catalog.ordered_templates():
            with self.subTest(template=template.id):
                project = ROOT / template.recipe_path
                packaged = ROOT / "src" / "noema_lab" / template.starter_resource
                self.assertEqual(packaged.read_bytes(), project.read_bytes())

    def test_ui_template_api_returns_the_selected_template_identity(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                Path(workspace),
                ROOT,
            )
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                catalog_payload = _get_json(base + "/api/recipe-templates")
                rows = {
                    row["id"]: row for row in catalog_payload["templates"]
                }
                self.assertEqual(set(rows), set(EXPECTED_TEMPLATE_IDENTITIES))

                for template in self.catalog.ordered_templates():
                    expected_name, expected_description = EXPECTED_TEMPLATE_IDENTITIES[
                        template.id
                    ]
                    response = _post_json(
                        base + "/api/recipe-templates/instantiate",
                        {"template_id": template.id},
                    )
                    recipe = response["recipe"]
                    provenance = response["provenance"]
                    summary = rows[template.id]
                    with self.subTest(template=template.id):
                        self.assertEqual(response["status"], "instantiated")
                        self.assertEqual(recipe["name"], expected_name)
                        self.assertEqual(recipe["description"], expected_description)
                        self.assertEqual(summary["recipe"]["name"], expected_name)
                        self.assertEqual(
                            summary["recipe"]["description"],
                            expected_description,
                        )
                        self.assertEqual(
                            summary["recipe"]["task_id"],
                            template.task_id,
                        )
                        self.assertEqual(
                            summary["recipe"]["execution_profile"],
                            template.execution_profile.to_dict(),
                        )
                        self.assertEqual(
                            recipe["execution_profile"],
                            template.execution_profile.to_dict(),
                        )
                        self.assertEqual(provenance["template_id"], template.id)
                        self.assertEqual(
                            provenance["source_reference"],
                            template.recipe_path,
                        )
                        self.assertEqual(
                            recipe["metadata"]["template_provenance"],
                            provenance,
                        )

                mimo = _post_json(
                    base + "/api/recipe-templates/instantiate",
                    {"template_id": "ai_phy.mimo_ofdm_channel_estimation.adapter"},
                )["recipe"]
                self.assertEqual(
                    mimo["name"],
                    "mimo_ofdm_mixed_tdl_channel_estimation",
                )
                self.assertNotIn("CompressAI", mimo["description"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
