from __future__ import annotations

import unittest

from noema_lab.core.research_catalog import (
    _datasets_from_list,
    _metrics_from_list,
    _tasks_from_list,
)
from noema_lab.core.suites_catalog import _benchmark_packs_from_list


class CatalogIdentityStrictnessTests(unittest.TestCase):
    def test_research_catalog_collections_reject_duplicate_ids(self):
        cases = (
            (
                _datasets_from_list,
                [
                    {"id": "same", "modality": "image"},
                    {"id": "same", "modality": "text"},
                ],
                "duplicate dataset id",
            ),
            (
                _tasks_from_list,
                [
                    {
                        "id": "same",
                        "area_id": "area",
                        "kind": "first",
                        "modality": "bits",
                    },
                    {
                        "id": "same",
                        "area_id": "area",
                        "kind": "second",
                        "modality": "bits",
                    },
                ],
                "duplicate task id",
            ),
            (
                _metrics_from_list,
                [{"id": "same"}, {"id": "same"}],
                "duplicate metric id",
            ),
        )
        for loader, payload, message in cases:
            with self.subTest(loader=loader.__name__):
                with self.assertRaisesRegex(ValueError, message):
                    loader(payload)

    def test_suite_catalog_rejects_duplicate_pack_ids(self):
        payload = [
            {"id": "same", "path": "first.yaml", "task": "first"},
            {"id": "same", "path": "second.yaml", "task": "second"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate benchmark pack id"):
            _benchmark_packs_from_list(payload, "suite.benchmark_packs")


if __name__ == "__main__":
    unittest.main()
