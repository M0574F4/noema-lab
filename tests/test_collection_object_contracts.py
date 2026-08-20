from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.core.semantic import KnowledgeBase, SemanticState, fact_set
from noema_lab.ops.external_task import _load_examples
from noema_lab.ops.foundation import KnowledgeBaseSourceOperation, _examples


class CollectionObjectContractTests(unittest.TestCase):
    def test_semantic_documents_reject_non_object_collection_entries(self):
        with self.assertRaisesRegex(
            ValueError,
            "knowledge_base facts item 1 must be an object",
        ):
            KnowledgeBase.from_dict(
                {
                    "kind": "knowledge_base",
                    "id": "fixture",
                    "facts": [
                        {
                            "subject": "a",
                            "predicate": "is",
                            "object": "b",
                        },
                        "silently dropped before",
                    ],
                }
            )

        with self.assertRaisesRegex(
            ValueError,
            "semantic.state states item 1 must be an object",
        ):
            SemanticState.from_dict(
                {
                    "kind": "semantic.state",
                    "modality": "text",
                    "states": [
                        {"id": "one", "concepts": ["one"]},
                        7,
                    ],
                }
            )

    def test_foundation_text_examples_reject_non_object_entries(self):
        with self.assertRaisesRegex(
            OperationError,
            "text.batch examples item 1 must be an object",
        ):
            _examples(
                {
                    "examples": [
                        {"id": "one", "text": "valid"},
                        ["not", "an", "object"],
                    ]
                }
            )

    def test_nested_semantic_facts_reject_non_object_entries(self):
        with self.assertRaisesRegex(
            ValueError,
            "semantic state facts item 1 must be an object",
        ):
            fact_set(
                {
                    "facts": [
                        {
                            "subject": "a",
                            "predicate": "is",
                            "object": "b",
                        },
                        "silently dropped before",
                    ]
                }
            )

    def test_inline_knowledge_facts_reject_non_object_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = OperationContext(
                recipe_name="strict_collections",
                step_id="knowledge",
                params={
                    "kb_id": "inline_json",
                    "facts_json": json.dumps(
                        [
                            {
                                "subject": "a",
                                "predicate": "is",
                                "object": "b",
                            },
                            7,
                        ]
                    ),
                },
                inputs={},
                run_dir=root,
                step_dir=root / "knowledge",
            )
            with self.assertRaisesRegex(
                OperationError,
                "facts_json item 1 must be a JSON object",
            ):
                KnowledgeBaseSourceOperation().run(context)

    def test_external_task_examples_reject_non_object_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.json"
            path.write_text(
                json.dumps(
                    {
                        "examples": [
                            {"id": "one", "label": "valid"},
                            None,
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                OperationError,
                "Task examples item 1 must be an object",
            ):
                _load_examples(path)


if __name__ == "__main__":
    unittest.main()
