from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from noema_lab.core.artifacts import artifact
from noema_lab.core.downloads import download_verified_https
from noema_lab.core.operations import Operation, OperationContext, OperationResult, object_schema

JsonDict = Dict[str, Any]

KODAK_BASE_URL = "https://r0k.us/graphics/kodak/kodak/"
KODAK_FILENAMES = ["kodim%02d.png" % index for index in range(1, 25)]
DOWNLOAD_TIMEOUT_S = 60
KODAK_SHA256 = {
    "kodim01.png": "a56e27cbf5f843c048b6af1d6e090760e9c92fadba88b7dee0205918a37523bd",
    "kodim02.png": "4f4b74a79237e311d72cad958237b5f7088d8bce1c82305ebefe1a70e3022dfd",
    "kodim03.png": "e25ca1ff2f0c0cb5fdfd5f9b0a0bb21ac4c3de3c84a67f35b09a85d3306249db",
    "kodim04.png": "e3b946107c5d3441c022f678d0c3caf1e224d81b1604ba840a4f88e562de61aa",
    "kodim05.png": "10349e963c5c813d327852f82c1795fa4148d69fedffc4c589bee458e3ac3d53",
    "kodim06.png": "363510303b715d4cbc384e1ce227e466b613a09e1b71ae985882bf8e7fbd9b18",
    "kodim07.png": "b77d3f006f42414bb242222e0482e750c0fb9e5ee8d4bed2f6f11c5605fe54a4",
    "kodim08.png": "ba23983c76b4832ee0e8af0592664756841a16779acd69f792e268fb6d13d6e7",
    "kodim09.png": "6a4361c2fc194feb4edaa9f9a4a0620fb9943e460ac7fdf037fb0f6dd6607a7d",
    "kodim10.png": "9dfb70f5867c29ff9ed6313683f19b3d867849e40fbc0c4c54a4a89df341cf23",
    "kodim11.png": "7936814b58b5387fce2e4e2488b4ec830dadd95fa9520f358ddb30990b50f2b6",
    "kodim12.png": "d78c37c2f04f23761ed2367dd77e2db584ddd4c3950833fecf89f199a8126980",
    "kodim13.png": "bc34a3ce58dea09dce1704c997171602de90cb34d0c8503a988b77f473d39b08",
    "kodim14.png": "55a94550ff18f3246c4074fd32b77b0c74447c26b6ad274d564d999c0450ba6e",
    "kodim15.png": "7538cbb80cb9103606c48b806eae57d56c885c7f90b9b3be70a41160f9cbb683",
    "kodim16.png": "a89c7268ccd4718ba424a99fc4643c572cf692ca6eae887185ceb4e9f11d2e54",
    "kodim17.png": "37afcc89fbdcb76d9518e04b2fc011027e2f4cd14b3b2f83cefd721641a47c5b",
    "kodim18.png": "1a9258c365988961d87a0598725b609139c303ad48a5aad6c503c3b1a87849aa",
    "kodim19.png": "b7450b264b1b0a411390d8931b112c27905a992520fc90569dc4b920aa32bbdc",
    "kodim20.png": "3b46c71e3b92a563820ba32936be8330c586c41f938efd94be938386aae4328a",
    "kodim21.png": "ac958597c82073f6bb65129c68f72b651db5b9efd82e11547d07350214bc268b",
    "kodim22.png": "1cee58eb1f2d9c7ebb254d208a03c783ce6cf2c4d8c2cf45e235dd23b4ce1b29",
    "kodim23.png": "e3111a2fd4da24af15d6459ef9eacfe54106b38e27b4a21821b75c3f5d2d5baf",
    "kodim24.png": "1071c68372cc5a01435c2c225a5cf7d4bb803846ec08bb6b3d6721b156d7cb96",
}
KODAK_SIZE_BYTES = {
    "kodim01.png": 736501, "kodim02.png": 617995, "kodim03.png": 502888,
    "kodim04.png": 637432, "kodim05.png": 785610, "kodim06.png": 618959,
    "kodim07.png": 566322, "kodim08.png": 788470, "kodim09.png": 582899,
    "kodim10.png": 593463, "kodim11.png": 621023, "kodim12.png": 531024,
    "kodim13.png": 822712, "kodim14.png": 692201, "kodim15.png": 612582,
    "kodim16.png": 534247, "kodim17.png": 602078, "kodim18.png": 780947,
    "kodim19.png": 671476, "kodim20.png": 492462, "kodim21.png": 637051,
    "kodim22.png": 701970, "kodim23.png": 557596, "kodim24.png": 706397,
}
KODAK_DATASET_PROVENANCE: JsonDict = {
    "manifest_version": "r0k.us-png-sha256-v1",
    "source_page": "https://r0k.us/graphics/kodak/",
    "download_base_url": KODAK_BASE_URL,
    "conversion_statement": "The distributor states that its Sun Raster to PNG conversion was lossless.",
    "license": {
        "status": "unverified_third_party_statement",
        "statement": "The distributor states that it understands Kodak released the images for unrestricted usage.",
        "authoritative_license_located": False,
        "redistribution_in_noema": False,
    },
    "publication_ready": False,
    "publication_blocker": "Archive authoritative image usage terms before publication; do not infer a license from the mirror statement.",
}


class KodakFilesOperation(Operation):
    id = "source.kodak_files"
    name = "Kodak image file manifest"
    output_kinds = {"images": "image.files"}
    params_schema = object_schema(
        {
            "dataset_dir": {"type": "string", "default": ".noema/datasets/kodak"},
            "limit": {"type": "integer", "default": 24, "minimum": 1, "maximum": 24},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        dataset_dir = Path(str(ctx.params.get("dataset_dir", ".noema/datasets/kodak")))
        limit = int(ctx.params.get("limit", 24))
        download_kodak_dataset(dataset_dir)
        paths = [dataset_dir / name for name in KODAK_FILENAMES[:limit]]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise RuntimeError(
                "Kodak files are missing after fetching the dataset. Missing: %s"
                % ", ".join(missing[:3])
            )
        manifest = {
            "dataset": "kodak",
            **KODAK_DATASET_PROVENANCE,
            "files": [str(path) for path in paths],
            "file_records": [kodak_file_record(path.name, path) for path in paths],
            "count": len(paths),
            "note": "This manifest keeps image decoding out of the core CLI; add an image.files_to_numpy adapter with Pillow or OpenCV.",
        }
        manifest_path = ctx.output_path("kodak_manifest", ".json")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return OperationResult(
            outputs={"images": artifact("image.files", manifest_path, manifest)},
            metadata={"dataset": "kodak", "count": len(paths)},
        )


def download_kodak_dataset(destination: Path, limit: Optional[int] = None) -> List[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    filenames = KODAK_FILENAMES[: int(limit or len(KODAK_FILENAMES))]
    downloaded = []
    for filename in filenames:
        target = destination / filename
        if not target.exists():
            _download_file(
                KODAK_BASE_URL + filename,
                target,
                timeout_s=DOWNLOAD_TIMEOUT_S,
                expected_sha256=KODAK_SHA256[filename],
                expected_size=KODAK_SIZE_BYTES[filename],
            )
        _verify_kodak_file(target, filename)
        downloaded.append(target)
    return downloaded


def kodak_file_record(filename: str, path: Optional[Path] = None) -> JsonDict:
    if filename not in KODAK_SHA256:
        raise RuntimeError("Unknown Kodak manifest filename: %s" % filename)
    record: JsonDict = {
        "filename": filename,
        "source_url": KODAK_BASE_URL + filename,
        "sha256": KODAK_SHA256[filename],
        "size_bytes": KODAK_SIZE_BYTES[filename],
    }
    if path is not None:
        record["path"] = str(path)
    return record


def _verify_kodak_file(path: Path, filename: str) -> None:
    expected_size = KODAK_SIZE_BYTES[filename]
    actual_size = int(path.stat().st_size)
    if actual_size != expected_size:
        raise RuntimeError(
            "Kodak dataset integrity mismatch for %s: expected %d bytes, got %d; "
            "remove the file and fetch it again"
            % (path, expected_size, actual_size)
        )
    digest = _file_sha256(path)
    if digest != KODAK_SHA256[filename]:
        raise RuntimeError(
            "Kodak dataset integrity mismatch for %s: expected sha256 %s, got %s; "
            "remove the file and fetch it again"
            % (path, KODAK_SHA256[filename], digest)
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(
    url: str,
    target: Path,
    timeout_s: int,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    try:
        download_verified_https(
            url,
            target,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            max_bytes=expected_size,
            timeout_s=timeout_s,
            opener=urllib.request.urlopen,
        )
    except Exception as exc:
        raise RuntimeError("Could not download Kodak file %s within %ss: %s" % (url, timeout_s, exc)) from exc
