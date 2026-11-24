"""从本地 WET 文件准备英文语料，并回读验证。"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path

from warcio.archiveiterator import ArchiveIterator
from warcio.warcwriter import WARCWriter

from cs336_data.wet_files import load_english_filter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "local-shared-data"


def digest_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record_digest(record, payload):
    identity = [record.rec_type, record.rec_headers.get_header("WARC-Record-ID"),
                record.rec_headers.get_header("WARC-Target-URI")]
    return hashlib.sha256(json.dumps(identity).encode() + b"\0" + payload).digest()


def prepare_file(source, destination, is_english):
    stats = Counter(documents=0, kept_documents=0, rejected_documents=0,
                    metadata_records=0, input_text_bytes=0, kept_text_bytes=0)
    expected = hashlib.sha256()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with source.open("rb") as stream, temporary.open("wb") as output:
            writer = WARCWriter(output, gzip=True)
            for record in ArchiveIterator(stream):
                payload = record.content_stream().read()
                if record.rec_type == "conversion":
                    stats["documents"] += 1
                    stats["input_text_bytes"] += len(payload)
                    keep = is_english(payload.decode("utf-8", errors="replace"))
                    stats["kept_documents" if keep else "rejected_documents"] += 1
                    if stats["documents"] % 10000 == 0:
                        print(f"[{source.name}] {stats['documents']:,} documents, "
                              f"{stats['kept_documents']:,} kept", flush=True)
                    if not keep:
                        continue
                    stats["kept_text_bytes"] += len(payload)
                else:
                    stats["metadata_records"] += 1
                expected.update(record_digest(record, payload))
                record.raw_stream = BytesIO(payload)
                writer.write_record(record)

        actual = hashlib.sha256()
        verified = Counter()
        with temporary.open("rb") as stream:
            for record in ArchiveIterator(stream):
                payload = record.content_stream().read()
                actual.update(record_digest(record, payload))
                if record.rec_type == "conversion":
                    if not is_english(payload.decode("utf-8", errors="replace")):
                        raise ValueError("Output contains a document below the English threshold")
                    verified["documents"] += 1
                else:
                    verified["metadata_records"] += 1
        if (actual.digest() != expected.digest()
                or verified["documents"] != stats["kept_documents"]
                or verified["metadata_records"] != stats["metadata_records"]):
            raise ValueError("Output record verification failed")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    return {"source": str(source), "source_sha256": digest_file(source),
            "output": str(destination), "output_sha256": digest_file(destination),
            "output_bytes": destination.stat().st_size, **stats,
            "verified": True, "records_sha256": actual.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, default=[
        DATA / "CC/example.warc.wet.gz",
        DATA / "CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz",
    ])
    parser.add_argument("--output-dir", type=Path, default=DATA / "english-wet-data")
    args = parser.parse_args()
    sources = [path.resolve() for path in args.inputs]
    if len({path.name for path in sources}) != len(sources):
        parser.error("Input filenames must be unique")
    for source in sources:
        if not source.is_file():
            parser.error(f"Input does not exist: {source}")
    output_dir = args.output_dir.resolve()
    if any(source == output_dir / source.name for source in sources):
        parser.error("Output must be separate from input")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = DATA / "classifiers/lid.176.bin"
    is_english = load_english_filter(model_path)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path), "model_sha256": digest_file(model_path),
        "english_threshold": 0.7, "text_truncated": False,
        "filter": "top label en with score >= 0.7; empty text rejected",
        "scope": "Local WET subset; language filtering only",
        "files": [],
    }
    for source in sources:
        print(f"Processing {source.name}", flush=True)
        result = prepare_file(source, output_dir / source.name, is_english)
        manifest["files"].append(result)
        print(f"Verified {result['kept_documents']:,}/{result['documents']:,} documents", flush=True)
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
