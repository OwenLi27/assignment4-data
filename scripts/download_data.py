"""下载本地基础资源；完整准备入口见 prepare_raw_data.py。"""
import argparse
import gzip
import hashlib
import shutil
import urllib.request
from pathlib import Path


def download_asset(url: str, out: Path, *, expected_sha256: str | None = None) -> None:
    def matches(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == expected_sha256

    if out.is_file() and out.stat().st_size and (expected_sha256 is None or matches(out)):
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "CS336-Local-Study/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
            expected_size = response.headers.get("Content-Length")
        size = temporary.stat().st_size
        if not size or (expected_size is not None and size != int(expected_size)):
            raise OSError(f"Incomplete download: {out.name}")
        if expected_sha256 is not None and not matches(temporary):
            raise OSError(f"SHA256 mismatch: {out.name}")
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)


def download_offline_files(*, root_path: Path, known_hashes: dict | None = None) -> None:
    root_path.mkdir(parents=True, exist_ok=True)
    known_hashes = known_hashes or {}

    def needs_download(path):
        if not path.is_file() or not path.stat().st_size:
            return True
        expected = known_hashes.get(str(path.resolve()))
        if expected is None:
            return False
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() != expected

    def fetch(url, path):
        download_asset(url, path, expected_sha256=known_hashes.get(str(path.resolve())))

    paloma_out = root_path / "tokenized_paloma_c4_100_domains_validation.bin"
    if needs_download(paloma_out):
        print(f"[huggingface] downloading {paloma_out.name}", flush=True)
        fetch(
            "https://huggingface.co/datasets/brunborg/cs336-a4/resolve/main/tokenized_paloma_c4_100_domains_validation.bin",
            paloma_out,
        )

    cc = root_path / "CC"
    cc.mkdir(parents=True, exist_ok=True)
    for kind, out_name in [("warc", "example.warc.gz"), ("wet", "example.warc.wet.gz")]:
        out = cc / out_name
        if needs_download(out):
            print(f"[cc] downloading {out_name}", flush=True)
            with urllib.request.urlopen(
                f"https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-12/{kind}.paths.gz"
            ) as r:
                first_path = gzip.decompress(r.read()).decode().splitlines()[0]
            fetch(f"https://data.commoncrawl.org/{first_path}", out)

    for rel_path, url in [
        (
            "classifiers/lid.176.bin",
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        ),
        (
            "classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin",
            "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin",
        ),
        (
            "classifiers/dolma_fasttext_nsfw_jigsaw_model.bin",
            "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin",
        ),
    ]:
        out = root_path / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        if needs_download(out):
            print(f"[file] downloading {rel_path}", flush=True)
            fetch(url, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-only", action="store_true", help="兼容旧命令；本脚本仅下载基础资源")
    parser.parse_args()
    download_offline_files(root_path=Path(__file__).resolve().parents[1] / "local-shared-data")
