"""统一准备 assignment4 本地资源；默认复用已有数据。"""
import argparse
import gzip
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import urllib.request

from cs336_data.data_preparation import atomic_json, digest_file, get_json
if __package__:
    from .download_data import download_asset, download_offline_files
else:
    from download_data import download_asset, download_offline_files

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "local-shared-data"


def snapshot(paths):
    return {str(path.resolve()): digest_file(path) for path in paths}


def stage_current(stage, inputs, config):
    if stage.get("inputs") != inputs or stage.get("config") != config or not stage.get("outputs"):
        return False
    try:
        return snapshot(map(Path, stage["outputs"])) == stage["outputs"]
    except OSError:
        return False


def find_wet_files(data):
    paths = sorted([*data.glob("*.warc.wet.gz"), *(data / "CC").glob("*.warc.wet.gz")])
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Raw WET filenames must be unique across local-shared-data and CC")
    return paths


def ensure_wet_files(target, crawl):
    paths = find_wet_files(DATA)
    if len(paths) >= target:
        return
    url = f"https://data.commoncrawl.org/crawl-data/{crawl}/wet.paths.gz"
    with urllib.request.urlopen(url, timeout=90) as response:
        candidates = gzip.decompress(response.read()).decode().splitlines()
    random.Random(336).shuffle(candidates)
    existing = {path.name for path in paths}
    candidates = [path for path in candidates if Path(path).name not in existing]
    missing = target - len(paths)
    if missing > len(candidates):
        raise ValueError("Requested more WET files than available in this crawl")
    for path in candidates[:missing]:
        print(f"[wet] downloading {Path(path).name}", flush=True)
        download_asset("https://data.commoncrawl.org/" + path, DATA / "CC" / Path(path).name)


def wiki_additional_count(root, files, target):
    complete = sum((root / "raw" / name).is_file()
                   and (root / "raw" / name).stat().st_size == metadata["size"]
                   for name, metadata in files.items())
    return max(0, target - complete)


def wiki_outputs(root):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    url_file = Path(manifest["url_file"])
    if not url_file.is_absolute():
        url_file = ROOT / url_file
    paths = [manifest_path, url_file]
    for shard in manifest["shards"]:
        name = shard["shard"]
        paths.extend([root / "raw" / name, root / "shards" / (name + ".urls.txt.gz"),
                      root / "shards" / (name + ".sample.jsonl.gz"),
                      root / "shards" / (name + ".done.json")])
    return paths


def prepare_wiki(date, target, all_shards):
    root = DATA / "wiki"
    inventory = root / f"enwiki-{date}-dumpstatus.json"
    if not inventory.exists():
        atomic_json(inventory, get_json(f"https://dumps.wikimedia.org/enwiki/{date}/dumpstatus.json"))
    job = json.loads(inventory.read_text())["jobs"]["articlesmultistreamdump"]
    if job["status"] != "done":
        raise ValueError("Wikipedia dump is not complete; select a completed --wiki-date")
    pattern = rf"enwiki-{date}-pages-articles-multistream\d+\.xml-p\d+p\d+\.bz2"
    files = {name: value for name, value in job["files"].items() if re.fullmatch(pattern, name)}
    if not files or target > len(files):
        raise ValueError("Requested Wikipedia shard count exceeds the official inventory")
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("version") == 3 and manifest.get("complete")
                and manifest.get("dump_date") == date
                and manifest["selected_shards"] >= target
                and (not all_shards or manifest["full_dump"])):
            paths = wiki_outputs(root)
            expected = {paths[1]: manifest["url_file_sha256"]}
            for shard in manifest["shards"]:
                name = shard["shard"]
                expected[root / "shards" / (name + ".urls.txt.gz")] = shard["urls_sha256"]
                expected[root / "shards" / (name + ".sample.jsonl.gz")] = shard["sample_sha256"]
            raw_valid = all((root / "raw" / shard["shard"]).is_file()
                            and digest_file(root / "raw" / shard["shard"], "sha1") == files[shard["shard"]]["sha1"]
                            for shard in manifest["shards"])
            if (raw_valid and all(path.is_file() for path in paths)
                    and all(digest_file(path) == sha for path, sha in expected.items())):
                print(f"[wiki] reuse {manifest['selected_shards']} verified shards", flush=True)
                return
    # 不让旧的提取缓存掩盖损坏的原始分片。
    for name, metadata in files.items():
        raw = root / "raw" / name
        if not raw.is_file() or digest_file(raw, "sha1") != metadata["sha1"]:
            (root / "shards" / (name + ".done.json")).unlink(missing_ok=True)
    extra = wiki_additional_count(root, files, target)
    options = ["--all-shards"] if all_shards else ["--additional-shards", str(extra)]
    run_script("prepare_wiki_references.py", "--root", str(root), "--date", date, "--workers", "1", *options)


def run_script(name, *arguments):
    subprocess.run([sys.executable, str(ROOT / "scripts" / name), *arguments], cwd=ROOT, check=True)


def english_outputs():
    manifest_path = DATA / "english-wet-data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return [manifest_path, *(Path(item["output"]) for item in manifest["files"])]


def prepare_english(allow_existing):
    sources = find_wet_files(DATA)
    if not sources:
        raise ValueError("No raw WET files; run the assets stage first")
    manifest_path = DATA / "english-wet-data/manifest.json"
    if allow_existing and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        records = manifest["files"]
        inputs = {str(path): digest_file(path) for path in sources}
        if (manifest.get("english_threshold") == 0.7 and manifest.get("text_truncated") is False
                and manifest["model_sha256"] == digest_file(DATA / "classifiers/lid.176.bin")
                and inputs == {item["source"]: item["source_sha256"] for item in records}
                and all(item.get("verified") and Path(item["output"]).is_file()
                        and digest_file(Path(item["output"])) == item["output_sha256"] for item in records)):
            print("[english] reuse verified English WET files", flush=True)
            return
    run_script("prepare_english_wet.py", "--inputs", *(str(path) for path in sources))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", choices=["assets", "wiki", "english"],
                        default=["assets", "wiki", "english"])
    parser.add_argument("--wet-files", type=int, default=2, help="Minimum raw WET files to prepare")
    parser.add_argument("--crawl-id", default="CC-MAIN-2026-17")
    parser.add_argument("--wiki-shards", type=int, default=1, help="Minimum Wikipedia shards to prepare")
    parser.add_argument("--wiki-date", default="20260501")
    parser.add_argument("--all-wiki-shards", action="store_true")
    args = parser.parse_args()
    if args.wet_files < 1 or args.wiki_shards < 1:
        parser.error("Shard counts must be positive")
    state_path = DATA / "raw-data-manifest.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    def stage(name, input_paths, config, action, output_paths):
        inputs = snapshot([Path(__file__), *input_paths()])
        previous = state.get(name, {})
        if stage_current(previous, inputs, config):
            print(f"[{name}] verified; skip", flush=True)
            return
        print(f"[{name}] preparing", flush=True)
        action()
        state[name] = {"inputs": snapshot([Path(__file__), *input_paths()]),
                       "config": config, "outputs": snapshot(output_paths())}
        atomic_json(state_path, state)

    assets = [DATA / "tokenized_paloma_c4_100_domains_validation.bin", DATA / "CC/example.warc.gz",
              DATA / "CC/example.warc.wet.gz", *(DATA / "classifiers" / name for name in
              ("lid.176.bin", "dolma_fasttext_hatespeech_jigsaw_model.bin", "dolma_fasttext_nsfw_jigsaw_model.bin"))]
    for name in ("assets", "wiki", "english"):
        if name not in args.steps:
            continue
        if name == "assets":
            def prepare_assets():
                previous = state.get("assets", {}).get("outputs", {})
                download_offline_files(root_path=DATA, known_hashes=previous)
                for path in find_wet_files(DATA):
                    if str(path) in previous and digest_file(path) != previous[str(path)]:
                        raise ValueError(f"Raw WET changed: {path}; restore it or use a new filename")
                ensure_wet_files(args.wet_files, args.crawl_id)
            stage(name, lambda: [ROOT / "scripts/download_data.py", Path(__file__)],
                  {"wet_files": args.wet_files, "crawl_id": args.crawl_id}, prepare_assets,
                  lambda: sorted(set(assets + find_wet_files(DATA))))
        elif name == "wiki":
            stage(name, lambda: [ROOT / "scripts/prepare_wiki_references.py", ROOT / "cs336_data/data_preparation.py"],
                  {"date": args.wiki_date, "shards": args.wiki_shards, "all": args.all_wiki_shards},
                  lambda: prepare_wiki(args.wiki_date, args.wiki_shards, args.all_wiki_shards),
                  lambda: wiki_outputs(DATA / "wiki"))
        else:
            stage(name, lambda: [ROOT / "scripts/prepare_english_wet.py", ROOT / "cs336_data/wet_files.py",
                                 DATA / "classifiers/lid.176.bin", *find_wet_files(DATA)],
                  {"english_threshold": 0.7}, lambda: prepare_english("english" not in state), english_outputs)
    print(f"Ready: {state_path}", flush=True)


if __name__ == "__main__":
    main()
