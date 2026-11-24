"""下载并校验选定的 Wikipedia 分片，提取引用链接。"""
from __future__ import annotations

import argparse
import bz2
import concurrent.futures
import gzip
import hashlib
import heapq
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from cs336_data.data_preparation import atomic_json, digest_file, download_file, get_json, reference_urls

VERSION = 3


def extract_shard(root: Path, name: str, metadata: dict, sample_size: int) -> dict:
    done = root / 'shards' / (name + '.done.json')
    output = root / 'shards' / (name + '.urls.txt.gz')
    samples_path = root / 'shards' / (name + '.sample.jsonl.gz')
    if done.exists() and output.exists() and samples_path.exists():
        manifest = json.loads(done.read_text())
        if (manifest.get('version') == VERSION and manifest.get('source_sha1') == metadata['sha1']
                and manifest.get('requested_sample_size') == sample_size):
            if digest_file(output) == manifest['urls_sha256'] and digest_file(samples_path) == manifest['sample_sha256']:
                print(f'[wiki] reuse {name}', flush=True)
                return manifest
    start = time.monotonic()
    raw = root / 'raw' / name
    print(f'[wiki] downloading {name} ({metadata["size"] / 1e6:.1f} MB)', flush=True)
    download_file('https://dumps.wikimedia.org' + metadata['url'], raw,
                  sha1=metadata['sha1'], size=metadata['size'])
    print(f'[wiki] extracting {name}', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    heap, selected = [], set()
    pages = count = 0
    with bz2.open(raw, 'rb') as source, gzip.open(temporary, 'wt', encoding='utf-8') as stream:
        context = ET.iterparse(source, events=('start', 'end'))
        _, xml_root = next(context)
        for event, elem in context:
            if event != 'end' or elem.tag.rsplit('}', 1)[-1] != 'page':
                continue
            ns = elem.findtext('{*}ns')
            if ns == '0':
                pages += 1
                title = elem.findtext('{*}title') or ''
                text = elem.findtext('{*}revision/{*}text') or ''
                for url in reference_urls(text):
                    stream.write(url + '\n')
                    count += 1
                    priority = int.from_bytes(hashlib.sha256(('336:' + url).encode()).digest()[:8], 'big')
                    if url in selected:
                        continue
                    entry = (-priority, url, title)
                    if len(heap) < sample_size:
                        heapq.heappush(heap, entry)
                        selected.add(url)
                    elif priority < -heap[0][0]:
                        removed = heapq.heapreplace(heap, entry)
                        selected.remove(removed[1])
                        selected.add(url)
            elem.clear()
            xml_root.clear()
    if not pages or not count:
        raise ValueError(f'Empty reference extraction from {name}: {pages} pages, {count} URLs')
    os.replace(temporary, output)
    with gzip.open(samples_path.with_suffix('.tmp'), 'wt', encoding='utf-8') as stream:
        for priority, url, title in sorted(heap, reverse=True):
            stream.write(json.dumps({'url': url, 'wiki_title': title, 'shard': name,
                                     'priority': -priority}, ensure_ascii=False) + '\n')
    os.replace(samples_path.with_suffix('.tmp'), samples_path)
    manifest = {'version': VERSION, 'shard': name, 'source_sha1': metadata['sha1'],
                'source_bytes': metadata['size'], 'article_pages': pages, 'reference_urls': count,
                'sample_size': len(heap), 'requested_sample_size': sample_size, 'urls_sha256': digest_file(output),
                'sample_sha256': digest_file(samples_path), 'elapsed_seconds': round(time.monotonic() - start, 1)}
    atomic_json(done, manifest)
    print(f'[wiki] done {name}: {pages:,} pages, {count:,} URLs', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('local-shared-data/wiki'))
    parser.add_argument('--date', default='20260501')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--sample-per-shard', type=int, default=6000)
    parser.add_argument('--existing-only', action='store_true', help='Process only already-downloaded, complete shards')
    parser.add_argument('--all-shards', action='store_true', help='Explicitly download the whole dump')
    parser.add_argument('--additional-shards', type=int, default=0, help='Download this many additional shards, smallest first')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    inventory_path = args.root / f'enwiki-{args.date}-dumpstatus.json'
    if not inventory_path.exists():
        atomic_json(inventory_path, get_json(f'https://dumps.wikimedia.org/enwiki/{args.date}/dumpstatus.json'))
    status = json.loads(inventory_path.read_text())
    job = status['jobs']['articlesmultistreamdump']
    if job['status'] != 'done':
        raise ValueError('Source dump is not complete')
    pattern = rf'enwiki-{args.date}-pages-articles-multistream\d+\.xml-p\d+p\d+\.bz2'
    files = {k: v for k, v in job['files'].items() if re.fullmatch(pattern, k)}
    if not files:
        raise ValueError('No matching shards in official inventory')
    inventory_count = len(files)
    if args.additional_shards < 0 or (args.existing_only and (args.all_shards or args.additional_shards)):
        parser.error('Choose existing-only, bounded additional shards, or explicit all-shards')
    if not args.all_shards:
        existing = {name: meta for name, meta in files.items()
                    if (args.root / 'raw' / name).exists()
                    and (args.root / 'raw' / name).stat().st_size == meta['size']}
        additional = sorted(((n, m) for n, m in files.items() if n not in existing), key=lambda item: item[1]['size'])[:args.additional_shards]
        files = {**existing, **dict(additional)}
        if not files:
            raise ValueError('No complete local raw shards. Use --additional-shards 1 to start bounded acquisition.')
    print(f'[wiki] selected inventory: {len(files)}/{inventory_count} shards, {sum(v["size"] for v in files.values()) / 1e9:.2f} GB', flush=True)
    manifests = []
    # 小分片优先。
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_shard, args.root, name, meta, args.sample_per_shard): name
                   for name, meta in sorted(files.items(), key=lambda pair: pair[1]['size'])}
        for future in concurrent.futures.as_completed(futures):
            manifests.append(future.result())
            atomic_json(args.root / 'progress.json', {'completed': len(manifests), 'total': len(files),
                                                    'reference_urls': sum(m['reference_urls'] for m in manifests)})
    suffix = '-subset' if len(files) < inventory_count else ''
    final = args.root / f'enwiki-{args.date}-extracted_urls{suffix}.txt.gz'
    with gzip.open(final.with_suffix('.tmp'), 'wt', encoding='utf-8') as out:
        for name in sorted(files):
            with gzip.open(args.root / 'shards' / (name + '.urls.txt.gz'), 'rt', encoding='utf-8') as stream:
                for line in stream:
                    out.write(line)
    os.replace(final.with_suffix('.tmp'), final)
    atomic_json(args.root / 'manifest.json', {'version': VERSION, 'dump_date': args.date,
                'complete': True, 'full_dump': len(files) == inventory_count,
                'official_inventory_shards': inventory_count, 'selected_shards': len(files),
                'shards': sorted(manifests, key=lambda m: m['shard']),
                'url_file': str(final), 'url_file_sha256': digest_file(final),
                'total_reference_urls': sum(m['reference_urls'] for m in manifests)})
    print(f'[wiki] COMPLETE: {final}', flush=True)


if __name__ == '__main__':
    main()
