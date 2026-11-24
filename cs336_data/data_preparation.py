"""本地数据准备工具，支持断点续跑。"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import tldextract

USER_AGENT = 'CS336DataPrep/1.0 (+https://github.com/OwenLi27/assignment4-data)'
DOMAIN_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)
REF = re.compile(r'<ref\b(?![^>]*?/\s*>)[^>]*>(.*?)</ref\s*>', re.I | re.S)
INERT = re.compile(r'<!--.*?-->|<(nowiki|pre|source|syntaxhighlight|code|math)\b[^>]*>.*?</\1\s*>', re.I | re.S)
URL = re.compile(r'https?://[^\s<>\[\]{}|"\x00-\x20]+', re.I)


def domain_of(url: str) -> str:
    host = (urllib.parse.urlsplit(url).hostname or '').lower().rstrip('.')
    parsed = DOMAIN_EXTRACTOR(host)
    return parsed.top_domain_under_public_suffix or host


def reference_urls(wikitext: str):
    """提取 ref 标签中的外部 HTTP(S) 链接。"""
    seen = set()
    for ref in REF.finditer(INERT.sub('', wikitext)):
        for match in URL.finditer(html.unescape(ref.group(1))):
            url = match.group().rstrip(".,;:'")
            while url.endswith(')') and url.count(')') > url.count('('):
                url = url[:-1]
            try:
                parsed = urllib.parse.urlsplit(url)
                host = (parsed.hostname or '').lower()
                if not host or parsed.username or parsed.password:
                    continue
                if host == 'wikipedia.org' or host.endswith('.wikipedia.org'):
                    continue
                url = urllib.parse.urlunsplit(parsed._replace(fragment=''))
            except ValueError:
                continue
            if url not in seen:
                seen.add(url)
                yield url


def normalize_text(text: str) -> str:
    text = re.sub(r'__label__\S*', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
    return ' '.join(text.split())


def split_for_domain(domain: str, seed: int = 336) -> str:
    number = int.from_bytes(hashlib.sha256(f'{seed}:{domain}'.encode()).digest()[:8], 'big')
    return 'valid' if number % 10 == 0 else 'train'


def digest_file(path: Path, algorithm: str = 'sha256') -> str:
    digest = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def get_json(url: str):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def download_file(url: str, path: Path, *, sha1: str, size: int, attempts: int = 4) -> Path:
    """断点续传；通过大小和校验和检查后保存。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == size and digest_file(path, 'sha1') == sha1:
        return path
    part = path.with_suffix(path.suffix + '.part')
    for attempt in range(attempts):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > size:
                part.unlink()
                offset = 0
            if offset < size:
                headers = {'User-Agent': USER_AGENT}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=90) as response:
                    if offset and response.status == 206:
                        if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                            raise ValueError('Invalid resume Content-Range')
                    elif offset:
                        offset = 0
                    with part.open('ab' if offset else 'wb') as output:
                        while block := response.read(1024 * 1024):
                            output.write(block)
            if part.stat().st_size != size:
                raise OSError(f'Incomplete download: {part.stat().st_size}/{size}')
            actual = digest_file(part, 'sha1')
            if actual != sha1:
                part.unlink()
                raise ValueError(f'SHA1 checksum mismatch: {actual} != {sha1}')
            os.replace(part, path)
            return path
        except (OSError, ValueError, urllib.error.URLError):
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError('Download did not finish')


def word_shingles(text: str) -> set[int]:
    import mmh3
    words = re.findall(r'\w+', text.casefold())
    return {mmh3.hash64(' '.join(words[i:i + 5]), signed=False)[0]
            for i in range(max(1, len(words) - 4))}


def minhash_signature(shingles: set[int]):
    import numpy as np
    rng = np.random.default_rng(336)
    a = rng.integers(1, 2**63, size=64, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    b = rng.integers(0, 2**64 - 1, size=64, dtype=np.uint64)
    values = np.fromiter(shingles, dtype=np.uint64)
    with np.errstate(over='ignore'):
        return (values[:, None] * a + b).min(axis=0)


def deduplicate_records(records: list[dict], threshold: float = 0.8, *, preserve_order=False):
    """先做精确去重，再用 MinHash 找候选、Jaccard 确认近重复。"""
    kept, shingle_sets, seen, buckets = [], [], set(), defaultdict(list)
    seen_urls = set()
    stats = Counter()
    # 同一文本同时来自两个来源时，优先保留 wiki 标签。
    ordered = records if preserve_order else sorted(records, key=lambda r: (r['label'] != 'wiki', hashlib.sha256(r['url'].encode()).hexdigest()))
    for record in ordered:
        record_urls = {record['url'], record.get('requested_url', record['url'])}
        if seen_urls & record_urls:
            stats['duplicate_urls'] += 1
            continue
        seen_urls.update(record_urls)
        canonical = ' '.join(re.findall(r'\w+', record['text'].casefold()))
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        if digest in seen:
            stats['exact_duplicates'] += 1
            continue
        shingles = word_shingles(record['text'])
        signature = minhash_signature(shingles)
        keys = [(band, signature[band * 4:(band + 1) * 4].tobytes()) for band in range(16)]
        possible = {idx for key in keys for idx in buckets[key]}
        near = False
        for idx in possible:
            other = shingle_sets[idx]
            if min(len(shingles), len(other)) / max(len(shingles), len(other)) < threshold:
                continue
            intersection = len(shingles & other)
            if intersection / (len(shingles) + len(other) - intersection) >= threshold:
                near = True
                break
        seen.add(digest)
        if near:
            stats['near_duplicates'] += 1
            continue
        record = dict(record, canonical_digest=digest)
        idx = len(kept)
        kept.append(record)
        shingle_sets.append(shingles)
        for key in keys:
            buckets[key].append(idx)
    stats['input'] = len(records)
    stats['kept'] = len(kept)
    return kept, dict(stats)


def expand_training_records(train, valid, candidates, domain_cap=25):
    """保留原数据，加入去重且与验证域名隔离的等量正负例。"""
    if Counter(r['label'] for r in train)['wiki'] != Counter(r['label'] for r in train)['cc']:
        raise ValueError('Base training set must be balanced')
    anchors = train + valid
    parents = {}
    def find(value):
        parents.setdefault(value, value)
        while value != parents[value]:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value
    for r in anchors + candidates:
        a, b = find(domain_of(r['url'])), find(domain_of(r.get('requested_url', r['url'])))
        parents[max(a, b)] = min(a, b)
    heldout = {find(domain_of(r['url'])) for r in valid}
    anchor_urls = {u for r in anchors for u in (r['url'], r.get('requested_url', r['url']))}
    eligible, stats = [], Counter()
    for r in candidates:
        if anchor_urls & {r['url'], r.get('requested_url', r['url'])}:
            stats['existing_url_excluded'] += 1
        elif find(domain_of(r['url'])) in heldout:
            stats['validation_domain_excluded'] += 1
        else:
            eligible.append(r)
    eligible.sort(key=lambda r: (r['label'] != 'wiki', hashlib.sha256(r['url'].encode()).hexdigest()))
    kept, dedup = deduplicate_records(anchors + eligible, preserve_order=True)
    if [r['url'] for r in kept[:len(anchors)]] != [r['url'] for r in anchors]:
        raise ValueError('Base dataset contains duplicates; cannot preserve its rows')
    counts = Counter((r['label'], find(domain_of(r['url']))) for r in train)
    pools = {'wiki': [], 'cc': []}
    for r in kept[len(anchors):]:
        group = find(domain_of(r['url']))
        key = (r['label'], group)
        if counts[key] >= domain_cap:
            stats['domain_cap_excluded'] += 1
            continue
        counts[key] += 1
        pools[r['label']].append(dict(r, split='train', domain_group=group))
    count = min(map(len, pools.values()))
    additions = [r for label in ('wiki', 'cc') for r in pools[label][:count]]
    additions.sort(key=lambda r: hashlib.sha256(f'shuffle:336:{r["url"]}'.encode()).hexdigest())
    return train + additions, {**stats, 'deduplication': dedup,
                              'eligible_per_class': {k: len(v) for k, v in pools.items()},
                              'added_per_class': count}


def select_splits(records: list[dict], per_class: int = 5000, domain_cap: int = 25, seed: int = 336):
    """合并原始和重定向域名，按域名分组并平衡取样。"""
    parents = {}
    def find(value):
        parents.setdefault(value, value)
        while value != parents[value]:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value
    for r in records:
        a, b = find(domain_of(r['url'])), find(domain_of(r.get('requested_url', r['url'])))
        parents[max(a, b)] = min(a, b)
    partitions = {'train': defaultdict(list), 'valid': defaultdict(list)}
    counts = Counter()
    for r in sorted(records, key=lambda r: hashlib.sha256(f'{seed}:{r["digest"]}'.encode()).hexdigest()):
        group = find(domain_of(r['url']))
        key = (r['label'], group)
        if counts[key] >= domain_cap:
            continue
        counts[key] += 1
        split = split_for_domain(group, seed)
        partitions[split][r['label']].append(dict(r, split=split, domain_group=group))
    nvalid = max(1, per_class // 10)
    result = {}
    for split, count in [('train', per_class - nvalid), ('valid', nvalid)]:
        for label in ['wiki', 'cc']:
            if len(partitions[split][label]) < count:
                raise ValueError(f'Not enough {split}/{label}: {len(partitions[split][label])} < {count}')
        chosen = [r for label in ['wiki', 'cc'] for r in partitions[split][label][:count]]
        result[split] = sorted(chosen, key=lambda r: hashlib.sha256(f'shuffle:{seed}:{r["digest"]}'.encode()).hexdigest())
    return result
