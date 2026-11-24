"""导出类别平衡、域名隔离的数据集并核验来源。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from cs336_data.data_preparation import (
    atomic_json, deduplicate_records, digest_file, domain_of, normalize_text, select_splits, expand_training_records,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'local-shared-data/quality'
OUTPUT = DATA / 'dataset'


def reference_candidates():
    manifest = json.loads((ROOT / 'local-shared-data/wiki/manifest.json').read_text())
    if not manifest['complete']:
        raise ValueError('Selected Wikipedia sources are incomplete')
    candidates = {}
    for shard in manifest['shards']:
        path = ROOT / 'local-shared-data/wiki/shards' / (shard['shard'] + '.sample.jsonl.gz')
        if digest_file(path) != shard['sample_sha256']:
            raise ValueError(f'Changed reference sample: {path}')
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                record = json.loads(line)
                candidates.setdefault(record['url'], record)
    return candidates, manifest


def fixture_digests():
    result = set()
    for p in (ROOT / 'tests/fixtures').glob('*quality*.txt'):
        original = p.read_text()
        normalized = normalize_text(original)
        truncated = normalized[:50000].rsplit(' ', 1)[0] if len(normalized) > 50000 else normalized
        for text in [original, normalized, truncated]:
            result.add(hashlib.sha256(' '.join(re.findall(r'\w+', text.casefold())).encode()).hexdigest())
    return result


def smoke_matches(report, artifacts):
    if not report or report.get('artifacts') != artifacts or 'smoke_train' not in report:
        return False
    smoke = report['smoke_train']
    path = Path(smoke.get('model', ''))
    return path.is_file() and digest_file(path) == smoke.get('model_sha256')


def describe(records):
    result = {}
    for label in ['wiki', 'cc']:
        rows = [r for r in records if r['label'] == label]
        lengths = [r['words'] for r in rows]
        domains = Counter(r['domain'] for r in rows)
        result[label] = {'documents': len(rows), 'domains': len(domains),
                         'words': sum(lengths), 'word_quantiles': dict(zip(['min', 'p10', 'median', 'p90', 'max'],
                             [round(float(v), 1) for v in np.quantile(lengths, [0, .1, .5, .9, 1])])),
                         'largest_domains': domains.most_common(10)}
    return result


def export(per_class, expand_from=None):
    if expand_from and OUTPUT.resolve() == expand_from.resolve():
        raise ValueError('Expansion output must differ from the base dataset')
    candidates, wiki_manifest = reference_candidates()
    con = sqlite3.connect(DATA / 'examples.sqlite3')
    completed_sources = dict(con.execute('SELECT path,sha256 FROM completed_sources'))
    records, excluded = [], Counter()
    fixtures = fixture_digests()
    for label, raw in con.execute('SELECT label,record FROM examples'):
        r = json.loads(raw)
        if label == 'cc' and completed_sources.get(r['source_file']) != r['source_sha256']:
            raise ValueError(f'CC record belongs to incomplete or changed source: {r["source_file"]}')
        if label == 'wiki' and r['requested_url'] not in candidates:
            excluded['not_in_verified_reference_sample'] += 1
            continue
        if label == 'wiki':
            source = candidates[r['requested_url']]
            r['wiki_title'] = source['wiki_title']
            r['wiki_shard'] = source['shard']
        canonical = hashlib.sha256(' '.join(re.findall(r'\w+', r['text'].casefold())).encode()).hexdigest()
        if canonical in fixtures:
            excluded['provided_test_fixture'] += 1
            continue
        records.append(r)
    print(f'[export] deduplicating {len(records):,} examples', flush=True)
    expansion = None
    if expand_from:
        base_manifest = json.loads((expand_from / 'manifest.json').read_text())
        if not base_manifest['ready']:
            raise ValueError('Base dataset has not passed its readiness audit')
        for name, metadata in base_manifest['artifacts'].items():
            if digest_file(expand_from / name) != metadata['sha256']:
                raise ValueError(f'Base artifact changed: {name}')
        base = {}
        for split in ('train', 'valid'):
            with gzip.open(expand_from / f'{split}.jsonl.gz', 'rt') as stream:
                base[split] = [json.loads(line) for line in stream]
        train, expansion = expand_training_records(base['train'], base['valid'], records)
        if len(train) == len(base['train']):
            raise ValueError('No balanced additions remain after expansion checks')
        splits = {'train': train, 'valid': base['valid']}
        dedup = expansion['deduplication']
        expansion.update(base_dataset=str(expand_from.resolve()), base_artifacts=base_manifest['artifacts'],
                         original_training_examples=len(base['train']))
        per_class = (len(train) + len(base['valid'])) // 2
    else:
        kept, dedup = deduplicate_records(records)
        splits = select_splits(kept, per_class=per_class)
    print(f'[export] deduplication: {dedup}', flush=True)
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for split, rows in splits.items():
        fasttext_path = output / f'{split}.txt'
        provenance = output / f'{split}.jsonl.gz'
        if expand_from and split == 'valid':
            for path in (fasttext_path, provenance):
                shutil.copyfile(expand_from / path.name, path)
                artifacts[path.name] = {'sha256': digest_file(path), 'bytes': path.stat().st_size}
            continue
        with fasttext_path.with_suffix('.tmp').open('w', encoding='utf-8') as f, gzip.open(provenance.with_suffix('.tmp'), 'wt', encoding='utf-8') as meta:
            for r in rows:
                f.write(f'__label__{r["label"]} {r["text"]}\n')
                meta.write(json.dumps(r, ensure_ascii=False) + '\n')
        os.replace(fasttext_path.with_suffix('.tmp'), fasttext_path)
        os.replace(provenance.with_suffix('.tmp'), provenance)
        for path in [fasttext_path, provenance]:
            artifacts[path.name] = {'sha256': digest_file(path), 'bytes': path.stat().st_size}
    sources = [dict(path=p, sha256=s, statistics=json.loads(st))
               for p, s, st in con.execute('SELECT path,sha256,stats FROM completed_sources')]
    failures = dict(con.execute('SELECT outcome,COUNT(*) FROM attempts GROUP BY outcome'))
    manifest = {'format_version': 1, 'ready': False, 'per_class': per_class, 'seed': 336,
                'label_semantics': {'wiki': 'English external pages cited in Wikipedia (proxy positive)',
                                    'cc': 'English pages from Common Crawl WARC (proxy negative)'},
                'configuration': {'extractor': 'cs336_data.extract_text.extract_plain_text_from_html',
                    'extractor_sha256': digest_file(ROOT / 'cs336_data/extract_text.py'),
                    'language_model_sha256': digest_file(ROOT / 'local-shared-data/classifiers/lid.176.bin'),
                    'english_threshold': .7, 'minimum_words': 50, 'maximum_characters': 50000,
                    'max_documents_per_label_domain_group': 25,
                    'split': 'registrable-domain groups, original/redirect domains joined; about 10% holdout',
                    'deduplication': 'canonical exact + 5-word shingles, 64-permutation MinHash, 16 bands, exact Jaccard >= 0.8'},
                'wikipedia': {k: v for k, v in wiki_manifest.items() if k != 'shards'},
                'cc_sources': sources, 'excluded': dict(excluded), 'deduplication': dedup,
                'crawl_outcomes': failures, 'statistics': {split: describe(rows) for split, rows in splits.items()},
                'artifacts': artifacts,
                'limitations': ['Source labels are noisy proxies, not factual correctness labels.',
                    'Wikipedia subset and reachable HTML pages introduce source/availability bias.',
                    'CC negatives come from two existing WARC shards, not the entire crawl.',
                    'MinHash is approximate; it does not prove absence of every possible near duplicate.',
                    'Texts are truncated to at most 50,000 characters.']}
    manifest['split_counts_per_class'] = {split: len(rows) // 2 for split, rows in splits.items()}
    if expansion:
        manifest['expansion'] = expansion
        manifest['configuration']['split'] = 'Frozen validation; preserve original training; balanced additions with joined-domain exclusion'
    atomic_json(output / 'manifest.json', manifest)
    con.close()
    print(f'[export] wrote {sum(map(len, splits.values())):,} examples to {output}', flush=True)


def audit(smoke_train, readiness_model=None):
    output = OUTPUT
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest['ready'] = False
    atomic_json(output / 'manifest.json', manifest)
    previous = json.loads((output / 'audit.json').read_text()) if (output / 'audit.json').exists() else None
    candidates, wiki_manifest = reference_candidates()
    fixtures = fixture_digests()
    all_rows, domains, hashes, urls = [], {}, {}, {}
    checks = Counter()
    source_hashes = {s['path']: s['sha256'] for s in manifest['cc_sources']}
    import fasttext
    language_model = fasttext.load_model(str(ROOT / 'local-shared-data/classifiers/lid.176.bin'))
    assert digest_file(ROOT / 'cs336_data/extract_text.py') == manifest['configuration']['extractor_sha256']
    assert digest_file(ROOT / 'local-shared-data/classifiers/lid.176.bin') == manifest['configuration']['language_model_sha256']
    for name, metadata in manifest['artifacts'].items():
        path = output / name
        assert path.stat().st_size == metadata['bytes'], f'Size changed: {name}'
        assert digest_file(path) == metadata['sha256'], f'Hash changed: {name}'
    for split in ['train', 'valid']:
        with gzip.open(output / f'{split}.jsonl.gz', 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        with (output / f'{split}.txt').open() as stream:
            lines = stream.read().splitlines()
        assert len(rows) == len(lines), 'FastText/provenance mismatch'
        nvalid = max(1, manifest['per_class'] // 10)
        expected = nvalid if split == 'valid' else manifest['per_class'] - nvalid
        expected = manifest.get('split_counts_per_class', {}).get(split, expected)
        assert Counter(r['label'] for r in rows) == {'wiki': expected, 'cc': expected}
        for line, r in zip(lines, rows, strict=True):
            assert line == f'__label__{r["label"]} {r["text"]}'
            assert normalize_text(r['text']) == r['text']
            assert len(r['text'].split()) >= 50 and len(r['text']) <= 50000
            assert r['english_score'] >= .7
            language, probability = language_model.predict(r['text'])
            assert language[0] == '__label__en' and float(probability[0]) >= .7, 'Language recheck failed'
            checks['language_rechecked'] += 1
            assert hashlib.sha256(r['text'].casefold().encode()).hexdigest() == r['digest']
            canonical = hashlib.sha256(' '.join(re.findall(r'\w+', r['text'].casefold())).encode()).hexdigest()
            assert canonical == r['canonical_digest'] and canonical not in fixtures
            assert canonical not in hashes, 'Duplicate text'
            hashes[canonical] = split
            assert r['url'] not in urls, 'Duplicate URL'
            urls[r['url']] = split
            for url in [r['url'], r.get('requested_url', r['url'])]:
                domain = domain_of(url)
                assert domain not in domains or domains[domain] == split, 'Domain leakage'
                domains[domain] = split
            if r['label'] == 'wiki':
                assert r['requested_url'] in candidates
                with gzip.open(r['html_file'], 'rb') as stream:
                    assert hashlib.sha256(stream.read()).hexdigest() == r['html_sha256']
                checks['positive_raw_html_verified'] += 1
            else:
                assert source_hashes.get(r['source_file']) == r['source_sha256'], 'Unverified CC source'
            all_rows.append(r)
        checks[f'{split}_examples'] = len(rows)
    if 'expansion' in manifest:
        expansion = manifest['expansion']
        for name in ('valid.txt', 'valid.jsonl.gz'):
            assert manifest['artifacts'][name] == expansion['base_artifacts'][name], 'Validation changed'
        with gzip.open(Path(expansion['base_dataset']) / 'train.jsonl.gz', 'rt') as stream:
            base_train = [json.loads(line) for line in stream]
        assert digest_file(Path(expansion['base_dataset']) / 'train.jsonl.gz') == expansion['base_artifacts']['train.jsonl.gz']['sha256']
        assert all_rows[:len(base_train)] == base_train, 'Original training rows changed'
        checks['frozen_validation_artifacts'] = 2
        checks['preserved_training_examples'] = len(base_train)
    for source in manifest['cc_sources']:
        assert digest_file(Path(source['path'])) == source['sha256']
        checks['cc_source_hashes_verified'] += 1
    for shard in wiki_manifest['shards']:
        raw = ROOT / 'local-shared-data/wiki/raw' / shard['shard']
        assert raw.stat().st_size == shard['source_bytes']
        assert digest_file(raw, 'sha1') == shard['source_sha1']
        shard_urls = ROOT / 'local-shared-data/wiki/shards' / (shard['shard'] + '.urls.txt.gz')
        assert digest_file(shard_urls) == shard['urls_sha256']
        checks['wiki_shards_verified'] += 1
    aggregate = Path(wiki_manifest['url_file'])
    if not aggregate.is_absolute():
        aggregate = ROOT / aggregate
    assert digest_file(aggregate) == wiki_manifest['url_file_sha256']
    # 对最终数据重新检查近重复。
    _, duplicate_check = deduplicate_records(all_rows)
    assert duplicate_check.get('near_duplicates', 0) == 0 and duplicate_check.get('exact_duplicates', 0) == 0
    report = {'artifacts': manifest['artifacts'], 'checks': dict(checks), 'unique_domains': len(domains), 'domain_overlap': 0,
              'exact_duplicate_count': 0, 'detected_near_duplicate_count': 0, 'provided_fixture_overlap': 0}
    if smoke_train or readiness_model:
        import fasttext
        # 本机 fastText 0.9.3 需至少 12 个线程，避免部分权重未初始化。
        if readiness_model:
            smoke_path = readiness_model.resolve()
            print('[audit] evaluating an existing model; no additional training', flush=True)
        else:
            print('[audit] training a one-epoch fastText readiness model', flush=True)
            model = fasttext.train_supervised(input=str(output / 'train.txt'), epoch=1, dim=32,
                        wordNgrams=2, minCount=2, bucket=100000, thread=12, verbose=0)
            smoke_path = output / 'readiness-smoke.bin'
            model.save_model(str(smoke_path))
        loaded = fasttext.load_model(str(smoke_path))
        n, precision, recall = loaded.test(str(output / 'valid.txt'))
        assert n == checks['valid_examples']
        assert set(loaded.get_labels()) == {'__label__wiki', '__label__cc'}
        report['smoke_train'] = {'training_performed': not bool(readiness_model),
                                'examples_evaluated': n, 'P@1': precision, 'R@1': recall,
                                'model': str(smoke_path), 'model_sha256': digest_file(smoke_path),
                                'note': 'Existing model inference check.' if readiness_model else 'One-epoch readiness check only.'}
    elif smoke_matches(previous, manifest['artifacts']):
        report['smoke_train'] = previous['smoke_train']
    atomic_json(output / 'audit.json', report)
    manifest['ready'] = smoke_matches(report, manifest['artifacts'])
    atomic_json(output / 'manifest.json', manifest)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['export', 'audit'])
    parser.add_argument('--per-class', type=int, default=5000)
    readiness = parser.add_mutually_exclusive_group()
    readiness.add_argument('--smoke-train', action='store_true')
    readiness.add_argument('--readiness-model', type=Path, help='Evaluate an existing model instead of training an extra smoke model')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--expand-from', type=Path, help='Preserve this dataset and add balanced training examples only')
    args = parser.parse_args()
    OUTPUT = args.output_dir.resolve()
    if args.stage == 'export':
        export(args.per_class, args.expand_from)
    else:
        audit(args.smoke_train, args.readiness_model)
