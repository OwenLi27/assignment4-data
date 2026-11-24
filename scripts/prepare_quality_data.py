"""准备英文正负例，保留来源和抓取进度。"""
from __future__ import annotations

import argparse
import concurrent.futures
import functools
import gzip
import hashlib
import http.client
import ipaddress
import json
import os
import random
import socket
import sqlite3
import threading
import time
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import Counter, defaultdict
from pathlib import Path

import fasttext
from warcio.archiveiterator import ArchiveIterator

from cs336_data.data_preparation import USER_AGENT, atomic_json, digest_file, domain_of, normalize_text
from cs336_data.extract_text import extract_plain_text_from_html

ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / 'local-shared-data/quality'
MAX_HTML = 5 * 1024 * 1024
MIN_WORDS = 50
MAX_CHARACTERS = 50000
LANGUAGE_THRESHOLD = 0.7
_model = None
_robots = {}
_robots_lock = threading.Lock()
_host_locks = defaultdict(threading.Lock)
_dns_mode = 'system'
_dns_cache = {}
_dns_lock = threading.Lock()


def database():
    QUALITY.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(QUALITY / 'examples.sqlite3', timeout=60)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('CREATE TABLE IF NOT EXISTS examples (label TEXT, url TEXT, digest TEXT, domain TEXT, record TEXT, PRIMARY KEY(label,url))')
    con.execute('CREATE TABLE IF NOT EXISTS attempts (url TEXT PRIMARY KEY, outcome TEXT, detail TEXT)')
    con.execute('CREATE TABLE IF NOT EXISTS completed_sources (path TEXT PRIMARY KEY, sha256 TEXT, stats TEXT)')
    return con


def make_record(payload: bytes, url: str, label: str, provenance: dict):
    text = normalize_text(extract_plain_text_from_html(payload))
    original_length = len(text)
    if len(text.split()) < MIN_WORDS:
        return None, 'too_short'
    labels, scores = _model.predict(text[:MAX_CHARACTERS])
    if labels[0] != '__label__en' or float(scores[0]) < LANGUAGE_THRESHOLD:
        return None, 'not_english'
    text = text[:MAX_CHARACTERS]
    if original_length > MAX_CHARACTERS:
        text = text.rsplit(' ', 1)[0]
    digest = hashlib.sha256(text.casefold().encode()).hexdigest()
    return {'label': label, 'url': url, 'domain': domain_of(url), 'text': text,
            'digest': digest, 'english_score': float(scores[0]),
            'original_characters': original_length, 'characters': len(text),
            'words': len(text.split()), **provenance}, 'ok'


def save_record(con, record):
    con.execute('INSERT OR IGNORE INTO examples VALUES (?,?,?,?,?)',
                (record['label'], record['url'], record['digest'], record['domain'], json.dumps(record, ensure_ascii=False)))


def negatives(args):
    con = database()
    for path in args.warc:
        path = path.resolve()
        digest = digest_file(path)
        prior = con.execute('SELECT sha256 FROM completed_sources WHERE path=?', (str(path),)).fetchone()
        if prior and prior[0] == digest:
            print(f'[cc] reuse {path.name}', flush=True)
            continue
        stats = Counter()
        with path.open('rb') as stream:
            for archive_record in ArchiveIterator(stream):
                if archive_record.rec_type != 'response':
                    continue
                stats['responses'] += 1
                headers = archive_record.http_headers
                if not headers or headers.get_statuscode() != '200' or 'html' not in (headers.get_header('Content-Type') or '').lower():
                    stats['not_200_html'] += 1
                    continue
                url = archive_record.rec_headers.get_header('WARC-Target-URI') or ''
                if not url.startswith(('http://', 'https://')):
                    stats['invalid_url'] += 1
                    continue
                payload = archive_record.content_stream().read(MAX_HTML + 1)
                if len(payload) > MAX_HTML:
                    stats['too_large'] += 1
                    continue
                try:
                    record, reason = make_record(payload, url, 'cc', {
                        'source': 'common_crawl', 'source_file': str(path), 'source_sha256': digest,
                        'warc_record_id': archive_record.rec_headers.get_header('WARC-Record-ID'),
                        'crawl_date': archive_record.rec_headers.get_header('WARC-Date'),
                    })
                    stats[reason] += 1
                    if record:
                        save_record(con, record)
                except Exception as error:
                    stats['extraction_error'] += 1
                if stats['responses'] % 2000 == 0:
                    con.commit()
                    print(f'[cc] {path.name}: {dict(stats)}', flush=True)
        con.execute('INSERT OR REPLACE INTO completed_sources VALUES (?,?,?)', (str(path), digest, json.dumps(stats)))
        con.commit()
        print(f'[cc] done {path.name}: {dict(stats)}', flush=True)
    con.close()


def dns_addresses(data):
    if data.get('Status') != 0:
        raise ValueError(f'DNS status {data.get("Status")}')
    addresses = [r['data'] for r in data.get('Answer', []) if r.get('type') in (1, 28)]
    if any(not ipaddress.ip_address(a).is_global for a in addresses):
        raise ValueError('Non-public DNS address')
    return addresses


def resolve_public(host, port):
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if not literal.is_global:
            raise ValueError('Non-public address')
        return
    if '.' not in host or host.endswith(('.local', '.localhost', '.internal')):
        raise ValueError('Non-public hostname')
    if _dns_mode == 'system':
        addresses = [a[4][0] for a in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]
    else:
        with _dns_lock:
            cached = _dns_cache.get(host)
        if cached and cached[0] > time.monotonic():
            addresses = cached[1]
        else:
            addresses = []
            for kind in ['A', 'AAAA']:
                query = urllib.parse.urlencode({'name': host.encode('idna').decode('ascii'), 'type': kind,
                                               'edns_client_subnet': '0.0.0.0/0'})
                request = urllib.request.Request('https://dns.google/resolve?' + query, headers={'User-Agent': USER_AGENT})
                with urllib.request.urlopen(request, timeout=6) as response:
                    data = json.load(response)
                addresses = dns_addresses(data)
                if addresses:
                    ttl = min([60] + [int(a.get('TTL', 0)) for a in data.get('Answer', [])])
                    with _dns_lock:
                        _dns_cache[host] = (time.monotonic() + ttl, addresses)
                    break
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError('Non-public or missing address')


def check_public_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Invalid external URL')
    if parsed.port not in (None, 80, 443):
        raise ValueError('Nonstandard port')
    resolve_public(parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80))


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def is_configured_proxy(host, port):
    # 允许已配置的本地代理；目标 URL 仍需检查公网地址。
    for value in urllib.request.getproxies().values():
        try:
            parsed = urllib.parse.urlsplit(value)
            proxy_port = parsed.port or (443 if parsed.scheme == 'https' else 80)
            if parsed.hostname == host and proxy_port == port:
                return True
        except ValueError:
            continue
    return False


class PublicHTTPConnection(http.client.HTTPConnection):
    def connect(self):
        super().connect()
        if not ipaddress.ip_address(self.sock.getpeername()[0]).is_global and not is_configured_proxy(self.host, self.port):
            self.close()
            raise ValueError('Non-public connected address')


class PublicHTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        super().connect()
        if not ipaddress.ip_address(self.sock.getpeername()[0]).is_global and not is_configured_proxy(self.host, self.port):
            self.close()
            raise ValueError('Non-public connected address')


class PublicHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(PublicHTTPConnection, req)


class PublicHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(PublicHTTPSConnection, req, context=self._context)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def open_public(url, *, timeout=12, redirects=True):
    check_public_url(url)
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname.encode('idna').decode('ascii')
    netloc = host + (f':{parsed.port}' if parsed.port else '')
    url = urllib.parse.urlunsplit((parsed.scheme, netloc,
        urllib.parse.quote(parsed.path, safe='/%:@!$&\'()*+,;=-._~'),
        urllib.parse.quote(parsed.query, safe='=&%/:?@!$\'()*+,;~-._'), ''))
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'text/html,application/xhtml+xml'})
    return urllib.request.build_opener(PublicHTTPHandler(), PublicHTTPSHandler(),
                SafeRedirect() if redirects else NoRedirect()).open(request, timeout=timeout)


def robots_allowed(url):
    parsed = urllib.parse.urlsplit(url)
    origin = f'{parsed.scheme}://{parsed.netloc}'
    with _robots_lock:
        cached = _robots.get(origin)
    if cached is None:
        try:
            with open_public(origin + '/robots.txt', timeout=8) as response:
                raw = response.read(512000).decode('utf-8', errors='replace')
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(raw.splitlines())
            cached = parser
        except urllib.error.HTTPError as error:
            cached = error.code == 404
        except Exception:
            cached = False
        with _robots_lock:
            _robots[origin] = cached
    return cached if isinstance(cached, bool) else cached.can_fetch(USER_AGENT, url)


def fetch_html(url):
    for _ in range(6):
        domain = domain_of(url)
        if domain == 'wikipedia.org':
            raise ValueError('redirect_to_wikipedia')
        with _host_locks[domain]:
            if not robots_allowed(url):
                raise ValueError('robots_disallowed_or_unavailable')
            parsed = urllib.parse.urlsplit(url)
            robot = _robots.get(f'{parsed.scheme}://{parsed.netloc}')
            delay = robot.crawl_delay(USER_AGENT) if isinstance(robot, urllib.robotparser.RobotFileParser) else None
            # 超长抓取间隔直接跳过，不缩短站点要求的等待时间。
            if delay and delay > 60:
                raise ValueError('robots_crawl_delay_exceeds_60s')
            time.sleep(max(.2, delay or 0))
            try:
                start = time.monotonic()
                with open_public(url, redirects=False) as response:
                    if 'html' not in response.headers.get('Content-Type', '').lower():
                        raise ValueError('not_html')
                    chunks, size = [], 0
                    while chunk := response.read(65536):
                        chunks.append(chunk)
                        size += len(chunk)
                        if size > MAX_HTML:
                            raise ValueError('too_large')
                        if time.monotonic() - start > 40:
                            raise ValueError('transfer_timeout')
                    return b''.join(chunks), response.geturl()
            except urllib.error.HTTPError as error:
                if error.code in (301, 302, 303, 307, 308) and error.headers.get('Location'):
                    url = urllib.parse.urljoin(url, error.headers['Location'])
                    error.close()
                else:
                    raise
    raise ValueError('too_many_redirects')


def cache_html(payload, digest):
    cache = QUALITY / 'html' / (digest + '.html.gz')
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        try:
            with gzip.open(cache, 'rb') as stream:
                if hashlib.sha256(stream.read()).hexdigest() == digest:
                    return cache
        except (OSError, EOFError):
            pass
    descriptor, temporary = tempfile.mkstemp(dir=cache.parent, suffix='.tmp')
    os.close(descriptor)
    try:
        with gzip.open(temporary, 'wb') as output:
            output.write(payload)
        os.replace(temporary, cache)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return cache


def crawl_one(candidate):
    url = candidate['url']
    try:
        payload, final_url = fetch_html(url)
        record, reason = make_record(payload, final_url, 'wiki', {
                'source': 'wikipedia_reference', 'requested_url': url,
                'wiki_title': candidate['wiki_title'], 'wiki_shard': candidate['shard'],
                'fetched_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'html_sha256': hashlib.sha256(payload).hexdigest(),
        })
        if record:
            record['html_file'] = str(cache_html(payload, record['html_sha256']))
        return record, reason
    except ValueError as error:
        return None, str(error)
    except urllib.error.HTTPError as error:
        return None, f'http_{error.code}'
    except Exception as error:
        return None, f'{type(error).__name__}: {str(error)[:160]}'


def crawl(args):
    con = database()
    candidates = {}
    for path in sorted((ROOT / 'local-shared-data/wiki/shards').glob('*.sample.jsonl.gz')):
        if not path.with_name(path.name.replace('.sample.jsonl.gz', '.done.json')).exists():
            continue
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                candidate = json.loads(line)
                candidates.setdefault(candidate['url'], candidate)
    if not candidates:
        raise ValueError('No completed Wikipedia reference samples yet')
    attempted = {r[0] for r in con.execute('SELECT url FROM attempts')}
    by_domain = Counter(dict(con.execute("SELECT domain, COUNT(*) FROM examples WHERE label='wiki' GROUP BY domain")))
    existing = sum(by_domain.values())
    if existing >= args.target and not args.candidate_file:
        print(f'[crawl] target already satisfied: {existing:,} >= {args.target:,}; no requests scheduled', flush=True)
        con.close()
        return
    # 按固定优先级取样，限制同域名占比。
    pending = []
    candidate_domains = Counter()
    for candidate in sorted(candidates.values(), key=lambda r: r['priority']):
        url = candidate['url']
        domain = domain_of(url)
        if url in attempted or by_domain[domain] >= args.domain_cap:
            continue
        if urllib.parse.urlsplit(url).path.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.mp3', '.mp4', '.svg')):
            continue
        if candidate_domains[domain] >= args.domain_cap * 3:
            continue
        candidate_domains[domain] += 1
        pending.append(candidate)
    stats = Counter()
    if args.candidate_file:
        # 固定批次逐条处理；续跑跳过已有结果。
        batch = [json.loads(line) for line in args.candidate_file.read_text().splitlines()]
        if len({r['url'] for r in batch}) != len(batch) or any(r['url'] not in candidates for r in batch):
            raise ValueError('Candidate batch must contain unique, known reference URLs')
        pending = [r for r in batch if r['url'] not in attempted]
    print(f'[crawl] {len(candidates):,} sampled URLs, {len(pending):,} pending, {existing:,} accepted', flush=True)
    iterator = iter(pending[:args.max_attempts])
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        def submit_next():
            while True:
                try:
                    candidate = next(iterator)
                except StopIteration:
                    return False
                if args.candidate_file or by_domain[domain_of(candidate['url'])] < args.domain_cap:
                    break
            futures[pool.submit(crawl_one, candidate)] = candidate
            return True
        for _ in range(args.workers):
            submit_next()
        while futures:
            finished, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                candidate = futures.pop(future)
                record, outcome = future.result()
                if record:
                    if by_domain[record['domain']] >= args.domain_cap:
                        record, outcome = None, 'domain_cap'
                    else:
                        before = con.total_changes
                        save_record(con, record)
                        if con.total_changes > before:
                            by_domain[record['domain']] += 1
                            existing += 1
                con.execute('INSERT OR REPLACE INTO attempts VALUES (?,?,?)', (candidate['url'], outcome.split(':')[0], outcome))
                stats[outcome.split(':')[0]] += 1
                con.commit()
                if sum(stats.values()) % 50 == 0:
                    print(f'[crawl] accepted={existing:,}, attempts={sum(stats.values()):,}, outcomes={dict(stats)}', flush=True)
                    atomic_json(QUALITY / 'crawl-progress.json', {'accepted': existing, 'new_attempts': sum(stats.values()), 'outcomes': stats})
                if args.candidate_file or existing < args.target:
                    submit_next()
    print(f'[crawl] finished: accepted={existing:,}, outcomes={dict(stats)}', flush=True)
    con.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='stage', required=True)
    neg = sub.add_parser('negatives')
    neg.add_argument('--warc', type=Path, nargs='+', default=[ROOT / 'local-shared-data/CC/example.warc.gz', ROOT / 'local-shared-data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz'])
    pos = sub.add_parser('crawl')
    pos.add_argument('--target', type=int, default=6500)
    pos.add_argument('--workers', type=int, default=12)
    pos.add_argument('--max-attempts', type=int, default=50000)
    pos.add_argument('--domain-cap', type=int, default=25)
    pos.add_argument('--candidate-file', type=Path, help='Frozen JSONL batch; resume skips recorded attempts')
    pos.add_argument('--dns', choices=['system', 'doh'], default='system',
                     help='doh uses bounded HTTPS DNS checks without changing OS settings')
    args = parser.parse_args()
    global _model, _dns_mode
    _dns_mode = getattr(args, 'dns', 'system')
    _model = fasttext.load_model(str(ROOT / 'local-shared-data/classifiers/lid.176.bin'))
    {'negatives': negatives, 'crawl': crawl}[args.stage](args)


if __name__ == '__main__':
    main()
