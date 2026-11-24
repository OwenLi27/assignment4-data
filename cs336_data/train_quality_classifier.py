"""fastText 分阶段实验，支持保存和续跑。"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import product
from pathlib import Path

import fasttext

from cs336_data.data_preparation import atomic_json, digest_file

ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / 'local-shared-data/quality'
DEFAULT_PARAMS = dict(epoch=5, lr=0.1, dim=64, wordNgrams=2, minCount=2,
                      bucket=200000, loss='softmax', thread=12, minn=0, maxn=0, seed=336)


def train_quality_classifier(training_data_path, validation_data_path, model_output_path,
                             epoch=5, **hyperparameters):
    """训练并保存模型，返回两份准确率和耗时。"""
    params = dict(DEFAULT_PARAMS, epoch=epoch, **hyperparameters)
    unknown = params.keys() - DEFAULT_PARAMS.keys()
    if unknown:
        raise ValueError(f'未知参数: {sorted(unknown)}')
    # 本机 fastText 0.9.3 需至少 12 个线程，避免部分权重未初始化。
    if params['thread'] < 12:
        raise ValueError('当前 fastText 构建请使用至少 12 个线程,避免未初始化权重')
    started = time.monotonic()
    model = fasttext.train_supervised(input=str(training_data_path), verbose=0, **params)
    train_seconds = time.monotonic() - started
    ntrain, train_accuracy, _ = model.test(str(training_data_path))
    nvalid, valid_accuracy, _ = model.test(str(validation_data_path))
    if not ntrain or not nvalid or not all(math.isfinite(x) for x in (train_accuracy, valid_accuracy)):
        raise ValueError('评估样本为空或准确率不是有限数值')
    if set(model.get_labels()) != {'__label__wiki', '__label__cc'}:
        raise ValueError('模型标签必须为 wiki 和 cc')
    output = Path(model_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.bin.tmp')
    try:
        model.save_model(str(temporary))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(train_examples=ntrain, valid_examples=nvalid,
                train_accuracy=train_accuracy, valid_accuracy=valid_accuracy,
                train_seconds=round(train_seconds, 3),
                elapsed_seconds=round(time.monotonic() - started, 3))


def inspect_dataset(path):
    """检查标签和正文，记录样本数与文件哈希。"""
    counts, digest = Counter(), hashlib.sha256()
    with path.open('rb') as stream:
        for number, line in enumerate(stream, 1):
            digest.update(line)
            fields = line.decode('utf-8').split(maxsplit=1)
            if len(fields) != 2 or fields[0] not in ('__label__wiki', '__label__cc') or not fields[1].strip():
                raise ValueError(f'数据格式错误: {path}:{number}')
            counts[fields[0]] += 1
    if set(counts) != {'__label__wiki', '__label__cc'}:
        raise ValueError(f'数据必须包含 wiki 和 cc 两类: {path}')
    return dict(path=str(path), sha256=digest.hexdigest(), examples=sum(counts.values()), labels=dict(counts))


def rank_results(records):
    """按验证准确率排序，同分按实验 ID 排序。"""
    unique = {r['experiment_id']: r for r in records if r['status'] == 'success'}
    return sorted(unique.values(), key=lambda r: (-r['valid_accuracy'], r['experiment_id']))


class ExperimentLimitReached(Exception):
    pass


class ExperimentRunner:
    """在固定数据和 fastText 版本下记录、恢复实验。"""

    def __init__(self, training_data_path, validation_data_path, output_dir, max_experiments=None,
                 retry_failed=False):
        self.train = Path(training_data_path).resolve()
        self.valid = Path(validation_data_path).resolve()
        self.output = Path(output_dir).resolve()
        if max_experiments is not None and max_experiments < 1:
            raise ValueError('max_experiments 必须为正数')
        self.limit = max_experiments
        self.retry_failed = retry_failed
        self.started = 0
        self.records = {}
        self.attempted = set()

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / '.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = dict(format_version=1, fasttext_version=version('fasttext'),
                            train=inspect_dataset(self.train), valid=inspect_dataset(self.valid))
            if manifest['train']['sha256'] == manifest['valid']['sha256']:
                raise ValueError('训练和验证数据不能是相同内容')
            manifest_path = self.output / 'experiment_manifest.json'
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                raise ValueError('数据或 fastText 版本与已有实验不一致,请使用新的 --output-dir')
            if not manifest_path.exists() and any((self.output / 'records').glob('*.json')):
                raise ValueError('已有结果但缺少数据清单,请使用新的 --output-dir')
            atomic_json(manifest_path, manifest)
            self.manifest = manifest
            for path in sorted((self.output / 'records').glob('*.json')):
                r = json.loads(path.read_text())
                model = Path(r['model_path'])
                if r['status'] == 'success' and (not model.is_file() or digest_file(model) != r['model_sha256']):
                    print(f'[重跑] 模型缺失或损坏: {r["experiment_id"]}', flush=True)
                    continue
                self.records[r['experiment_id']] = r
            self.write_summary()
            return self
        except BaseException:
            self.lock.close()
            raise

    def __exit__(self, *exc):
        self.lock.close()

    def run(self, overrides, stage):
        params = dict(DEFAULT_PARAMS, **overrides)
        experiment_id = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
        previous = self.records.get(experiment_id)
        if previous and (previous['status'] == 'success' or not self.retry_failed or experiment_id in self.attempted):
            return previous
        if self.limit is not None and self.started >= self.limit:
            raise ExperimentLimitReached
        self.started += 1
        self.attempted.add(experiment_id)
        model_path = self.output / 'models' / f'{experiment_id}.bin'
        record = dict(experiment_id=experiment_id, stage=stage, params=params,
                      model_path=str(model_path), started_at=datetime.now(timezone.utc).isoformat())
        key_params = ' '.join(f'{k}={params[k]}' for k in
                              ('epoch', 'lr', 'dim', 'wordNgrams', 'minCount', 'bucket', 'minn', 'maxn'))
        print(f'[{self.started}] {key_params}', flush=True)
        try:
            metrics = train_quality_classifier(self.train, self.valid, model_path, **params)
            if (metrics['train_examples'] != self.manifest['train']['examples'] or
                    metrics['valid_examples'] != self.manifest['valid']['examples']):
                raise ValueError('评估样本数与数据清单不一致')
            record.update(metrics, status='success', model_sha256=digest_file(model_path))
            print(f'  train={metrics["train_accuracy"]:.2%} valid={metrics["valid_accuracy"]:.2%}', flush=True)
        except Exception as error:
            record.update(status='failed', error=f'{type(error).__name__}: {error}')
            print(f'  [失败,继续下一组] {record["error"]}', flush=True)
        atomic_json(self.output / 'records' / f'{experiment_id}.json', record)
        self.records[experiment_id] = record
        self.write_summary()
        return record

    def write_summary(self):
        ranked = rank_results(self.records.values())
        failed = sorted((r for r in self.records.values() if r['status'] != 'success'),
                        key=lambda r: r['experiment_id'])
        fields = ['experiment_id', 'stage', 'status', *DEFAULT_PARAMS,
                  'train_examples', 'valid_examples', 'train_accuracy', 'valid_accuracy',
                  'train_seconds', 'elapsed_seconds', 'model_path', 'error']
        temporary = self.output / 'results.csv.tmp'
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for r in ranked + failed:
                writer.writerow({**r, **r['params']})
        os.replace(temporary, self.output / 'results.csv')
        if ranked:
            atomic_json(self.output / 'best.json', ranked[0])
        else:
            (self.output / 'best.json').unlink(missing_ok=True)


def run_batch(training_data_path, validation_data_path, output_dir, *, mode='search',
              max_experiments=None, base_params=None, epochs=(5, 10, 20), learning_rates=(0.1, 0.3, 0.5),
              retry_failed=False):
    """按验证结果分阶段搜索，默认最多 32 个配置。"""
    if mode not in ('baseline', 'epochs', 'search'):
        raise ValueError(f'未知模式: {mode}')
    if not epochs or not learning_rates or any(e < 1 for e in epochs) or any(lr <= 0 for lr in learning_rates):
        raise ValueError('训练轮数和学习率列表必须非空且为正数')
    base = dict(DEFAULT_PARAMS, **(base_params or {}))
    with ExperimentRunner(training_data_path, validation_data_path, output_dir, max_experiments, retry_failed) as runner:
        complete = True

        def stage(name, configurations):
            records = [runner.run(p, name) for p in configurations]
            ranked = rank_results(records)
            if not ranked:
                raise RuntimeError(f'{name} 阶段全部失败,无法继续选择参数')
            return ranked[:2]

        try:
            if mode == 'baseline':
                stage('baseline', [base])
            elif mode == 'epochs':
                stage('epochs', [dict(base, epoch=e) for e in epochs])
            else:
                top = stage('A_epoch_lr', [dict(base, epoch=e, lr=lr) for e, lr in product(epochs, learning_rates)])
                top = stage('B_word_features', [dict(r['params'], wordNgrams=n, minCount=c)
                            for r in top for n, c in product((1, 2, 3), (1, 2, 5))])
                top = stage('C_capacity', [dict(r['params'], dim=d, bucket=b)
                            for r in top for d, b in product((64, 128), (200000, 1000000))])
                stage('D_char_features', [dict(top[0]['params'], minn=a, maxn=b) for a, b in ((0, 0), (3, 6))])
        except ExperimentLimitReached:
            complete = False
            print('[已达到本次上限] 保留结果;重新运行可继续。', flush=True)
        ranked = rank_results(runner.records.values())
        failed = [r for r in runner.records.values() if r['status'] == 'failed']
        print(f'\n成功 {len(ranked)} 组,失败 {len(failed)} 组,本次运行 {runner.started} 组。')
        if ranked:
            best = ranked[0]
            print(f'最佳: train={best["train_accuracy"]:.2%} valid={best["valid_accuracy"]:.2%}')
        print(f'结果: {runner.output / "results.csv"}')
        return dict(complete=complete, results=ranked + failed, best=ranked[0] if ranked else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('search', 'epochs', 'baseline'), default='search',
                        help='search=四阶段搜索（默认）;epochs=仅比较轮数;baseline=单次基线')
    parser.add_argument('--train', type=Path, default=QUALITY / 'dataset/train.txt')
    parser.add_argument('--valid', type=Path, default=QUALITY / 'dataset/valid.txt')
    parser.add_argument('--output-dir', type=Path, default=QUALITY / 'experiments')
    parser.add_argument('--epochs', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--learning-rates', type=float, nargs='+', default=[0.1, 0.3, 0.5])
    parser.add_argument('--seed', type=int, default=336)
    parser.add_argument('--max-experiments', type=int, help='本次最多新运行几组,已完成组不计入')
    parser.add_argument('--retry-failed', action='store_true', help='显式重试已有失败配置;默认跳过并继续未运行配置')
    args = parser.parse_args()
    try:
        report = run_batch(args.train, args.valid, args.output_dir, mode=args.mode,
                           max_experiments=args.max_experiments, base_params={'seed': args.seed},
                           epochs=args.epochs, learning_rates=args.learning_rates, retry_failed=args.retry_failed)
    except KeyboardInterrupt:
        print('\n已中断;完成的实验已保存,重新运行同一命令即可继续。')
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'错误: {error}\n')
    return int(any(r['status'] == 'failed' for r in report['results']))

if __name__ == '__main__':
    raise SystemExit(main())

