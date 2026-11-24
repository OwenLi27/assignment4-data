import os
import mmh3
import unicodedata
import random
import shutil

from collections import defaultdict

SEED_BASE = 336

class Document:
    def __init__(self, doc_id: int, path: os.PathLike, signature: list[int] | None):
        self.id = doc_id
        self.path = path
        self.signature = signature


def normalize_text(text: str) -> str:
    # 小写
    text = text.lower()
    # 标准化为NFD形式
    text = unicodedata.normalize("NFD", text)
    # 去掉重音符号等字符
    text = "".join(c for c in text if not unicodedata.category(c).startswith("M"))
    # 标点和符号用空格代替
    text = "".join(c if not unicodedata.category(c).startswith("P") else " " for c in text)
    text = "".join(c if not unicodedata.category(c).startswith("S") else " " for c in text)
    # 合并空白
    text = " ".join(text.split())
    return text

def text_to_ngrams(text: str, n: int) -> set[str]:
    ngrams = set()
    words = text.split()
    if not words:
        return ngrams
    if len(words) < n:
        ngrams.add(" ".join(words))
        return ngrams
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i : i + n])
        ngrams.add(ngram)
    return ngrams

def minhash_signature(ngrams: set[str], num_hashes: int) -> list[int] | None:
    if isinstance(num_hashes, bool) or not isinstance(num_hashes, int) or num_hashes <= 0:
        raise ValueError("num_hashes must be a positive integer")
    if not ngrams:
        return None

    signature = []
    for i in range(num_hashes):
        min_hash = min(mmh3.hash(ngram, SEED_BASE + i) for ngram in ngrams)
        signature.append(min_hash)
    return signature

def to_lsh_buckets(
    documents: list[Document], num_bands: int, num_hashes: int
) -> list[dict[tuple[int, ...], list[Document]]]:
    buckets = [{} for _ in range(num_bands)]
    r = num_hashes // num_bands
    for i in range(num_bands):
        for doc in documents:
            if doc.signature is None:
                continue
            band_signature = tuple(doc.signature[i * r : (i + 1) * r])
            if band_signature not in buckets[i]:
                buckets[i][band_signature] = []
            buckets[i][band_signature].append(doc)
    return buckets


def generate_candidate_pairs_from_buckets(buckets: list[dict[tuple[int, ...], list[Document]]]) -> set[tuple[int, int]]:
    pairs = set()
    for bucket in buckets:
        for docs in bucket.values():
            # 单个文档不成对，跳过
            if len(docs) < 2:
                continue
            # docs本身已经id增序，直接用索引生成不重复的文档对
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    pairs.add((docs[i].id, docs[j].id))
    return pairs


def cal_jaccard_similarity(ngram_set1: set[str], ngram_set2: set[str]) -> float | None:
    if not ngram_set1 or not ngram_set2:
        return None
    intersection = ngram_set1 & ngram_set2
    union = ngram_set1 | ngram_set2
    return len(intersection) / len(union)


def get_cached_ngrams_set(doc: Document, cache: dict[int, set[str]], ngrams: int) -> set[str]:
    if doc.id in cache:
        return cache[doc.id]
    with open(doc.path, "r", encoding="utf-8") as f:
        ngram_set = text_to_ngrams(normalize_text(f.read()), ngrams)
    cache[doc.id] = ngram_set
    return ngram_set


# 并查集merge文档pair，返回每个文档的根节点id列表
def union_find_merge_documents(similar_pairs: set[tuple[int, int]], num_documents: int) -> list[int]:
    parent = list(range(num_documents))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    for id1, id2 in similar_pairs:
        union(id1, id2)

    for i in range(num_documents):
        find(i)

    return parent


def minhash_lsh_deduplicate(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):

    documents = []
    for index, input_file in enumerate(input_files):
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
            normalized_text = normalize_text(text)
            ngram_set = text_to_ngrams(normalized_text, ngrams)
            signature = minhash_signature(ngram_set, num_hashes)
            documents.append(
                Document(
                    doc_id=index,
                    path=input_file,
                    signature=signature,
                )
            )
    buckets = to_lsh_buckets(documents, num_bands, num_hashes)
    candidate_pairs = generate_candidate_pairs_from_buckets(buckets)
    # 缓存候选对的ngrams，避免重复打开文件阅读
    candidate_cache_ngram_dic = {}
    similar_pairs = set()
    for id1, id2 in candidate_pairs:
        doc1 = documents[id1]
        doc2 = documents[id2]
        ngram_set1 = get_cached_ngrams_set(doc1, candidate_cache_ngram_dic, ngrams)
        ngram_set2 = get_cached_ngrams_set(doc2, candidate_cache_ngram_dic, ngrams)
        similarity = cal_jaccard_similarity(ngram_set1, ngram_set2)
        if similarity is not None and similarity > jaccard_threshold:
            similar_pairs.add((id1, id2))
    # 并查集合并文档对，得到每个文档的根节点id列表
    root_ids = union_find_merge_documents(similar_pairs, len(documents))

    root_to_members = defaultdict(list)

    for doc_id, root_id in enumerate(root_ids):
        root_to_members[root_id].append(doc_id)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 每个member簇随机选取一个文档输出 固定随机种子
    rng = random.Random(42)

    for root_id, members in root_to_members.items():
        chosen_doc = rng.choice(members)
        chosen_doc_path = documents[chosen_doc].path
        # 复制原始文档和文件名 到 output_directory
        shutil.copy(chosen_doc_path, os.path.join(output_directory, os.path.basename(chosen_doc_path)))
