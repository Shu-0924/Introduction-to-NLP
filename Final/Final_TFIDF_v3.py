from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import numpy as np
import json
import glob

ARTICLE_FOLDER_PATH = '../articles/'
TFIDF_THRESHOLD = 0.18


def load_json(path):
    with open(path, 'r', encoding='utf-8') as jsonfile:
        content = json.load(jsonfile)
    return content


def article_cleaner(article):
    article = '|'.join(article).split('|')
    article = '_'.join(article).split('_')
    article = '*'.join(article).split('*')
    article = '['.join(article).split('[')
    article = ']'.join(article).split(']')
    article = '<'.join(article).split('<')
    article = '>'.join(article).split('>')
    article = '›'.join(article).split('›')
    article = '--'.join(article).split('--')
    result = []
    for segment in article:
        result += segment.split('. ')
    return [' '.join(segment.split()) for segment in result]


def get_claim_tfidf(claim):
    claim_tfidf_model = TfidfVectorizer(stop_words='english')
    claim_tfidf_matrix = claim_tfidf_model.fit_transform(claim).toarray()
    vocab_list = claim_tfidf_model.get_feature_names_out()
    result = {}
    for vocab, score in zip(vocab_list, claim_tfidf_matrix[0]):
        result[vocab] = score
    return result


def get_article_importance(str_list, claim_tfidf):
    claim_vocab_list = list(claim_tfidf)
    claim_vocab_pos = dict()
    tfidf_model = TfidfVectorizer(stop_words='english')
    try:
        tfidf_fit = tfidf_model.fit_transform(str_list).toarray()
    except ValueError:
        return None
    vocab_dict = tfidf_model.vocabulary_
    for vocab in claim_vocab_list:
        if vocab_dict.get(vocab) is not None:
            claim_vocab_pos[vocab] = vocab_dict[vocab]

    result = [0] * tfidf_fit.shape[0]
    for i, arr in enumerate(tfidf_fit):
        for vocab, pos in claim_vocab_pos.items():
            result[i] += arr[pos] * claim_tfidf[vocab]
    return np.array(result)


def get_evidence_vector(claim, data_id):
    claim_tfidf_dict = get_claim_tfidf(claim)
    json_list = glob.glob(ARTICLE_FOLDER_PATH + f'{data_id}_*')
    result = []
    for json_path in json_list:
        article = load_json(json_path)
        article = article_cleaner(article)
        tfidf_list = get_article_importance(article, claim_tfidf_dict)
        if tfidf_list is None:
            continue
        idx = np.argmax(tfidf_list)
        if idx < 20 or idx > tfidf_list.shape[0] - 20 or np.max(tfidf_list) < TFIDF_THRESHOLD:
            continue
        evidence = ' '.join(article[idx-3:idx+3])
        evidence = ' '.join(evidence.split())
        if len(evidence.split()) > 5:
            result.append(evidence)
    return result


evidence_data = {}
data = load_json('./test.json')
for i in tqdm(range(len(data))):
    task = data[i]
    evidence_vector = get_evidence_vector([task['metadata']['claim']], task['metadata']['id'])
    evidence_data[task['metadata']['id']] = evidence_vector

with open('./test_evidence.json', 'w', encoding='utf-8') as fout:
    json.dump(evidence_data, fout, ensure_ascii=False, indent=4)
