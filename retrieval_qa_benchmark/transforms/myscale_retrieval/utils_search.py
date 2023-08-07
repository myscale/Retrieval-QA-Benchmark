import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import re
import torch.nn.functional as F

## test preprocess
def text_preprocess(text):
    from nltk.corpus import wordnet, stopwords
    from nltk import word_tokenize, pos_tag
    from nltk.stem import WordNetLemmatizer
    
    def punctuation_filter(text):
        filter_text = re.sub(r'[^a-zA-Z0-9\s]','',string= text)
        return filter_text

    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def lemmatize_sentence(sentence):
        res = []
        lemmatizer = WordNetLemmatizer()
        for word, pos in pos_tag(word_tokenize(sentence)):
            wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
            res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
        return res
    
    text = punctuation_filter(text)
    stop_words = stopwords.words('english')
    res = lemmatize_sentence(text)
    words = [word.lower() for word in res if word not in stop_words and len(word)>1]
    
    return words


## get titles & paragraphs
def para_id_to_entry(para_id, dataset, start_para_list):
    para_id = int(para_id)
    if start_para_list is None:
        title = dataset[para_id]['title']
        para = dataset[para_id]['text']
    else:
        import bisect
        title_id = bisect.bisect(start_para_list, para_id)
        title = dataset[title_id-1]['title']
        para = [para for para in dataset[title_id-1]['text'].split('\n\n') if len(re.split(' |\n',para)) > 5][para_id-start_para_list[title_id-1]]
    return title, para

def para_id_list_to_entry(para_id_list, dataset, dataset_name):
    if dataset_name[0] == 'wikipedia':
        start_para_list = np.load('/mnt/workspaces/yongqij/evals/data/para/start_para.npy')
    else:
        start_para_list = None
    entry_list = []
    if type(para_id_list[0]) == list or type(para_id_list[0]) == np.ndarray:
        for paras_id in para_id_list:
            entries = []
            for i in range(len(paras_id)):
                para_id = paras_id[i]
                title, para = para_id_to_entry(para_id, dataset, start_para_list)
                entries.append((i, para_id, title, para))
            entry_list.append(entries)
    else:
        for i in range(len(para_id_list)):
            para_id = para_id_list[i]
            title, para = para_id_to_entry(para_id, dataset, start_para_list)
            entry_list.append((i, para_id, title, para))
    return entry_list


## embedding filter
def index_search(query_list, index, index_path, num_filtered):
    import faiss
    import os
    if os.path.split(index_path)[-1] == "IVFSQ_IP.index":
        faiss.normalize_L2(query_list)
    #     print('IP index: normalize query')
    # else:
    #     print('L2 index: no normalize query')
    index.nprobe = 128
    D_list, para_id_list = index.search(query_list, num_filtered) 
    return D_list, para_id_list


## hybrid search
def bm25(keywords, words_para_list):
    from rank_bm25 import BM25Okapi
    bm25_para = BM25Okapi(words_para_list)
    scores = bm25_para.get_scores(keywords)
    rank = pd.DataFrame(scores).rank(ascending = False, method = 'average').values.reshape(-1)
    return rank, scores

def SimMax(query, embeddings):
    # (q_seq_sz, dim) * (b_sz, d_seq_sz, dim).T = (b_sz, q_seq_sz, d_seq_sz)
    mat = torch.matmul(query.unsqueeze(0), embeddings.permute(0, 2, 1))
    score = mat.amax(2).sum(1).data.cpu()
    return score

def Colbert_single(args, is_filter=True):
    with torch.no_grad():
        i, sentences, query, tokenizer, model, worker_id = args
        query = F.normalize(query, p=2, dim=1)
        if is_filter:
            import string
            puncts = string.punctuation
            punct_tokens = set()
            for punct in puncts:
                punct_tokens.update(tokenizer(punct)['input_ids'][1:-1])
        scores = np.zeros(len(sentences))
        for j in range(len(sentences)):
            sentence = sentences[j]
            tokenizes = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)['input_ids'].to(f'cuda:{worker_id}')
            tokenizes[0][1] = 2
            embeddings = model(tokenizes)['pooler_output']
            embeddings = F.normalize(embeddings, p=2, dim=2)
            if is_filter:
                tokenizes = tokenizes[0].cpu().numpy()
                for idx in range(len(tokenizes)):
                    if tokenizes[idx] in punct_tokens:
                        embeddings[0][idx] = 0
            scores[j] = SimMax(query, embeddings)
            break
        return i, scores
    
def rank_colbert(question, entries, colbert_tokenizer, colbert_model, work_id=0, batch_size=1):
    question = '# ' + question
    sentences = [f"{entry[2]}\n{entry[3]}" for entry in entries]
    sentences = ['# ' + sentence for sentence in sentences]
    scores = np.zeros(len(sentences))
    q_token_ids = colbert_tokenizer(question, return_tensors='pt')['input_ids'].to(f'cuda:{0}')
    q_token_ids[0][1] = 1
    query = colbert_model(q_token_ids)['pooler_output'][0]
    for i, score in map(Colbert_single, map(lambda i: (i, sentences[i:min(i + batch_size, len(sentences))], query, colbert_tokenizer, colbert_model, work_id), range(0, len(sentences), batch_size))):
        scores[i:min(i + batch_size, len(sentences))] = score
    q_token_ids = q_token_ids[0].cpu().numpy()
    rank = pd.DataFrame(scores).rank(ascending = False, method = 'average').values.reshape(-1)
    return rank, scores

def RRF(rank_list, k_list):
    score_rrf = None
    for rank, k in zip(rank_list, k_list):
        if score_rrf is None:
            score_rrf = 1 / (k + rank)
        else:
            score_rrf += 1 / (k + rank)
    return score_rrf

def rank_result(question, entries, with_title, rank_dict, colbert_tokenizer=None, colbert_model=None):
    rank_emb = np.array(range(1,len(entries) + 1), dtype = np.int32)
    para_id = []
    title = []
    para = []
    for entry in entries:
        para_id.append(entry[1])
        title.append(entry[2])
        para.append(entry[3])
    keywords = text_preprocess(question)
    if not with_title:
        words_para_list = [text_preprocess(_para) for _para in para]
    else:
        words_para_list = [text_preprocess(_title) + text_preprocess(_para) for _title, _para in zip(title, para)]
    db_names = ['para_id', 'rank_emb']
    for rank_name in rank_dict.keys():
        if rank_name == 'mpnet':
            db_names.append('rank_emb')
        elif rank_name == 'bm25':
            rank_bm25, score_bm25 = bm25(keywords, words_para_list)
            db_names.extend(['rank_bm25', 'score_bm25'])
        elif rank_name == 'colbert':
            rank_col, score_col = rank_colbert(question, entries, colbert_tokenizer, colbert_model)
            db_names.extend(['rank_col', 'score_col'])
        else:
            raise ValueError(f'rank_name {rank_name} is not supported')
    db_names.extend(['title', 'para'])
    result_db = pd.DataFrame()
    for name in db_names:
        result_db[name] = eval(name)
    return result_db

def rrf_result(result_db, rank_dict):
    _dict = {'mpnet':'rank_emb', 'bm25':'rank_bm25', 'colbert':'rank_col'}
    ranks = []
    rrf_coefficients = []
    for rank_name in rank_dict.keys():
        ranks.append(result_db[_dict[rank_name]].values)
        rrf_coefficients.append(rank_dict[rank_name])
    score_rrf = RRF(ranks, rrf_coefficients)
    rank_rrf = pd.DataFrame(score_rrf).rank(ascending = False, method = 'average').values.reshape(-1)
    rrf_db = result_db
    rrf_db['rank_rrf'] = rank_rrf
    rrf_db['score_rrf'] = score_rrf
    return rrf_db

def rrf_hybrid_search(question_list, entry_list, num_selected, with_title, rank_dict, colbert_tokenizer, colbert_model, show_progress):
    _entry_list = []
    for i in tqdm(range(len(question_list)), disable = not show_progress):
        result_db = rank_result(question_list[i], entry_list[i], with_title, rank_dict, colbert_tokenizer, colbert_model)
        result_db = rrf_result(result_db, rank_dict)
        paras_id = result_db.sort_values(by = 'rank_rrf')['para_id'].head(num_selected).values
        titles = result_db.sort_values(by = 'rank_rrf')['title'].head(num_selected).values
        paras = result_db.sort_values(by = 'rank_rrf')['para'].head(num_selected).values
        entries = []
        for j in range(len(paras_id)):
            entries.append((j, paras_id[j], titles[j], paras[j]))
        _entry_list.append(entries)
    return _entry_list