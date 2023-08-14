import re
from typing import List, Optional

import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")


## test preprocess
def text_preprocess(text: str) -> List[str]:
    from nltk import pos_tag, word_tokenize
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer

    def punctuation_filter(text: str) -> str:
        filter_text = re.sub(r"[^a-zA-Z0-9\s]", "", string=text)
        return filter_text

    def get_wordnet_pos(treebank_tag: str) -> Optional[str]:
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return None

    def lemmatize_sentence(sentence: str) -> List[str]:
        res = []
        lemmatizer = WordNetLemmatizer()
        for word, pos in pos_tag(word_tokenize(sentence)):
            wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
            res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
        return res

    text = punctuation_filter(text)
    stop_words = stopwords.words("english")
    res = lemmatize_sentence(text)
    words = [word.lower() for word in res if word not in stop_words and len(word) > 1]

    return words
