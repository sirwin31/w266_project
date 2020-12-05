import copy
import itertools
import random
import pathlib
import pickle
from collections import defaultdict
import json

import sklearn.model_selection
import tqdm
import torch
import torch.utils.data

def load(datasets=["gossipcop", "politifact"],
         test=False,
         max_records=None,
         path=r"../data/FakeNewsNet/filtered_dataset/"):
    """Loads data from text files.
    
    Args:
        datasets: List of strings that indicates which datasets to load.
            Optional. Default is ["gossipcop", "politifact"]. Other options
            are ["politifact"] or ["gossipcop"]
        test: Boolean, optional. If False (default), training datasets are
            loaded. Othewise test datasets are loaded.
        path: String. Relative path from jupyter notebook to location of
            folder containing datasets. Optional. Default is
            r"../data/FakeNewsNet/filtered_dataset/"
    
    Returns: Tuple containing two lists. First list contains articles as
        strings. Second list contains labels, with 1 indicating article is
        fake and 0 indicating real.
    """
    # Select either test or training datasets
    sfx = "_test" if test else "_train"
    datasets = [dataset + sfx for dataset in datasets]
    
    # Lists to hold function output
    texts = []
    labels = []
    
    # Loop through all folders
    data_dir = pathlib.Path(path)
    for dataset in datasets:
        for label in ["real", "fake"]:
            for text_file  in tqdm.tqdm((data_dir/dataset/label).iterdir(),
                                       desc=dataset + ":" + label):
                if text_file.is_dir():
                    continue
                text = text_file.read_text()
                texts.append(text)
                labels.append(0 if label == "real" else 1)

    if max_records is None:
        return texts, labels
    else:
        zipped_data = list(zip(texts, labels))
        random.shuffle(zipped_data)
        texts = [x[0] for x in zipped_data]
        labels = [x[1] for x in zipped_data]
        return texts[:max_records], labels[:max_records]

    
def load2(datasets=["gossipcop", "politifact"],
         test=False,
         max_records=None,
         path=r"../data/FakeNewsNet/filtered_dataset/",
         min_word_count = 0,
         max_word_count = float("inf")):
    """Loads data from text files of specified size.
    
    Args:
        datasets: List of strings that indicates which datasets to load.
            Optional. Default is ["gossipcop", "politifact"]. Other options
            are ["politifact"] or ["gossipcop"]
        test: Boolean, optional. If False (default), training datasets are
            loaded. Othewise test datasets are loaded.
        path: String. Relative path from jupyter notebook to location of
            folder containing datasets. Optional. Default is
            r"../data/FakeNewsNet/filtered_dataset/"
        min_word_count: Integer.  Only retrieve documents that are equal to
            or larger than this word count.
        max_word_count: Integer.  Only retrieve documents that smaller than
            this word count.
    
    Returns: Tuple containing two lists. First list contains articles as
        strings. Second list contains labels, with 1 indicating article is
        fake and 0 indicating real.
    """
    # Select either test or training datasets
    sfx = "_test" if test else "_train"
    datasets = [dataset + sfx for dataset in datasets]
    
    # Lists to hold function output
    texts = []
    labels = []
    sources = []
    
    # Loop through all folders
    data_dir = pathlib.Path(path)
    for dataset in datasets:
        for label in ["real", "fake"]:
            for text_file  in tqdm.tqdm((data_dir/dataset/label).iterdir(),
                                       desc=dataset + ":" + label):
                if text_file.is_dir():
                    continue
                    
                text = text_file.read_text()
                word_count = len(text.split())
                
                if word_count >= min_word_count and word_count < max_word_count:
                    texts.append(text)
                    labels.append(0 if label == "real" else 1)
                    sources.append(dataset)

    if max_records is None:
        return texts, labels, sources
    else:
        zipped_data = list(zip(texts, labels, sources))
        random.shuffle(zipped_data)
        texts = [x[0] for x in zipped_data]
        labels = [x[1] for x in zipped_data]
        sources = [x[2] for x in zipped_data]
            
        return texts[:max_records], labels[:max_records], sources[:max_records]


def get_size(path=r"../data/FakeNewsNet/filtered_dataset/"):
    data_dir = pathlib.Path(path)
    for dataset in data_dir.iterdir():
        if dataset.name in ["removed_files", "removed_files.pickle"]:
            continue
        for label in dataset.iterdir():
            if label.is_dir():
                print(f"{dataset.name}_{label.name}:", len(list(label.iterdir())))
            
            
class PfDataset(torch.utils.data.Dataset):
    """Creates a dataset from Longformer encodings."""
    def __init__(self, encodings, labels, sources=None):
        self.encodings = encodings
        self.labels = labels
        self.sources = sources
        
    def __getitem__(self, idx):
        item = {key: val[idx]
                for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        if self.sources is not None:
            item["sources"] = self.sources[idx]
        return item
    
    def __len__(self):
        return len(self.labels)

    
def load_dataset_from_path(path):
    """Loads dataset from encodings file located at `path`."""
    with open(path, "rb") as pfile:
        encodings, labels = pickle.load(pfile)
    return PfDataset(encodings, labels)


def load_dataset_from_path2(path):
    """Loads dataset from encodings file located at `path`."""
    with open(path, "rb") as pfile:
        encodings, labels, sources = pickle.load(pfile)
    return PfDataset(encodings, labels, sources)
        
    
def get_word_counts(datasets=["gossipcop", "politifact"],
         test=False,
         path=r"../data/FakeNewsNet/filtered_dataset/"):
    """Extracts word count per document from text files.
    
    Args:
        datasets: List of strings that indicates which datasets to load.
            Optional. Default is ["gossipcop", "politifact"]. Other options
            are ["politifact"] or ["gossipcop"]
        test: Boolean, optional. If False (default), training datasets are
            loaded. Othewise test datasets are loaded.
        path: String. Relative path from jupyter notebook to location of
            folder containing datasets. Optional. Default is
            r"../data/FakeNewsNet/filtered_dataset/"
    
    Returns: Tuple containing four lists. First list contains labels as 
        strings.  Second list contains the dataset that that document came
        from.  Third list contains the name of the file foacr reference
        purposes.  Fourth list contains the total word count of that document.
    """
    # Select either test or training datasets
    sfx = "_test" if test else "_train"
    datasets = [dataset + sfx for dataset in datasets]
    
    # Lists to hold function output
    word_counts = []
    dataset_names = []
    file_names = []
    labels = []
    
    # Loop through all folders
    data_dir = pathlib.Path(path)
    for dataset in datasets:
        for label in ["real", "fake"]:
            for text_file  in tqdm.tqdm((data_dir/dataset/label).iterdir(),
                                       desc=dataset + ":" + label):
                if text_file.is_dir():
                    continue
                
                word_count = len(text_file.read_text().split())
                
                labels.append(label)
                dataset_names.append(dataset)
                file_names.append(text_file)
                word_counts.append(word_count)
                
    return labels, dataset_names, file_names, word_counts        
        
def compile_word_frequency_corpus(data_path, datasets, min_char=0, max_char = float("inf"), test=False, use_doc_freq = False):
    """Loads article data from files stored ini `data_path`.
    
    The folder structure must be:
    data_path\
        politifact\
            fake\
            real\
        gossipcop\
            fake\
            real\
            
    Args:
        data_path: str, relative path to folders with articles.
        datasets: list, either with both["politifact", "gossipcop"], 
            or just ["politifact"] or ["gossipcop"]
        min_char: int, optional. If set to an integer, function ignores
            files with fewer than min_char characters.
        test: bool, optional. If True, retrieves test data.
        metric: bool.  If False, returns term frequency (a word appearing
            multiple times in a document is counted every time).  
            If True, returns document frequency (a word appearing multiple
            times in a document is only counted once per document).
    Returns:
        Python dictionary object where each key contains a Python list.
        The keys are: "texts", "labels", "article_chars", "article_words"
        "article_tokens", "title_chars", "title_tokens", "sources",
        "file_names", "titles", "stripped_titles"
        
        If val_split parameter is specified, passes a tuple of two dictionaries,
        with the first dictionary being the training data and the second dictionary
        as the validation data.
    """
    stripped_titles = []
    real_articles_word_dict = defaultdict(int)
    fake_articles_word_dict = defaultdict(int)

    data_dir = pathlib.Path(data_path)
    for source in datasets:
        suffix = "_test" if test else "_train"
        for label in ["real", "fake"]:
            folder_path = data_dir / (source + suffix) / label
            for jfile in tqdm.tqdm(folder_path.iterdir(), desc=f"{source}:{label}"):
                
                if jfile.is_dir():
                    continue
                    
                data = json.loads(jfile.read_text())
                stripped_titles.append(strip_title(data["title"]))

                text = data["text"].lower()
                word_count = len(text.split())
                
                if word_count >= min_char and word_count < max_char:
                    if data["label"] == 0:
                        if use_doc_freq == False:
                            for word in data["text"].lower().split(" "):
                                if word.isalpha():
                                    real_articles_word_dict[word] += 1
                        elif use_doc_freq == True:
                            for word in set(data["text"].lower().split(" ")):
                                if word.isalpha():
                                    real_articles_word_dict[word] += 1
                                    
                    elif data["label"] == 1:
                        if use_doc_freq == False:
                             for word in data["text"].lower().split(" "):
                                if word.isalpha():
                                    fake_articles_word_dict[word] += 1
                        elif use_doc_freq == True:
                             for word in set(data["text"].lower().split(" ")):
                                if word.isalpha():
                                    fake_articles_word_dict[word] += 1   

    return real_articles_word_dict, fake_articles_word_dict

##################################################
## New functions and classes added on 25 Nov 2020
##################################################

def strip_title(title, chars=["|", " - "]):
    """Splits `title` by chars and returns first or longest portion.
    
    This function is called by 
    If the split string occurs only once, resulting in splitting
    title into two strings, and the first split is three or more
    tokens long, the first split is returned.
        If the split string occurs more than once, resulting in
    splitting the string into three or more sustrings, the longest
    substrng is returned. If two or more substrings are the same
    length, the earliest substring will be returned.

    If none of the elements of `chars` are in `title`, returns
    `title` unchanged.
    """
    for char in chars:
        if title.find(char) != -1:
            splits = title.split(char)
            if (len(splits) == 2 and len(splits[0]) > 2 and len(splits[1]) > 3):
                return splits[0]
            max_idx = 0
            max_len = 0
            for idx, split in enumerate(splits):
                split_len = len(split)
                if idx == 0:
                    max_len = split_len
                else:
                    if split_len > max_len:
                        max_len = split_len
                        max_idx = idx
            return splits[max_idx]
    return title


def load_full_data(data_path, datasets, min_char=None, 
                   max_records=None, val_split=None,
                   test=False):
    """Loads article data from files stored ini `data_path`.
    
    The folder structure must be:
    data_path\
        politifact\
            fake\
            real\
        gossipcop\
            fake\
            real\
            
    Args:
        data_path: str, relative path to folders with articles.
        datasets: list, either with both["politifact", "gossipcop"], 
            or just ["politifact"] or ["gossipcop"]
        min_char: int, optional. If set to an integer, function ignores
            files with fewer than min_char characters.
        max_records: optional, int. Specifies maximum number of records
            to return.
        val_split: float, optional. Number between 0 and 1 represeting proportion
            of data in validation set.
        test: bool, optional. If True, retrieves test data.
            
    Returns:
        Python dictionary object where each key contains a Python list.
        The keys are: "texts", "labels", "article_chars", "article_words"
        "article_tokens", "title_chars", "title_tokens", "sources",
        "file_names", "titles", "stripped_titles"
        
        If val_split parameter is specified, passes a tuple of two dictionaries,
        with the first dictionary being the training data and the second dictionary
        as the validation data.
    """
    texts = []
    titles = []
    stripped_titles = []
    labels = []
    article_chars = []
    article_words = []
    article_tokens = []
    title_chars = []
    title_tokens = []
    sources = []
    file_names = []
    
    data_dir = pathlib.Path(data_path)
    for source in datasets:
        suffix = "_test" if test else "_train"
        for label in ["real", "fake"]:
            folder_path = data_dir / (source + suffix) / label
            for jfile in tqdm.tqdm(folder_path.iterdir(), desc=f"{source}:{label}"):
                if jfile.is_dir():
                    continue
                data = json.loads(jfile.read_text())
                if data["article_length"] < min_char:
                    continue
                texts.append(data["text"])
                titles.append(data["title"])
                stripped_titles.append(strip_title(data["title"]))
                labels.append(data["label"])
                article_chars.append(data["article_length"])
                article_words.append(len(data["text"].split()))
                article_tokens.append(data["article_token_len"])
                title_chars.append(data["title_length"])
                title_tokens.append(data["title_token_len"])
                sources.append(data["source"])
                file_names.append(data["file_name"])
                
    zipped_data = list(zip(texts, labels, article_chars, article_words,
                          article_tokens, title_chars, title_tokens,
                          sources, file_names, titles, stripped_titles))
    random.shuffle(zipped_data)
    
    if max_records is None:
        max_records = len(texts)
    zipped_data = zipped_data[:max_records]    
    
    if val_split is not None:
        all_data = sklearn.model_selection.train_test_split(
                                    zipped_data, test_size=val_split)
    else:
        all_data = (zipped_data,)

    results = []
    for zdata in all_data:
        result = {"texts": [x[0] for x in zdata],
                  "labels": [x[1] for x in zdata],
                  "article_chars": [x[2] for x in zdata],
                  "article_words": [x[3] for x in zdata],
                  "article_tokens": [x[4] for x in zdata],
                  "title_chars": [x[5] for x in zdata],
                  "title_tokens": [x[6] for x in zdata],
                  "sources": [x[7] for x in zdata],
                  "file_names": [x[8] for x in zdata],
                  "titles": [x[9] for x in zdata],
                  "stripped_titles": [x[10] for x in zdata]
                 }
        results.append(result)
    if len(results) == 1:
        return results[0]
    else:
        return results
    
    
class FNDataset(torch.utils.data.Dataset):
    """A dataset that with articles, titles, and metadata
    
    To instantiate an FNDataset, pass a dataset created with
    `load_full_data()` and a Huggingface tokenizer object to the
    __init__() method.
    
    When indexed, this object returns a dictionary with keys
    including input_ids, attention_mask, labels, texts, titles, etc.
    """
    def __init__(self, data, tokenizer, titles=True):
        self._encodes_titles = titles
        if titles:
            self.encodings = tokenizer(data["stripped_titles"], data["texts"],
                                       truncation=True, padding=True)
        else:
            self.encodings = tokenizer(data["texts"],
                                       truncation=True, padding=True)
        for key, val in data.items():
            setattr(self, key, val)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        for key in self.__dir__():
            if key not in ["labels", "encodings"] and key[0] != "_":
                item[key] = getattr(self, key)[idx]

        return item

    def __len__(self):
        return len(self.labels)
    
    
def datasetFN_set_labels_to_source(dataset):
    """Modifies FNDataset to be sorted by topic chooser model.
    
    This function modifies the dataset in place.
    
    Modifies FNDataset labels such that the label is 1 if the
    article comes from the politifact dataset and 0 if from the
    gossipcop dataset. The original fake and real labels are stored
    in the `fake_labels` attribute.
    
    Args: dataset, an FNDataset object.
    
    Returns a reference to the original FNDataset.
    """
    dataset.fake_labels = torch.tensor(dataset.labels)
    dataset.labels = torch.tensor([1 if x == "politifact" else 0 for x in dataset.sources])
    return dataset


def subset_FNDataset(dataset, file_names):
    """Returns a subset of an FNDataset.
    
    Args:
        dataset: An FNDatset object
        file_names: List of strings, the file names that should be
            retained in the dataset.
            
    Returns a deep copy of the FNDataset object.
    """
    filter_list = [fname in file_names for fname in dataset.file_names]
       
    subset = copy.deepcopy(dataset)
    subset.encodings["input_ids"] = list(itertools.compress(
        subset.encodings["input_ids"], filter_list))
    subset.encodings["attention_mask"] = list(itertools.compress(
        subset.encodings["attention_mask"], filter_list))
                         
    for attrib in subset.__dir__():
        if attrib[0] != "_" and attrib != "encodings":
            setattr(subset, attrib,
                    list(itertools.compress(getattr(subset, attrib),
                                            filter_list)))
    return subset