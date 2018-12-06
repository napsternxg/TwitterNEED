# coding: utf-8

from bs4 import BeautifulSoup
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm

tokenizer = TweetTokenizer()
    
def parse_tweet_xml(tweet):
    data = {}
    data["tweet_id"] = tweet.find("TweetId").text
    data["tweet_text"] = tweet.find("TweetText").text
    data["tweet_group"] = tweet.find("TweetGroup").text
    data["annotations"] = []
    for mention in tweet.find_all("Mention"):
        mention_data = {}
        mention_data["text"] = mention.find("Text").text
        mention_data["start_idx"] = int(mention.find("StartIndx").text)
        mention_data["end_idx"] = mention_data["start_idx"] + len(mention_data["text"])
        try:
            assert data["tweet_text"][mention_data["start_idx"]:mention_data["end_idx"]] == mention_data["text"], (
            "Mismatch. Expected: {}, Found: {}"
            ).format(data["tweet_text"][mention_data["start_idx"]:mention_data["end_idx"]], mention_data["text"])
        except AssertionError as e:
            # in the micropost file certain start indexes are shifted by a value. This can be used to fix that. 
            for inc in [1,2,3]:
                if data["tweet_text"][mention_data["start_idx"]+inc:mention_data["end_idx"]+inc] == mention_data["text"]:
                    mention_data["start_idx"] += inc
                    mention_data["end_idx"] += inc
                    break
            else:
                print("couldnt fix")
                raise e
        mention_data["entity"] = mention.find("Entity").text
        data["annotations"].append(mention_data)
    return data


def get_conll_from_annotation(annotations, text):
    # get start, end, type
    annotations = [(v["start_idx"], v["end_idx"], "ENTITY") for v in annotations]
    # Sort annotations by start
    annotations = sorted(annotations, key=lambda v: v[0])
    if annotations:
        assert annotations[-1][1] <= len(text), (
                "Text length should be greater than or equal to last annotation end. "
                "len({!r}) = {}, annotations={}"
                ).format(text, len(text), annotations)
    prev = 0
    conll_data = []
    for start, end, entity_type in annotations:
        if prev != start:
            tokens = tokenizer.tokenize(text[prev:start])
            conll_data.extend([(t, "O") for t in tokens])
        tokens = tokenizer.tokenize(text[start:end])
        # NEEL entity marks do not consider the hashtag or mention sign. 
        # Hence we need to add those back to the token
        # to ensure the tokenization is compatible with nltk tokenizer
        if start > 0 and text[start-1] in set(["@", "#"]):
            tokens[0] = "{}{}".format(conll_data.pop()[0], tokens[0])
        conll_data.extend([
            (t, "{}-{}".format("B" if i == 0 else "I", entity_type))
            for i,t in enumerate(tokens)
            ])
        prev = end
    tokens = tokenizer.tokenize(text[prev:])
    conll_data.extend([(t, "O") for t in tokens])
    conll_str = "\n".join("\t".join(row) for row in conll_data)
    return conll_str

def xml2conll(input_file, output_file):
    with open(input_file) as fp:
        xml_data = fp.read()
    xml_tree = BeautifulSoup(xml_data, "xml")
    with open(output_file, "w+") as fp_out:
        for tweet in tqdm(xml_tree.find_all("Tweet"), desc=input_file):
            try:
                tweet_data = parse_tweet_xml(tweet)
                conll_str = get_conll_from_annotation(tweet_data["annotations"], tweet_data["tweet_text"])
            except AssertionError as e:
                print(e)
                raise
                continue
            except IndexError as e:
                print("text={!r}, annotations={}".format(tweet_data["annotations"], tweet_data["tweet_text"]))
                raise
            print(conll_str, file=fp_out, end="\n\n")

DEFAULT_INPUT_OUTPUT_FILES = [
        ("data/raw/Brian Collection.xml", "data/conll/Brian Collection.conll"),
        ("data/raw/Mena Collection.xml", "data/conll/Mena Collection.conll"),
        ("data/raw/Microposts2014_Collection_train.xml", "data/conll/Microposts2014_Collection_train.conll"),
        ]


if __name__ == "__main__":
    for input_file, output_file in DEFAULT_INPUT_OUTPUT_FILES:
        xml2conll(input_file, output_file)
