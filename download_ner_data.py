import sys
from pathlib import Path
from subprocess import check_output

from transformers import AutoTokenizer


def preprocess(dataset, tokenizer, max_len, ofp):
    subword_len_counter = 0

    max_len -= tokenizer.num_special_tokens_to_add()

    with open(dataset, "rt") as fp:
        for line in fp:
            line = line.rstrip()

            if not line:
                print(line, end="\n", file=ofp)
                subword_len_counter = 0
                continue

            token = line.split()[0]

            current_subwords_len = len(tokenizer.tokenize(token))

            # Token contains strange control characters like \x96 or \x95
            # Just filter out the complete line
            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                print("", end="\n", file=ofp)
                print(line, end="\n", file=ofp)
                subword_len_counter = current_subwords_len
                continue

            subword_len_counter += current_subwords_len
            print(line, end="\n", file=ofp)


def download_data(url, filepath):
    response = requests.get(url)
    if response.ok:
        with open(filepath, "w") as fp:
            fp.write(response.content.decode("utf8"))
        return filepath


if __name__ == "__main__":

    datadir = Path(sys.argv[1])
    model_name_or_path = sys.argv[2]
    max_len = int(sys.argv[3])

    if not datadir.is_dir():
        datadir.mkdir(exist_ok=True)

    # ## GDrive at https://drive.google.com/drive/folders/1kC0I2UGl2ltrluI9NqDjaQJGw5iliw_J
    # ## Monitor for changes and eventually migrate to nlp dataset
    output = check_output(
        "curl -L \"https://drive.google.com/uc?export=download&id=1Jjhbal535VVz2ap4v4r_rN1UEHTdLK5P\" | grep -v \"^#\" | cut -f 2,3 | tr '\t' ' '",
        shell=True,
        encoding="utf-8",
    )
    with open(datadir / "train.bio", "wt") as ofp:
        ofp.write(output)
    output = check_output(
        "curl -L \"https://drive.google.com/uc?export=download&id=1ZfRcQThdtAR5PPRjIDtrVP7BtXSCUBbm\" | grep -v \"^#\" | cut -f 2,3 | tr '\t' ' '",
        shell=True,
        encoding="utf-8",
    )
    with open(datadir / "dev.bio", "wt") as ofp:
        ofp.write(output)
    output = check_output(
        "curl -L \"https://drive.google.com/uc?export=download&id=1u9mb7kNJHWQCWyweMDRMuTFoOHOfeBTH\" | grep -v \"^#\" | cut -f 2,3 | tr '\t' ' '",
        shell=True,
        encoding="utf-8",
    )
    with open(datadir / "test.bio", "wt") as ofp:
        ofp.write(output)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    with open(datadir / "train.txt", "wt") as ofp:
        preprocess(datadir / "train.bio", tokenizer, max_len, ofp)
    with open(datadir / "dev.txt", "wt") as ofp:
        preprocess(datadir / "dev.bio", tokenizer, max_len, ofp)
    with open(datadir / "test.txt", "wt") as ofp:
        preprocess(datadir / "test.bio", tokenizer, max_len, ofp)

    output = check_output(
        'cat {} {} {} | cut -d " " -f 2 | grep -v "^$"| sort | uniq'.format(
            datadir / "train.txt",
            datadir / "dev.txt",
            datadir / "test.txt",
        ),
        shell=True,
        encoding="utf-8",
    ).strip()
    with open(datadir / "labels.txt", "wt") as ofp:
        ofp.write(output)
