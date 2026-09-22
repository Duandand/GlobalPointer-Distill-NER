# coding=utf-8
import collections
import unicodedata


# 统一unicode编码
def convert_to_unicode(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


# 导入字典
def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


# 判断是否是控制字符（此处认为`\t`,`\n`,`\r`不是控制字符）
def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


# 判断是否为空白
def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


# 按空格分词
def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


# 判断是否是标点符号（把ASCII表中非字符和数字的字符，以及unicode里分类类型为标点的都认为是标点符号）
def _is_punctuation(char):
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


# token与id之间的相互转换
def convert_by_vocab(vocab, items):
    output = []
    for item in items:
        if item in vocab:
            output.append(vocab[item])
        else:
            try:
                output.append(vocab['[UNK]'])
            except:
                output.append("[UNK]") # add by ddd
    return output


class FullTokenizer(object):

    def __init__(self, vocab_file, do_lower_case=True, num_seperation=False):
        self.do_lower_case = do_lower_case
        self.num_seperation = num_seperation
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    def tokenize(self, text):
        tokens = self.basic_tokenizer.tokenize(text)

        if self.num_seperation:
            tokens_num_seperation = []
            for token in tokens:
                if token.isalnum() and not token.isalpha():
                    tokens_num_seperation.extend(list(token))
                else:
                    tokens_num_seperation.append(token)
            return tokens_num_seperation
        else:
            return tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


# 基本分词器
class BasicTokenizer(object):

    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    # 去除声调
    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    # 按标点符号分词
    def _run_split_on_punc(self, text):
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    # 在每个CJK字符两侧加空格
    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    # 判断是否是CJK字符-https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    def _is_chinese_char(self, cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or
                (cp >= 0x3400 and cp <= 0x4DBF) or
                (cp >= 0x20000 and cp <= 0x2A6DF) or
                (cp >= 0x2A700 and cp <= 0x2B73F) or
                (cp >= 0x2B740 and cp <= 0x2B81F) or
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or
                (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True
        return False

    # 清理文本，包括去除控制字符，统一空白为空格，全角转半角
    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            elif cp == 12288:  # 全角空格直接转换
                cp = 32
            elif 65374 >= cp >= 65281:  # 全角字符（除空格）根据关系转化
                cp -= 65248
            char = chr(cp)

            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
