import re


class Tokenizer(object):

    REPLACE_ABBRE_NOT = re.compile("n't"), " not"
    REPLACE_QUOTES = re.compile(r'[‘’“”"~]'), " "
    REPLACE_SINGLE_QUOTES_BOUNDARY = re.compile(r"\'\b|\b\'"), " "
    REPLACE_PARENTHESIS = re.compile(r'[()\[\]]'), " "
    REPLACE_PUNCTUATION = re.compile(r'[\?!,:;]'), " "
    REPLACE_PERIOD_WORD_BOUNDARY = re.compile(r'\. '), " "
    REPLACE_PERIOD_EOL = re.compile(r'\.$'), " "
    REPLACE_BAR = re.compile(r'\|'), " "
    REPLACE_DASH = re.compile(r'[\-—_]'), " "
    REPLACE_DOTS = re.compile(r'…|\.{2,}'), " "
    REPLACE_TAGS = re.compile(r'#'), " "
    REPLACE_SLASH = re.compile(r'/'), " "

    TOKEN_REGEXES = [
        REPLACE_QUOTES,
        REPLACE_SINGLE_QUOTES_BOUNDARY,
        REPLACE_PARENTHESIS,
        REPLACE_PUNCTUATION,
        REPLACE_BAR,
        REPLACE_DASH,
        REPLACE_ABBRE_NOT,
        REPLACE_TAGS,
        REPLACE_DOTS,
        REPLACE_SLASH,
        REPLACE_PERIOD_WORD_BOUNDARY,
        REPLACE_PERIOD_EOL
    ]

    def tokenize(self, text):
        text = text.lower()
        for regexp, substitution in self.TOKEN_REGEXES:
            text = regexp.sub(substitution, text)
        return text.split()

