import urllib.parse
import re

def waf_tokenizer(text):
    text = urllib.parse.unquote(str(text)).lower()
    return re.findall(r'\w+|[^\w\s]', text)