from datetime import time
from urllib.parse import unquote_plus,urlparse
from html import unescape
from wordsegment import load, segment
#from transformers import BertTokenizer

import re
import time
import numpy as np
import enchant
from transformers import BertTokenizer
# Final Version 
d = enchant.Dict(tag="en_US")   # Initialize 'en_US' dictionary
load()  # Load WordSegment library
tokenizer_ = BertTokenizer.from_pretrained('bert-base-uncased')

### Suspicious Pattern Extraction ###
def suspicious_pattern_extractor(url:str):
    pre='None'
    url_word=[]
    url_word_digit=''
    url_word_letter=''
    url_word_symbol=''

    for i in url:
        if i.isalpha():
            if pre=='digit': ## Digit found after Alphabets
                url_word.append('[digit]')
                url_word_digit=''
            if pre=='symbol': ## Symbols found after Alphabets
                url_word.append(url_word_symbol)
                url_word_symbol=''
            url_word_letter=url_word_letter+i
            pre='letter'
        
        if i.isnumeric():
            if pre=='letter': ## Letter found after Numerics
                url_word.append(url_word_letter)
                url_word_letter=''
            if pre=='symbol': ## Symbol found after Numerics
                url_word.append(url_word_symbol)
                url_word_symbol=''
            url_word_digit=url_word_digit+i
            pre='digit'
        
        if not i.isalnum():
            if pre=='letter': ## Letter found after Symbols
                url_word.append(url_word_letter)
                url_word_letter=''
            if pre=='digit': ## Digit found after Symbols
                url_word.append('[digit]')
                url_word_digit=''
            url_word_symbol=url_word_symbol+i
            pre='symbol'
    
    if pre=='letter':
        url_word.append(url_word_letter)
    elif pre=='digit':
        url_word.append('[digit]')
    else:
        url_word.append(url_word_symbol)

    return url_word

### Recursive Tokenizer Algorithm ###
def recursive_tz_journal(word,tokenizer_:BertTokenizer):

    # Step 1: Tokenized Words [List] using Bert Tokenizer
    word_tz=tokenizer_.tokenize(word)  

    if len(word_tz)==1: # Single Word <Cannot Tokenize anymore>
        # Return Single Word
        return word_tz[0]
    
    # Step 2: Recursion
    else:   # Multiple Words <Recursively Tokenize from 2nd Word again>

        rec_str=word[len(word_tz[0]):]
        
        return word_tz[0]+'-'+recursive_tz_journal(rec_str,tokenizer_)

### SegURLizer Algorithm ###
def SegURLizer(url:str):
    
    wc_word=[] 
    sp_chars=[] 
    ws_word=[] 
    tmpSegURL=[]  
    segURL=[]   
    
    # Record START time of SegURLizer
    start_time = time.time()

    # Unquote URLs
    url=unescape(unquote_plus(url, encoding="utf-8")) 
    
    # URL Parser
    if not str(url).lower().startswith('http:') and not str(url).lower().startswith('https:'):
        parsedURL=urlparse('http://'+url)
    else:
        parsedURL=urlparse(url)
    
    # Concatenate Protocol + Domain + Path
    if not url.lower().startswith('http'):
        url = parsedURL.netloc.lower()+parsedURL.path.lower()
    else:
        url = parsedURL.scheme.lower()+'://'+parsedURL.netloc.lower()+parsedURL.path.lower()

    # Step 0.0: Tokenizing by Special Characters
    spTokenized=re.split(r"[-_;:,.=?^@\s$&?+!*\'()[\]{}|\"%~#<>/]",url)

    # Step 0.1: Extrating Special Characters
    string_check= re.compile(r'[-_;:,.=@\s$&?^+!*\'()[\]{}|\"%~#<>/]')  
    for i in url:   
        if (not i.isalnum()) and (string_check.search(i)!=None): 
            sp_chars.append(i)

    # Step 1: Validating URL using Word Decomposer for Suspicious Pattern
    # + Truncating Tokenized Word in range of [0:50]
    for each in spTokenized:

        # Step 1.1: To validate each WORD is NOT LARGER THAN 50, 
        if len(each)>0 and len(each)<=50:

            # Step 1.2: Decompose Consecutive Letter-Digit-Symbol patten
            str_cont_lst = suspicious_pattern_extractor(each.lower()) #### Tokenize by Letter-Digit-Letter/Digit-Letter-Digit

            # Add validated WORD [0<word<=50] to wc_word
            wc_word=wc_word+str_cont_lst

        elif len(each)>50:

            # Step 1.2: Decompose Consecutive Letter-Digit-Symbol patten
            str_cont_lst = suspicious_pattern_extractor(each[:50].lower())    #### Tokenize by Letter-Digit-Letter/Digit-Letter-Digit

            # Add truncated WORD [0<word<=50] to wc_word
            wc_word=wc_word+str_cont_lst

        if len(sp_chars) >0:    # Reorder Validated Word [List] : wc_word, w/ respective Special Characters [List] : sp_chars
            if str(sp_chars[0])==' ':
                sp_chars.remove(sp_chars[0])
            else:
                wc_word.append(str(sp_chars[0]))
                sp_chars.remove(sp_chars[0])
        
    # Step 2: WordSegmenting a Validated Word [List] : wc_word
    for decomp_word in wc_word:
        if str(decomp_word).isalnum():
            ws=segment(decomp_word)
            if len(ws) == 0:
                ws_word = ws_word + [decomp_word]
            ws_word=ws_word+ws
        else:
            ws_word=ws_word+[decomp_word]
        
    word_latency_recursion=0.0
    # Step 3: Recursive_tokenization
    for word in ws_word:
        
        # Step 3.1: Validating w/ Enchant Dictionary
        if d.check(word) or d.check(str(word).upper()):
            tmpSegURL.append(word) # Add to tmpSegURL [list]
        
        # Check alphanumeric character a-zA-Z0-9
        elif str(word).encode().isalpha(): 
            
            # Step 3.2: Recursive Tokenization of each word after WordSegmented
            start_recursion = time.time()
            wr_word=re.split(r"[-]",recursive_tz_journal(word,tokenizer_)) 
            stop_recursion = time.time()
                
            word_latency_recursion = word_latency_recursion + (stop_recursion-start_recursion)
            
            # Step 3.3: Prefixing "##"
            for i in wr_word:
                word2append = ''

                # 1. English Dictionary Check==True & 2. First Word Check==True
                if (d.check(i) or d.check(str(i).upper())) and (i ==wr_word[0]):    
                    word2append = str(i).lower() 

                # 1. English Dictionary Check==True & 2. First Word Check==Flase
                elif (d.check(i) or d.check(str(i).upper())) and (i !=wr_word[0]):  
                    word2append = str(i).lower()   

                # 1. English Dictionary Check==False & 2. First Word Check==True
                elif (not (d.check(i) or d.check(str(i).upper()))) and (i ==wr_word[0]):    
                    word2append = str(i).lower()   
                        
                else:
                    # Prefix '##', meaning as RANDOM words
                    word2append = '##'+ str(i).lower()  
                tmpSegURL.append(word2append)
        
        # Replace Numeric Character w/ [digit]      
        else: 
            if str(word).isnumeric():
                tmpSegURL.append('[digit]')
            else:
                tmpSegURL.append(word)

    end_time=time.time()    # Record END time of SegURLizer
    segURLizer_timer_=end_time-start_time  # Record preprocessing time of SegURLizer
    url_latency_recursion = word_latency_recursion # Record preprocessing time of Recursion
    
    string_check= re.compile(r'[-_;:,.=@\s$&?^+!*\'()[\]{}|\"%~#<>/]')  # Special Characters [string]

    # Step 4: Converting non-English words to [UNK]
    # Check tmpSegURL [list] is neither alpha numeric characters nor special characters 
    for each in tmpSegURL:

        # Substituting non-English words as [UNK]
        # Is neither alpha numeric characters nor special characters ?
        if (string_check.search(each)==None) and (not str(each).encode().isalnum()) and (len(str(each).encode().strip())>0):    
            segURL.append('[UNK]')  # Convert to [UNK] word & Add to Final Word [List] : segURL

        elif (len(str(each).encode().strip())>0):
            segURL.append(each) # Add to Final Word [List] : segURL

    segURL = str(' '.join(segURL))
    return url, segURL, segURLizer_timer_, url_latency_recursion
