import re, numpy as np, pandas as pd
from tld import get_tld
from math import factorial,log2
import socket
from urllib.parse import urlparse,unquote_plus

def F1_checkNoOfDotIn_(parsed_:str):
    return parsed_.count('.')

def F2_checkNoOfSPIn_(parsed_:str,noOfDotIn_:int): # No of Special Charactr in Domain excluding Dot
    return sum(1 for i in parsed_ if not i.isalnum()) - noOfDotIn_

def F3_checkNoOfDigit(str_:str):
    return sum(1 for i in str_ if i.isnumeric())

def F4_checkNoOfAlpha(netloc:str): # No of Alhpabetic Character in Domain
    return sum(1 for i in netloc if i.isalpha())

def retrieveIPAddress(domain:str):
    ipInDomain = 0
    # Get the IP address associated with the hostname
    try:
        ip_address = socket.gethostbyname(domain)
        ipInDomain = 1
    except socket.gaierror:
        return '',ipInDomain
    return ip_address,ipInDomain

def is_valid_ip_in_url(ip_address):

    # Regular expression pattern to match a valid IP address
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    
    # Search for the IP pattern in the URL
    ip_match = re.search(ip_pattern, ip_address)
    
    if ip_match:
        return True
    else:
        return False
    
def F5_retrieveTldFldInFQDN(parsedDomain:str,netloc:str):
    ipInFQDN = 0
    ip_address = ''
    try:
        res = get_tld(parsedDomain,as_object=True)
    except:
        ip_address=netloc.split(':')[0]
        if is_valid_ip_in_url(ip_address):
            ipInFQDN=1
        else:
            ip_address=''
            ipInFQDN=0
        return ip_address,'','',netloc,netloc, ipInFQDN,0,0,0,0
    
    topdomain = res.tld
    firstdomain = res.fld
    subdomain = res.subdomain

    noOfSusWordInFQDN = checkSusWordInFQDN(netloc) 
    noOfSubLevelInFQDN = checkNoOfSubLevelInFQDN(netloc)
    brandName,lenOfbrandName = checkLenOfMainDomainInDomain(firstdomain,topdomain)
    lenOfSubDomainInFQDN = checkLenOfSubDomainInFQDN(subdomain)
    
    return ip_address,subdomain,topdomain,firstdomain,brandName,ipInFQDN,noOfSusWordInFQDN,noOfSubLevelInFQDN,lenOfbrandName,lenOfSubDomainInFQDN

def F6_checkAvgWordLength(len_words):
    return np.round(np.mean(len_words),3) if len(len_words) > 0 else 0

def F7_checkLongWordLength(len_words):
    return np.max(len_words) if len(len_words) > 0 else 0

def F8_checkHostLength(len_words):
    return sum(len_words)

def F9_checkNoOfHyphen(hostName:str):
    return hostName.count('-')

def F10_checkNoOfUnderscore(hostName:str):
    return hostName.count('_')

def F11_checkGTLDExist(tld:str):
    gTLDs = ['.com','.info','.net','.org','.biz','.name','.pro','.arpa','.aero','.asia','.cat','.coop','.edu','.gov','.int','.jobs','.mil','.mobi','.tel','.travel','.xxx']

    for gTLD in gTLDs:
        if tld.endswith(gTLD):
            return 1
        if str('.'+tld).__contains__(gTLD):
            split_url = str('.'+tld).split(gTLD+'.')
            if len(split_url) > 1:
                return 1
    return 0

def F12_computeEntropy(input_str:str): # Entropy of Non-alphanumeric Character
    total_length=len(input_str)
    nan_count={}
    for i in input_str:
        if not i.isalnum():
            if i not in nan_count.keys():
                nan_count[i]=1
            else:
                nan_count[i]=nan_count[i]+1
    prob_nan=np.round(sum((i/total_length)*log2(total_length/i) for i in nan_count.values()),3) # Log base 2
    
    return prob_nan

def F13_checkBrandExist(subdomain:str,path:str,top_500:set):
    brandInSub, brandInPath = 0, 0
    for row in top_500:
        row = str(row)
        if subdomain.__contains__(row):
            foundInSub = F13_checkBrandExistInDomain(row=row,subdomain=subdomain)
            if foundInSub:
                brandInSub = foundInSub
        if path.__contains__(row):
            foundInPath = F13_checkBrandExistInPath(row=row,path=path)
            if foundInPath:
                brandInPath = foundInPath
        
    return brandInSub,brandInPath

def F13_checkBrandExistInDomain(row:str,subdomain:str):
    pattern = r"[/\.]([\w-]+\.)?" + re.escape(row) + r"\."
    match = re.search(pattern, subdomain)

    if bool(match):
        return 1
    return 0

def F13_checkBrandExistInPath(row:str,path:str):
    pattern_path = r"[./]" + re.escape(row) + r"[./]"
    match_path = re.search(pattern_path, path)

    if bool(match_path):
        return 1
    return 0

def F14_checkNoSubfolder(parsedPath:str): # No of Sub-folder and Slash_index in Path
    count = 0
    indexOfSlashInPath = []
    if len(parsedPath)>1:
        for i in range(len(parsedPath)):
            if (i == len(parsedPath)-1) and (parsedPath[i] == '/'):
                count += 1
                indexOfSlashInPath.append(i)
            else:
                if (parsedPath[i]=='/') and (i==0) and (parsedPath[i+1]!='/'):
                    indexOfSlashInPath.append(i)
                elif (parsedPath[i]=='/') and (i!=0) and (parsedPath[i+1]!='/') and (parsedPath[i-1]!='/'):
                    count += 1
                    indexOfSlashInPath.append(i)
    else:
        return 0,[]
    return count,indexOfSlashInPath

def F15_checkNoOfAlnumChar(parsedPath:str): # No of Alpha-Numeric Character in Path
    return sum(1 for i in parsedPath if i.isalnum())

def F16_checkNoOfSPInPath(parsedPath:str): # No of Special Charactr in Path excluding '/'
    return sum(1 for i in parsedPath if not i.isalnum()) - parsedPath.count('/')

def F17_calculate_distributions(parsedPath:str,indexOfSlashInPath:list,noOfSubfolderInPath:int):
    # Step 1 : Calculate k
    if noOfSubfolderInPath == 0:
        return 0.0, 0.0
    alpha_seq_in_pth, k_alpha_seq_depth, num_seq_in_pth, k_num_seq_depth = checkNoOfAlnumOfSeqInPath(parsedPath,indexOfSlashInPath,noOfSubfolderInPath)
    
    # Step 2 : Calculate Lambda
    lambda_alpha, lambda_num = calculate_lambda(alpha_seq_in_pth,k_alpha_seq_depth,num_seq_in_pth,k_num_seq_depth,noOfSubfolderInPath)

    # Step 3 : Calculate poison
    poison_alpha, poison_num = calculate_poison(lambda_alpha,lambda_num,k_alpha_seq_depth,k_num_seq_depth)
    
    return poison_alpha, poison_num

def F18_checkQueryInPath(parsedQuery:str):  # Find Query in Path
    return 1 if len(parsedQuery)>0 else 0

def F19_checkNoOfHTTPSInPath(all_path:str):
    HTTPandWWWInPath = re.findall(r'((?:http[s]?/)|(?:www/))', all_path.lower())
    return len(HTTPandWWWInPath)

def F20_checkAtInURL(url:str):
    return 1 if '@' in url else 0

def F21_checkBaseUrlLength(url:str):
    return len(url)

def F22_checkNoOfQuestionmark(url:str):
    return url.count('?')

def F23_checkNoOfSlash(unquote_url:str):
    return unquote_url.count('/')

def F24_checkRatioOfHyphen(url:str):
    return np.round(url.count('-')/len(url),3)

def checkSusWordInFQDN(netloc:str): # No of Suspicious Word in Domain
    misspelled_words = []
    netloc_words = re.split(r"[-_;:,.=?@\s$&?+!*\'()[\]{}|\"%~#<>/]",netloc)   # Tokenizing by Special Characters
    netloc_ = ' '.join(netloc_words).strip().split()
    for ele_ in netloc_:
        if ele_.isalnum() and not ele_.isalpha() and not ele_.isnumeric():
            misspelled_words.append(ele_)
    noOfSusWordInFQDN = len(misspelled_words)

    return noOfSusWordInFQDN

def checkLenOfSubDomainInFQDN(sub_:str): # Length of Sub-domain in Domain
    return len(sub_)

def checkLenOfMainDomainInDomain(fld_:str, tld_:str): # Length of Main-domain=domainName in Domain
    brandName = fld_[:-len(tld_)-1]
    return brandName,len(brandName)

def checkNoOfSubLevelInFQDN(netloc:str): # Level of Domain
    return len(netloc.split('.'))

def checkSeqfolderInPath(parsedPath:str,indexOfSlashInPath:list,noOfSubfolderInPath:int): # Sequence of Alpha-Numeric Character of Sub-folder in Path 
    if len(indexOfSlashInPath)>0 and noOfSubfolderInPath>0:
        sequenceOfSubfolderInPath = []
        indx = 0
        head = indx
        tail = indx+1
        while (tail<len(indexOfSlashInPath)):
            sequenceOfSubfolderInPath.append(parsedPath[indexOfSlashInPath[head]+1:indexOfSlashInPath[tail]])
            head = tail
            tail = head+1
        return sequenceOfSubfolderInPath
    else:
        return []

def checkNoOfAlnumOfSeqInPath(parsedPath:str,indexOfSlashInPath:list,noOfSubfolderInPath:int):
    # Initialize alpha and num array with max_depth
    alpha_in_pth = [0]*noOfSubfolderInPath
    num_in_pth = [0]*noOfSubfolderInPath
    alpha_depth, num_depth = 0, 0
    
    # Check NULL in Path
    if noOfSubfolderInPath == 0:
        return alpha_in_pth, alpha_depth, num_in_pth, num_depth
    
    # Tokenize '/' to find depth k
    sequenceOfSubfolderInPath = checkSeqfolderInPath(parsedPath,indexOfSlashInPath,noOfSubfolderInPath)
    
    # Count depth k and save alpha-numeric character for weight w
    for pth in range(len(sequenceOfSubfolderInPath)):
        alpha_count = 0
        num_count = 0
        for char_ in sequenceOfSubfolderInPath[pth]:
            if str(char_).isalpha():
                alpha_count += 1
            if str(char_).isnumeric():
                num_count += 1
        if alpha_count > 0:
            alpha_in_pth[pth] = alpha_count
            alpha_depth += 1
        if num_count > 0:
            num_in_pth[pth] = num_count
            num_depth += 1
    
    return alpha_in_pth, alpha_depth, num_in_pth, num_depth

# Calculate Lambda
def calculate_lambda(alpha_seq_in_pth,k_alpha_seq_depth,num_seq_in_pth,k_num_seq_depth,noOfSubfolderInPath):
    # Sum alpha,num 
    alpha_count = np.sum(alpha_seq_in_pth)
    num_count = np.sum(num_seq_in_pth)
    total_counts = alpha_count + num_count
    lambda_alpha, lambda_num = 0.0, 0.0
    # Calculate weight w
    if total_counts > 0:
        if alpha_count > 0:
            w_ = alpha_count / total_counts
            w_alpha = np.round(w_,3)
            # Calculate lambda
            lambda_alpha = np.round(w_alpha*k_alpha_seq_depth/noOfSubfolderInPath,3)
        if num_count > 0:
            w_ = num_count / total_counts
            w_num = np.round(w_,3)
            # Calculate lambda
            lambda_num = np.round(w_num*k_num_seq_depth/noOfSubfolderInPath,3)
    
    return lambda_alpha,lambda_num

# Calculate poison
def calculate_poison(lambda_alpha,lambda_num,k_alpha_seq_depth,k_num_seq_depth):
    poison_alpha, poison_num = 0.0, 0.0
    # Calculate poison distribution
    e = 2.718
    if lambda_alpha > 0.0:
        poison_alpha = np.round(((e**(-lambda_alpha)) * (lambda_alpha**k_alpha_seq_depth)) / factorial(k_alpha_seq_depth),3)
    if lambda_num > 0.0:
        poison_num = np.round(((e**(-lambda_num)) * (lambda_num**k_num_seq_depth)) / factorial(k_num_seq_depth),3)
    return poison_alpha,poison_num

def parseWWW(parsedName:str):
    if parsedName.startswith('www.'):
        parsedName = parsedName[4:]
    elif parsedName.startswith('ww2.'):
        parsedName = parsedName[4:]
    return parsedName

def extractWordLengthInFQDN(hostName:str):
    hostname_words = re.split(r"[-_.=?/]",hostName)   # Tokenizing by Special Characters
    hostname_words = ' '.join(hostname_words).split()
    len_words = [len(word) for word in hostname_words]
    return len_words

def extractWordLengthInDomain(domain:str):
    domain_words = re.split(r"[-_.=?/]",domain)   # Tokenizing by Special Characters
    domain_words = ' '.join(domain_words).split()
    len_words = [len(word) for word in domain_words]
    return len_words

def extract_wordSeq(path:str):
    word_seq = re.split(r"[,-.()[\]/]",path)
    word_len = [len(s) for s in word_seq if s !='']
    word_seq = [s.lower() for s in word_seq if s !='']
    return word_len,word_seq

def checkPhishHintedWordInHostname(hostName:str):
    phish_hinted_list = ['wp','login','includes','admin','content','site','images','js','alibaba','css','myaccount','dropbox','themes','plugins','signin','view']
    hostname_words = re.split(r"[-_.=?/]",hostName.lower())   # Tokenizing by Special Characters
    return int(bool(set(hostname_words).intersection(phish_hinted_list))==True)

def checkPhishHintedWordInPath(word_seq:list):
    phish_hinted_list = ['wp','login','includes','admin','content','site','images','js','alibaba','css','myaccount','dropbox','themes','plugins','signin','view']
    return int(bool(set(word_seq).intersection(phish_hinted_list))==True) if len(word_seq) > 0 else 0

def func_extractor(url:str,top_500:set):

    parsedDomain , parsedScheme = '' , ''
    unquote_url = unquote_plus(url.strip())
    if unquote_url.startswith('http'):
        parsedURL = urlparse(unquote_url)
        parsedScheme = parsedURL.scheme+'://'
    else:
        parsedURL = urlparse('https://'+unquote_url)
        parsedScheme = ''

    parsedDomain = parsedScheme+parsedURL.netloc
    parsedPath = parsedURL.path
    parsedParam = parsedURL.params
    if len(parsedURL.query) > 1:
        parsedQuery = '?' + parsedURL.query
    else:
        parsedQuery = parsedURL.query
    parsedFragment = parsedURL.fragment

    ipAddress,subdomain,topdomain,firstdomain,domainName,D7_ipInFQDN,D8_noOfSusWordInFQDN,D9_noOfSubLevelInFQDN,D10_lenOfBrandname,D11_lenOfSubDomainInFQDN = F5_retrieveTldFldInFQDN(parsedDomain,parsedURL.netloc)
    domain = firstdomain

    # Domain-based Features
    D1_noOfDotInFQDN = F1_checkNoOfDotIn_(parsedURL.netloc)
    D2_noOfDotInDomain = F1_checkNoOfDotIn_(domain)
    D3_noOfSPInFQDN = F2_checkNoOfSPIn_(parsedURL.netloc,D1_noOfDotInFQDN)
    D4_noOfSPInDomain = F2_checkNoOfSPIn_(domain,D2_noOfDotInDomain)
    D5_noOfDigitInFQDN = F3_checkNoOfDigit(parsedURL.netloc)
    D6_noOfAlphaCharInFQDN = F4_checkNoOfAlpha(parsedURL.netloc)
    
    domainWordLength = extractWordLengthInFQDN(parsedURL.netloc)
    D12_avgWordLengthInFQDN = F6_checkAvgWordLength(domainWordLength)
    D13_longWordLengthInFQDN = F7_checkLongWordLength(domainWordLength)
    D14_LengthInDomain = F8_checkHostLength(extractWordLengthInDomain(domain))
    D15_noOfHyphenInFQDN = F9_checkNoOfHyphen(parsedURL.netloc)
    D16_noOfUnderscoreInFQDN = F10_checkNoOfUnderscore(parsedURL.netloc)
    D17_gTLDExistInFQDN = F11_checkGTLDExist(topdomain)
    D18_entropyNANInFQDN = F12_computeEntropy(parsedURL.netloc)
    D19_brandInSubdomain, P1_brandInPath = F13_checkBrandExist(subdomain,parsedPath+parsedParam+parsedQuery+parsedFragment,top_500)

    # Path-based Features
    P2_noOfSubfolderInPath, indexOfSlashInPath = F14_checkNoSubfolder(parsedPath)
    P3_noOfAlnumCharInPath = F15_checkNoOfAlnumChar(parsedPath)
    P4_noOfSPInPath = F16_checkNoOfSPInPath(parsedPath)
    P5_poiDistriOfAlphaInPath, P6_poiDistriOfNumInPath = F17_calculate_distributions(parsedPath,indexOfSlashInPath,P2_noOfSubfolderInPath)
    # Query-based Features
    P7_queryExistInQuery = F18_checkQueryInPath(parsedQuery)
    P8_noOfDigitInPath = F3_checkNoOfDigit(parsedPath)
    pathWordLength,pathWordSeq = extract_wordSeq(parsedPath)
    P9_avgWordLengthInPath = F6_checkAvgWordLength(pathWordLength)
    P10_longWordLengthInPath = F7_checkLongWordLength(pathWordLength)
    P11_noOfHTTPSandWWWInPath = F19_checkNoOfHTTPSInPath(parsedPath+parsedParam+parsedQuery+parsedFragment)
    P12_entropyNANInPath = F12_computeEntropy(parsedPath)

    # URL-based Features
    strpURL = unquote_url.split('?')[0].strip()
    U1_atExistInURL = F20_checkAtInURL(unquote_url)
    U2_baseUrlLengthInURL = F21_checkBaseUrlLength(strpURL)
    U3_noOfQuestionmarkInURL = F22_checkNoOfQuestionmark(unquote_url)
    U4_noOfSlashInURL = F23_checkNoOfSlash(unquote_url)
    U5_ratioOfHyphenInURL = F24_checkRatioOfHyphen(strpURL)

    phishHintedWordInFQDN  = checkPhishHintedWordInHostname(parsedURL.netloc)
    phishHintedWordInPath = checkPhishHintedWordInPath(pathWordSeq)

    return D1_noOfDotInFQDN,D2_noOfDotInDomain,D3_noOfSPInFQDN,D4_noOfSPInDomain,D5_noOfDigitInFQDN,D6_noOfAlphaCharInFQDN,D7_ipInFQDN,D8_noOfSusWordInFQDN,D9_noOfSubLevelInFQDN,D10_lenOfBrandname,\
        D11_lenOfSubDomainInFQDN,D12_avgWordLengthInFQDN,D13_longWordLengthInFQDN,D14_LengthInDomain,D15_noOfHyphenInFQDN,D16_noOfUnderscoreInFQDN,D17_gTLDExistInFQDN,D18_entropyNANInFQDN,D19_brandInSubdomain,\
        P1_brandInPath,P2_noOfSubfolderInPath,P3_noOfAlnumCharInPath,P4_noOfSPInPath,P5_poiDistriOfAlphaInPath,P6_poiDistriOfNumInPath,P7_queryExistInQuery,P8_noOfDigitInPath,P9_avgWordLengthInPath,P10_longWordLengthInPath,\
        P11_noOfHTTPSandWWWInPath,P12_entropyNANInPath,U1_atExistInURL,U2_baseUrlLengthInURL,U3_noOfQuestionmarkInURL,U4_noOfSlashInURL,U5_ratioOfHyphenInURL, phishHintedWordInFQDN,phishHintedWordInPath

def extract_NLP_feat(df:pd.DataFrame,path:str):
    top_500 = pd.read_csv(path+'/top-1m.csv/top-500.csv',encoding='utf-8',sep=',')
    top_500 = set(top_500['URLs'].values)
    D1_noOfDotInDomain_,D2_noOfDotInHostname_,D3_noOfSPInDomain_,D4_noOfSPInHostname_,D5_noOfDigitInDomain_,D6_noOfAlphaCharInDomain_,D7_ipInDomain_,D8_noOfSusWordInDomain_,D9_noOfSubLevelInDomain_,D10_lenOfBrandname_,\
        D11_lenOfSubDomainInDomain_,D12_avgWordLengthInHostname_,D13_longWordLengthInHostname_,D14_hostLengthInHostname_,D15_noOfHyphenInHostname_,D16_noOfUnderscoreInHostname_,D17_gTLDExistInTLD_,D18_entropyNANInDomain_,D19_brandInSubdomain_,\
        P1_brandInPath_,P2_noOfSubfolderInPath_,P3_noOfAlnumCharInPath_,P4_noOfSPInPath_,P5_poiDistriOfAlphaInPath_,P6_poiDistriOfNumInPath_,P7_queryExistInQuery_,P8_noOfDigitInPath_,P9_avgWordLengthInPath_,P10_longWordLengthInPath_,\
        P11_noOfHTTPSandWWWInPath_,P12_entropyNANInPath_,U1_atExistInURL_,U2_baseUrlLengthInURL_,U3_noOfQuestionmarkInURL_,U4_noOfSlashInURL_,U5_ratioOfHyphenInURL_ = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        
    for indx,row in df.iterrows():
        url = str(row['URLs'])

        D1_noOfDotInDomain,D2_noOfDotInHostname,D3_noOfSPInDomain,D4_noOfSPInHostname,D5_noOfDigitInDomain,D6_noOfAlphaCharInDomain,D7_ipInDomain,D8_noOfSusWordInDomain,D9_noOfSubLevelInDomain,D10_lenOfBrandname,\
        D11_lenOfSubDomainInDomain,D12_avgWordLengthInHostname,D13_longWordLengthInHostname,D14_hostLengthInHostname,D15_noOfHyphenInHostname,D16_noOfUnderscoreInHostname,D17_gTLDExistInTLD,D18_entropyNANInDomain,D19_brandInSubdomain,\
        P1_brandInPath,P2_noOfSubfolderInPath,P3_noOfAlnumCharInPath,P4_noOfSPInPath,P5_poiDistriOfAlphaInPath,P6_poiDistriOfNumInPath,P7_queryExistInQuery,P8_noOfDigitInPath,P9_avgWordLengthInPath,P10_longWordLengthInPath,\
        P11_noOfHTTPSandWWWInPath,P12_entropyNANInPath,U1_atExistInURL,U2_baseUrlLengthInURL,U3_noOfQuestionmarkInURL,U4_noOfSlashInURL,U5_ratioOfHyphenInURL, phishHintedWordInFQDN,phishHintedWordInPath = func_extractor(url,top_500)

        D1_noOfDotInDomain_.append(D1_noOfDotInDomain)
        D2_noOfDotInHostname_.append(D2_noOfDotInHostname)
        D3_noOfSPInDomain_.append(D3_noOfSPInDomain)
        D4_noOfSPInHostname_.append(D4_noOfSPInHostname)
        D5_noOfDigitInDomain_.append(D5_noOfDigitInDomain)
        D6_noOfAlphaCharInDomain_.append(D6_noOfAlphaCharInDomain)
        D7_ipInDomain_.append(D7_ipInDomain)
        D8_noOfSusWordInDomain_.append(D8_noOfSusWordInDomain)
        D9_noOfSubLevelInDomain_.append(D9_noOfSubLevelInDomain)
        D10_lenOfBrandname_.append(D10_lenOfBrandname)
        D11_lenOfSubDomainInDomain_.append(D11_lenOfSubDomainInDomain)
        D12_avgWordLengthInHostname_.append(D12_avgWordLengthInHostname)
        D13_longWordLengthInHostname_.append(D13_longWordLengthInHostname)
        D14_hostLengthInHostname_.append(D14_hostLengthInHostname)
        D15_noOfHyphenInHostname_.append(D15_noOfHyphenInHostname)
        D16_noOfUnderscoreInHostname_.append(D16_noOfUnderscoreInHostname)
        D17_gTLDExistInTLD_.append(D17_gTLDExistInTLD)
        D18_entropyNANInDomain_.append(D18_entropyNANInDomain)
        D19_brandInSubdomain_.append(D19_brandInSubdomain)
        P1_brandInPath_.append(P1_brandInPath)
        P2_noOfSubfolderInPath_.append(P2_noOfSubfolderInPath)
        P3_noOfAlnumCharInPath_.append(P3_noOfAlnumCharInPath)
        P4_noOfSPInPath_.append(P4_noOfSPInPath)
        P5_poiDistriOfAlphaInPath_.append(P5_poiDistriOfAlphaInPath)
        P6_poiDistriOfNumInPath_.append(P6_poiDistriOfNumInPath)
        P7_queryExistInQuery_.append(P7_queryExistInQuery)
        P8_noOfDigitInPath_.append(P8_noOfDigitInPath)
        P9_avgWordLengthInPath_.append(P9_avgWordLengthInPath)
        P10_longWordLengthInPath_.append(P10_longWordLengthInPath)
        P11_noOfHTTPSandWWWInPath_.append(P11_noOfHTTPSandWWWInPath)
        P12_entropyNANInPath_.append(P12_entropyNANInPath)
        U1_atExistInURL_.append(U1_atExistInURL)
        U2_baseUrlLengthInURL_.append(U2_baseUrlLengthInURL)
        U3_noOfQuestionmarkInURL_.append(U3_noOfQuestionmarkInURL)
        U4_noOfSlashInURL_.append(U4_noOfSlashInURL)
        U5_ratioOfHyphenInURL_.append(U5_ratioOfHyphenInURL)

    nlp_features_df = pd.DataFrame(list(zip(D1_noOfDotInDomain_,D2_noOfDotInHostname_,D3_noOfSPInDomain_,D4_noOfSPInHostname_,D5_noOfDigitInDomain_,D6_noOfAlphaCharInDomain_,D7_ipInDomain_,D8_noOfSusWordInDomain_,D9_noOfSubLevelInDomain_,D10_lenOfBrandname_,\
        D11_lenOfSubDomainInDomain_,D12_avgWordLengthInHostname_,D13_longWordLengthInHostname_,D14_hostLengthInHostname_,D15_noOfHyphenInHostname_,D16_noOfUnderscoreInHostname_,D17_gTLDExistInTLD_,D18_entropyNANInDomain_,D19_brandInSubdomain_,\
        P1_brandInPath_,P2_noOfSubfolderInPath_,P3_noOfAlnumCharInPath_,P4_noOfSPInPath_,P5_poiDistriOfAlphaInPath_,P6_poiDistriOfNumInPath_,P7_queryExistInQuery_,P8_noOfDigitInPath_,P9_avgWordLengthInPath_,P10_longWordLengthInPath_,\
        P11_noOfHTTPSandWWWInPath_,P12_entropyNANInPath_,U1_atExistInURL_,U2_baseUrlLengthInURL_,U3_noOfQuestionmarkInURL_,U4_noOfSlashInURL_,U5_ratioOfHyphenInURL_)),\
        columns=['D1_noOfDotInDomain','D2_noOfDotInHostname','D3_noOfSPInDomain','D4_noOfSPInHostname','D5_noOfDigitInDomain','D6_noOfAlphaCharInDomain','D7_ipInDomain','D8_noOfSusWordInDomain','D9_noOfSubLevelInDomain','D10_lenOfBrandname',\
        'D11_lenOfSubDomainInDomain','D12_avgWordLengthInHostname','D13_longWordLengthInHostname','D14_hostLengthInHostname','D15_noOfHyphenInHostname','D16_noOfUnderscoreInHostname','D17_gTLDExistInTLD','D18_entropyNANInDomain','D19_brandInSubdomain',\
        'P1_brandInPath','P2_noOfSubfolderInPath','P3_noOfAlnumCharInPath','P4_noOfSPInPath','P5_poiDistriOfAlphaInPath','P6_poiDistriOfNumInPath','P7_queryExistInQuery','P8_noOfDigitInPath','P9_avgWordLengthInPath','P10_longWordLengthInPath',\
        'P11_noOfHTTPSInPath','P12_entropyNANInPath','U1_atExistInURL','U2_baseUrlLengthInURL','U3_noOfQuestionmarkInURL','U4_noOfSlashInURL','U5_ratioOfHyphenInURL'])
    
    return nlp_features_df
