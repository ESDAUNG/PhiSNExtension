# PhiSNExtension
This is an implementation of "Explainable Phishing URL Detection: Browser Extension," published at コンピュータセキュリティシンポジウム2024論文集
## 
## Abstract
A phishing attack is a fraudulent mechanism deployed by malicious actors, impersonating credential authorities to acquire confidential information. A phishing URL is a URL-based intrusion that lures users into exposing personal data. Although prior research focuses on URL-based detection, the explainability of prediction falls behind compared to other areas, such as natural language processing (NLP) and recommendation systems. Moreover, surprisingly, phishing detection systems cannot enhance user awareness and remains a mysterious black-box system. Therefore, not only is phishing URL detection crucial, but the explainability of prediction is also indispensable to leverage user awareness. To the best of our knowledge, a space exists for explainable prediction with a browser plugin for better user awareness. Therefore, this study emphasizes (i) a Chrome browser extension and (ii) explainable phishing detection for better understanding.

## 
## Structure of Folders
### - Frontend (Client-side, URL PopUp Version1.0) 
Development is implemented using Javascript, HTML and CSS (excuted in browser).
### - Backend (Server-side, PhiSN-Server) 
Development is implemented using Python (WSGI-based, using Flask).
PhiSN-Server/src-folder for models and implementations. 

PhiSN-Server/data-folder/sampleData.csv for SHAP explanation. 

Note: training data cannot be provided due to its size.
## 
### Cite as 
@inproceedings{

 author = {Aung, Eint Sandi and Matsumoto, Tsuneo and Kido, Fuyuko and Yamana, Hayato},
 
 book = {コンピュータセキュリティシンポジウム2024論文集},
 
 month = {Oct},

 pages = {1064--1071},
 
 publisher = {情報処理学会},
 
 title = {Explainable Phishing URL Detection: Browser Extension},
 
 year = {2024}
 
}
##
