import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']= '-1'
import preloader

preloader.random.seed(42)
preloader.np.random.seed(42)

src_folder = "path-to-src-folder"
data_folder = src_folder + "path-to-data-folder"
result_folder = "path-to-result-folder"

# Load Trained-Vocab file
from_disk = preloader.pickle.load(open("{}urlVectorizer.pkl".format(src_folder),"rb"))
loaded_vectorized_layer = preloader.preprocessing.TextVectorization.from_config(from_disk["config"])

# Assign Trained-Weights
loaded_vectorized_layer.set_weights(from_disk["weights"])

pkl_reader = preloader.pd.read_pickle(f"{src_folder}urlVectorizer.pkl")['weights']
ids = pkl_reader[1]
words = pkl_reader[0]
word2id = preloader.pd.DataFrame(columns=['Word_name','IDs'])
word2id['Word_name'] = [bs.decode('utf-8') for bs in words]
word2id['IDs'] = ids
word2id = word2id.sort_values(by=['IDs'],ascending=True)

# Load Top500 Legitimate Domains
top_500 = preloader.pd.read_csv(src_folder+"top-500.csv",encoding="utf-8",sep=",")
top_500 = set(top_500['URLs'].values)

# Load Trained-Model [H-DNN-LST-DNN.h5] : a hybrid-DNN model from NLP-DNN and URL-LSTM
lstm_model = preloader.tf.keras.models.load_model(src_folder+"H-DNN-LST-DNN.h5",custom_objects={'TokenAndPositionEmbedding': preloader.embeddingModule.TokenAndPositionEmbedding})

################################################# Loading Training Dataset ##############################################
# Load the BERT tokenizer and model
#model_name = 'bert-base-uncased'
#tokenizer = preloader.BertTokenizer.from_pretrained(model_name)

# Set data size for SHAP
len2fitExp = 500

X_train_URL_data = preloader.pd.read_csv(data_folder+"X_train_.csv", encoding='utf-8', sep=',')#[:1000]
X_train_NLP_data = preloader.pd.read_csv(data_folder+"X_train_nlp_url_.csv", encoding='utf-8', sep=',')#[:1000]

nlp_feature_names =  ['D1_noOfDotInFQDN','D2_noOfDotInDomain','D3_noOfSPInFQDN','D4_noOfSPInDomain','D5_noOfDigitInFQDN','D6_noOfAlphaCharInFQDN','D7_ipInFQDN','D8_noOfSusWordInFQDN','D9_noOfSubLevelInFQDN','D10_lenOfBrandname',\
        'D11_lenOfSubDomainInFQDN','D12_avgWordLengthInFQDN','D13_longWordLengthInFQDN','D14_LengthInDomain','D15_noOfHyphenInFQDN','D16_noOfUnderscoreInFQDN','D17_gTLDExistInFQDN','D18_entropyNANInFQDN','D19_brandInSubdomain',\
        'P1_brandInPath','P2_noOfSubfolderInPath','P3_noOfAlnumCharInPath','P4_noOfSPInPath','P5_poiDistriOfAlphaInPath','P6_poiDistriOfNumInPath','P7_queryExistInQuery','P8_noOfDigitInPath','P9_avgWordLengthInPath','P10_longWordLengthInPath',\
        'P11_noOfHTTPSandWWWInPath','P12_entropyNANInPath','U1_atExistInURL','U2_baseUrlLengthInURL','U3_noOfQuestionmarkInURL','U4_noOfSlashInURL','U5_ratioOfHyphenInURL']

X_trian_reduced = preloader.pd.read_csv(data_folder+'sampledData.csv',encoding='utf-8',sep=',')
X_train_URL_Word = X_trian_reduced['URL_Word'].to_list()
y_train = X_trian_reduced['Labels']
X_train_NLP_data = X_trian_reduced.iloc[:,3:]
X_train_NLP_data.columns = nlp_feature_names

#X_sampled = X_trian_reduced
#X_sampled.to_csv(data_folder+'sampledData.csv',encoding='utf-8',index=False,sep=',')

def map_NLP_Features(features:list):
    nlp_feature_ = ['No. of Dot [.] in entire domain, i.e., Fully Qualified Domain Name (FQDN)', 'No. of Dot [.] in effective domain',
                         'No. of special characters in entire domain, i.e., Fully Qualified Domain Name (FQDN)', 'No. of special characters in effective domain',
                         'No. of digits in entire domain, i.e., Fully Qualified Domain Name (FQDN)', 'No. of alphabets in entire domain, i.e., Fully Qualified Domain Name (FQDN)',
                         'IP Address in domain','Suspicious word pattern, i.e., digit(alphabet)-alphabet(digit) in domain',
                         'No. of hierarchies in domain','Length of brandname','Length of sub-domain(s)','Average length of segmented words',
                         'Longest length of segmented words','Length of effective domain','Hyphen [-] in entire domain',
                         'Underscore [_] in domain','Generic Top-Level-Domain (gTLD)','Information on non-alphanumeric characters, i.e., symbols, in entire domain',
                         'Brandname in sub-domain','Brandname in path','No. of hierarchies in path','No. of alphanumeric characters, e.g., digits and alphabets, in path',
                         'No. of special characters in path','Distribution of alphabets in path','Distribution of digits in path','Query in path',
                         'No. of digits in path','Average length of segmented words in path','Longest length of segmented words in path',
                         'Embedded Link in path','Information on non-alphanumeric characters, i.e., symbols, in path',
                         'At [@] embedded in entire URL','Length of entire URL','Question mark [?] in entire URL','Slash [/] in entire URL',
                         'Hyphen[-] to entire URL Ratio']
    mapped_nlp_features = list(map(lambda x, y : y, features,nlp_feature_))
    return mapped_nlp_features

################################################# Vectorizing Training Dataset ##############################################

#print('Vectorizing Training Data: URL.....')
X_train_start_time_url4url = preloader.time()
X_train_vectorized_url = loaded_vectorized_layer(preloader.np.array(X_train_URL_Word)).numpy()
X_train_stop_time_url4url = preloader.time()
X_train_time_url = X_train_stop_time_url4url - X_train_start_time_url4url
print(f'Vectorizing Training Data: URL.....Finished. Took {preloader.np.round(X_train_time_url, 3)} sec.')
X_train_vectorized_url = preloader.tf.convert_to_tensor(X_train_vectorized_url,dtype=preloader.tf.float32).numpy()

#print('Preprocessing for Training NLP Features >>>  \n')
X_train_start_time_nlp4url = preloader.time()
X_train_nlp_features4URL = preloader.tf.convert_to_tensor(X_train_NLP_data,dtype=preloader.tf.float32).numpy()
X_train_stop_time_nlp4url = preloader.time()
X_train_time_nlp = X_train_stop_time_nlp4url - X_train_start_time_nlp4url
print(f'Extracting Training Data: URL.....Finished. Took {preloader.np.round(X_train_time_nlp, 3)} sec.')

################################# Compute SHAP #############################

def u_lstm(model:preloader.tf.keras.Model): 
    ### Hybrid Parameters >>>
    custmetric=['accuracy', preloader.metrics.Precision(name='precision'), preloader.metrics.Recall(name='recall'), preloader.metrics.FalsePositives(name='false_positives'), preloader.metrics.TruePositives(name='true_positives'), preloader.metrics.FalseNegatives(name='false_negatives'), preloader.metrics.TrueNegatives(name='true_negatives')]
    
    ### URL Parameters >>>
    url_lstm_loss='binary_crossentropy'
    url_lstm_optimizer='Adamax'
    url_lstm_learning_rate=0.001

    url_layers = [0,1,2,3,5,7,9,11,13,21]
    s_lstm_model = preloader.tf.keras.Sequential()
    for i, layer in enumerate(model.layers):
        if i in url_layers:
            s_lstm_model.add(layer)
    ###### Initializing Word-LSTM Model Optimizer #################
    optimizer_class = getattr(preloader.importlib.import_module(
        'tensorflow.keras.optimizers'), url_lstm_optimizer)
    custoptimizer = optimizer_class(learning_rate=url_lstm_learning_rate)

    ###### Compiling Word-LSTM Model Optimizer #################
    s_lstm_model.compile(loss={'url_lstm_outputs': url_lstm_loss}, optimizer=custoptimizer, metrics=custmetric)
    return s_lstm_model

def n_dnn(model:preloader.tf.keras.Model):    
    ### Hybrid Parameters >>>
    custmetric=['accuracy', preloader.metrics.Precision(name='precision'), preloader.metrics.Recall(name='recall'), preloader.metrics.FalsePositives(name='false_positives'), preloader.metrics.TruePositives(name='true_positives'), preloader.metrics.FalseNegatives(name='false_negatives'), preloader.metrics.TrueNegatives(name='true_negatives')]
    
    ### NLP Parameters >>>
    nlp_dnn_loss='binary_crossentropy'
    nlp_dnn_optimizer='Adam'
    nlp_dnn_learning_rate=0.001

    nlp_layers = [4,6,8,10,12,20]
    n_dnn_model = preloader.tf.keras.Sequential()
    for i, layer in enumerate(model.layers):
        if i in nlp_layers:
            n_dnn_model.add(layer)
    ###### Initializing Word-LSTM Model Optimizer #################
    optimizer_class = getattr(preloader.importlib.import_module(
        'tensorflow.keras.optimizers'), nlp_dnn_optimizer)
    custoptimizer = optimizer_class(learning_rate=nlp_dnn_learning_rate)

    ###### Compiling Word-LSTM Model Optimizer #################
    n_dnn_model.compile(loss={'nlp_dnn_outputs': nlp_dnn_loss}, optimizer=custoptimizer, metrics=custmetric)
    return n_dnn_model            

def retreive_explaination(predict_method:str,shap_values,data:preloader.np.array,feature_names:list,path:str,t_exp:int):

    # Visualize the SHAP values for the first sample
    #shap.initjs()
    exp = preloader.shap.Explanation(values=shap_values[0],base_values=shap_values.base_values[0],data=data,feature_names=feature_names)
    preloader.shap.plots.waterfall(exp,show = False)
    preloader.plt.tight_layout()
    preloader.plt.savefig('{}shapwaterfall_{}_{}.png'.format(path,predict_method,t_exp))
    preloader.plt.close()
    wf_exp_img = '{}shapwaterfall_{}_{}.png'.format(path,predict_method,t_exp)

    preloader.shap.summary_plot(shap_values,feature_names = feature_names,features=data,show = False) #max_display = word_count,
    preloader.plt.tight_layout()
    preloader.plt.savefig('{}shapSummary_{}_{}.png'.format(path,predict_method,t_exp))
    preloader.plt.close()
    sm_exp_img = '{}shapSummary_{}_{}.png'.format(path,predict_method,t_exp)

    return wf_exp_img, sm_exp_img

def retrieve_s_features(data:list,mode:str,t_exp:int):
    word_seq, id_seq = [], []
    word_seq_org = []
    pos = 0
    for url_i in data:
        for i in url_i:
            if int(i) in word2id['IDs'].to_list():
                idx = word2id['IDs'].to_list().index(i)
                act_word = str(word2id['Word_name'].to_list()[idx]) + ' @pos=' + str(pos)
                org_word = str(word2id['Word_name'].to_list()[idx])
                if idx == 0:
                    act_word = '[PAD]@pos=' + str(pos)
                    org_word = '[PAD]'
                word_seq.append(act_word)
                id_seq.append(idx)
                word_seq_org.append(org_word)
            pos += 1
    s_feature_names = preloader.pd.DataFrame(columns=['word','id','org_word'])
    s_feature_names['word'] = word_seq
    s_feature_names['id'] = id_seq
    s_feature_names['org_word'] = word_seq_org
    s_feature_names = s_feature_names.drop_duplicates(inplace=False)
    s_feature_names.to_csv('{}{}_s_feature_names_{}.csv'.format(data_folder,mode,t_exp),encoding='utf-8',index=False)
    return s_feature_names

def train_shap(base_model:preloader.tf.keras.Model,mode:str,t_exp:int):

    ################### Computing S-LSTM SHAP ######################
    X_train_s_feature_names = retrieve_s_features(list(X_train_vectorized_url[:,:]),'train',0)['word']
    s_lstm_model = u_lstm(base_model)
    url_explainer = preloader.shap.Explainer(s_lstm_model.predict,preloader.np.array(X_train_vectorized_url[:,:]))
    url_train_shap_values = url_explainer(preloader.np.array(X_train_vectorized_url[:,:]))
    url_wf_exp_img, url_sm_exp_img = retreive_explaination('url'+mode,url_train_shap_values,preloader.np.array(X_train_vectorized_url[:,:]),X_train_s_feature_names,data_folder,t_exp)
    print('SHAP Explanation >>> URL Finished')

    ################### Computing N-DNN SHAP ######################
    n_dnn_model = n_dnn(base_model)
    nlp_explainer = preloader.shap.Explainer(n_dnn_model.predict,preloader.np.array(X_train_nlp_features4URL[:,:]))
    nlp_train_shap_values = nlp_explainer(preloader.np.array(X_train_nlp_features4URL[:,:]))
    nlp_wf_exp_img, nlp_sm_exp_img = retreive_explaination('nlp'+mode,nlp_train_shap_values,preloader.np.array(X_train_nlp_features4URL[:,:]),nlp_feature_names,data_folder,t_exp)
    print('SHAP Explanation >>> NLP Finished')

    return url_explainer, nlp_explainer, s_lstm_model, n_dnn_model

### Computing SHAP ####
url_explainer, nlp_explainer, s_lstm_model, n_dnn_model = train_shap(lstm_model,'_train',0)

def fit_shap(url_explainer:preloader.shap.Explainer, nlp_explainer:preloader.shap.Explainer, X_test_vectorized_url:preloader.tf.convert_to_tensor, X_test_nlp_features4URL:preloader.tf.convert_to_tensor, predicted_result:str, predicted_url_result:str, predicted_nlp_result:str,h_predicted:float,s_predicted:float,n_predicted:float):
    
    t_exp = preloader.time()
    effected_url_csv, effected_nlp_csv = preloader.pd.DataFrame(), preloader.pd.DataFrame()
    url_wf_exp_img, url_sm_exp_img, nlp_wf_exp_img, nlp_sm_exp_img, expStr = '', '', '', '', ''
    pad_idx = list(preloader.np.array(X_test_vectorized_url[0])).index(0)
    
    s_feature = retrieve_s_features(list(preloader.np.array(X_test_vectorized_url)[:,:pad_idx]),'test',t_exp)
    s_feature_names = s_feature['word']
    url_test_shap_values = url_explainer(preloader.np.array(X_test_vectorized_url))[:,:pad_idx]
    X_test_vectorized_url = preloader.np.array(X_test_vectorized_url)[:,:pad_idx]
    shap_value_instance = url_test_shap_values.values[0][:pad_idx]
    url_exp_result = shap_scorer(shap_value_instance,list(X_test_vectorized_url[0]),s_feature['org_word'],data_folder,'url_test',t_exp,False)

    n_feature_names = map_NLP_Features(nlp_feature_names)
    nlp_test_shap_values = nlp_explainer(preloader.np.array(X_test_nlp_features4URL))
    
    features4URL = [preloader.np.round(float(i),2) for i in list(X_test_nlp_features4URL[0].numpy())]
    nlp_exp_result = shap_scorer(nlp_test_shap_values.values[0],features4URL,n_feature_names,data_folder,'nlp_test',t_exp,True)
    
    url_wf_exp_img, url_sm_exp_img = retreive_explaination('url_test',url_test_shap_values,preloader.np.array(X_test_vectorized_url),s_feature_names,data_folder,t_exp)
    nlp_wf_exp_img, nlp_sm_exp_img = retreive_explaination('nlp_test', nlp_test_shap_values,preloader.np.array(X_test_nlp_features4URL),nlp_feature_names,data_folder,t_exp)
    
    effected_url_csv, effected_nlp_csv, expStr = explain_result(url_exp_result,nlp_exp_result,predicted_result,predicted_url_result,predicted_nlp_result,h_predicted,s_predicted,n_predicted)
    
    return url_wf_exp_img, url_sm_exp_img, url_exp_result, nlp_wf_exp_img, nlp_sm_exp_img, nlp_exp_result, effected_url_csv, effected_nlp_csv, expStr

def shap_scorer(shap_values,features4URL:list,feature_names:list,path:str,predict_method:str,t_exp:int,sort_:bool):

    shap_values = [preloader.np.round(value*100,1) for value in shap_values]
    shap_df = preloader.pd.DataFrame({'Feature Name':feature_names, 'Feature':features4URL, 'Shap Value':shap_values, 'Position':[i for i in range(len(feature_names))]})
    if sort_:
        shap_df = shap_df.sort_values(by='Shap Value',ascending=False)
    shap_df.to_csv('{}shap_scores_{}_{}.csv'.format(path,predict_method,t_exp),sep=',',encoding='utf-8',index=False)

    return shap_df

#################################################################################
def model_evaluation(model,_nlp_features4URL,_vectorized_url):
    model_start_time = preloader.time() # Record start-time for Prediction
    predicted_nlp_,predicted_url_,predicted_hybrid=model.predict(x=[_nlp_features4URL,_vectorized_url],verbose=0)
    model_end_time = preloader.time() # Record stop-time for Prediction
    time2test_url= model_end_time-model_start_time
    return time2test_url,predicted_nlp_,predicted_url_,predicted_hybrid

def decodeURLs(url_input:str):
    urls = url_input.split('%0D%0A') # Split Input URL by NEWLINE
    decoded_urls = []
    for url in urls:
        url = preloader.unquote_plus(url,encoding='utf-8') # Decode URL
        if url.endswith('/'): # Remove '/' at the end of URL
            url = url[:-1]
        print('URL : ',url)
        decoded_urls.append(url)
    return decoded_urls

def explain_result(url_csv:preloader.pd.DataFrame, nlp_csv:preloader.pd.DataFrame, predicted_result:str,predicted_url_result:str,predicted_nlp_result:str,h_predicted:float,s_predicted:float,n_predicted:float):
    effected_url_features = preloader.pd.DataFrame(columns=['Feature Name', 'Feature', 'Shap Value'])
    effected_nlp_features = preloader.pd.DataFrame(columns=['Feature Name', 'Feature', 'Shap Value'])
    url_sorted = url_csv.sort_values(by='Shap Value',ascending=False)
    expStr = "<hr>"
    
    #if predicted_result == 'Phishing':
    #    expStr = "Hybrid model predicted as <b>Phishing</b> w/ "+str(preloader.np.round(h_predicted,2))+" percentage.<hr><hr>"
    #elif predicted_result == 'Legitimate':
    #    expStr = "Hybrid model predicted as <b>Legitimate</b> w/ "+str(preloader.np.round(h_predicted*100,2))+" percentage.<hr><hr>"
    
    if (predicted_url_result == 'Phishing') and (predicted_result == 'Phishing'):
        effected_url_features = url_sorted[:5]
        #expURLStr = list(map(lambda x, y : x + " w/ score " + str(y), effected_url_features['Feature Name'],effected_url_features['Shap Value']))
        #expURLStr = [word for word in effected_url_features['Feature Name']]
        expURLStr = []
        for idx,row in url_csv.iterrows():
            if str(row['Feature Name']).startswith('##'):
                url_csv.loc[idx,'Feature Name'] = str(row['Feature Name'])[2:]

        for idx,row in url_csv.iterrows():
            if row['Position'] in list(effected_url_features['Position']):
                expURLStr.append('<b>'+row['Feature Name']+'</b>')
            else:
                expURLStr.append(row['Feature Name'])

        #expStr += "Segmentation model predicted as <b>Phishing</b> w/ "+str(preloader.np.round(s_predicted,2))+" percentage.<br>"+\
        expStr += "Suspicious Words [BOLD] are: <br>"+"".join(expURLStr)+"<hr>"
    
    if (predicted_nlp_result == 'Phishing') and (predicted_result == 'Phishing'):
        effected_nlp_features = nlp_csv[:3]
        feat_map = []
        for feat in effected_nlp_features['Feature'].to_list():
            if feat == 0.0:
                feat_map.append('zero')
            else:
                feat_map.append(feat)
        effected_nlp_features.loc[:,'Feature'] = feat_map
        expNLPStr = list(map(lambda x, y : x + " [ == " + str(y)+"] seems suspicious.", effected_nlp_features['Feature Name'],effected_nlp_features['Feature']))
        #expNLPStr = [feat for feat in effected_nlp_features['Feature Name']]
        #expStr += "NLP model predicted as <b>Phishing</b> w/ "+str(preloader.np.round(n_predicted,2))+" percentage.<br>"+\
        expStr += "<hr>Suspicious Features are: <br>"+"<br>".join(expNLPStr)+"<hr><hr>"
    
    return effected_url_features, effected_nlp_features, expStr

def detectorModule(url_input:str):
    
    urls = decodeURLs(url_input)
    TTP_, predicted_result_, CONFD_, prob_  = [], [], [], []
    nlpCONFD_, urlCONFD_ = [], []
    expFlag = 0
    head = 0
    tail = 0

    try:
        for url in urls:
            # Segment URL by SegURLizer
            url, segURL, segURL_latency, recursion_latency = preloader.SegURLizer.SegURLizer(url=url)
            start_time_url4url = preloader.time() # Record start-time for Segmentation
            # Vectorize Input URL
            _vectorized_url = loaded_vectorized_layer(preloader.np.array([segURL])).numpy().tolist()[0]
            stop_time_url4url = preloader.time() # Record stop-time for Segmentation
            _time_url = stop_time_url4url-start_time_url4url
            _vectorized_url= preloader.tf.convert_to_tensor([_vectorized_url])

            start_time_nlp4url = preloader.time() # Record start-time for NLP feature Extraction
            _nlp_features4URL = preloader.tf.convert_to_tensor([preloader.NLPextractor.func_extractor(url,top_500)[:36]])
            stop_time_nlp4url = preloader.time() # Record stop-time for NLP feature Extraction
            _time_nlp = stop_time_nlp4url-start_time_nlp4url
        
            time2test_url,predicted_nlp_,predicted_url_,predicted_hybrid = model_evaluation(lstm_model,_nlp_features4URL,_vectorized_url)

            h_predicted = predicted_hybrid[0][0]
            s_predicted = predicted_url_[0][0]
            n_predicted = predicted_nlp_[0][0]
            predicted_result = "Phishing" if h_predicted >= 0.5 else "Legitimate"
            #predicted_url_result = "Phishing" if s_predicted >= 0.5 else "Legitimate"
            #predicted_nlp_result = "Phishing" if n_predicted >= 0.5 else "Legitimate"

            if predicted_result == "Phishing":
                expFlag = 1

            TTP = preloader.np.round(segURL_latency+_time_url+_time_nlp+time2test_url,3)
            TTP_.append(TTP)
            predicted_result_.append(predicted_result)
            CONFD = str(preloader.np.round(predicted_hybrid[0][0]*100,2))
            CONFD_.append(CONFD)
            nlpCONFD = str(preloader.np.round(predicted_nlp_[0][0]*100,2))
            nlpCONFD_.append(nlpCONFD)
            urlCONFD = str(preloader.np.round(predicted_url_[0][0]*100,2))
            urlCONFD_.append(urlCONFD)

            # Write Prediction Result to a .CSV file
            tail=tail+1
            if tail%1000==0 or tail==len(urls):
                now = preloader.datetime.now()
                reqTime = now.strftime("%y%m%d%H%M%S") # Record time of writing file

                result_df = preloader.pd.DataFrame()
                result_df['URL'] = urls[head:tail]
                result_df['Prediction'] = predicted_result_[head:tail]
                result_df['TTP'] = TTP_[head:tail]
                result_df['Confidence'] = CONFD_[head:tail]
                result_df['NLP Confidence'] = nlpCONFD_[head:tail]
                result_df['URL Confidence'] = urlCONFD_[head:tail]
                result_df.to_csv(result_folder+'predicted_result_{}.csv'.format(reqTime), encoding='utf-8', sep=',', index=False)
                head= tail
    except Exception as e:
        return urls, predicted_result_, CONFD_, urlCONFD_, nlpCONFD_, TTP_, expFlag
    return urls, predicted_result_, CONFD_, urlCONFD_, nlpCONFD_, TTP_, expFlag

def explainerModule(urls,h_predicted_score,s_predicted_score,n_predicted_score):
    expStr = ''
    for i in range(len(urls)):
        url = urls[i]
        h_prediction = "Phishing" if float(h_predicted_score[i]) >= 50.0 else "Legitimate"
        s_prediction = "Phishing" if float(s_predicted_score[i]) >= 50.0 else "Legitimate"
        n_prediction = "Phishing" if float(n_predicted_score[i]) >= 50.0 else "Legitimate"
        
        url, segURL, segURL_latency, recursion_latency = preloader.SegURLizer.SegURLizer(url=url)
        vectorized4URL = preloader.tf.convert_to_tensor([loaded_vectorized_layer(preloader.np.array([segURL])).numpy().tolist()[0]])
        nlp4URL = preloader.tf.convert_to_tensor([preloader.NLPextractor.func_extractor(url,top_500)[:36]])

        with preloader.concurrent.futures.ThreadPoolExecutor() as executor:
            future_1 = executor.submit(fit_shap,url_explainer,nlp_explainer,vectorized4URL,nlp4URL,h_prediction,s_prediction,n_prediction,float(h_predicted_score[i]),float(s_predicted_score[i]),float(n_predicted_score[i]))
            futures_1 = [future_1]
            for future in preloader.concurrent.futures.as_completed(futures_1):
                try:
                    url_wf_exp_img, url_sm_exp_img, url_exp_result, nlp_wf_exp_img, nlp_sm_exp_img, nlp_exp_result, effected_url_csv, effected_nlp_csv, expStr = future.result()
                except Exception as e:
                    print(f'Fit_shap raised as exception: {e}')
    
    return expStr
