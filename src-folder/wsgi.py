from flask import Flask,request 
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
import logging
import phishDetector

app = Flask(__name__)

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1)

# Applying security-related configurations
app.config['CONTENT_SECURITY_POLICY'] = "default-src 'self'"
app.config['X_CONTENT_TYPE_OPTIONS'] = 'nosniff'
app.config['X_FRAME_OPTIONS'] = 'DENY' # Prevent iFrame
app.config['X_XSS_PROTECTION'] = 1 # Prevent cross side scripting
app.config['REFERRER_POLICY'] = 'strict-origin-when-cross-origin'

cors1 = CORS(app, resources={r"/validate_url4PhiSN": {"origins": "*"}})
cors2 = CORS(app, resources={r"/explain_url4PhiSN": {"origins": "*"}})
cors2 = CORS(app, resources={r"/valAndExp_url4PhiSN": {"origins": "*"}})

@app.route('/validate_url4PhiSN', methods=['POST','OPTIONS'])
def validate_url4PhiSN():
    app.logger.info('Accessed from {request.remote_add}')
    if request.method == 'OPTIONS':
        # Handle preflight OPTIONS request
        logging.info('Received OPTION Request')
        return '', 200
    
    try:
        data = request.json
        url = data['urls']

        logging.info(f'Received POST Request with URL(s): {url}')

        # Predict Phising / Legitimate
        urls, predicted_result, prtProb , urlCONFD_, nlpCONFD_, TTP, expFlag = phishDetector.detectorModule(url_input=url)

        # Prepare the response
        response_data = {'urls':urls, 'predicted_res': predicted_result, 'prob': prtProb, 's_prob': urlCONFD_, 'n_prob': nlpCONFD_, 'ttp': TTP, 'explanation': expFlag, 'errorMessage':''}
        logging.info('Processed POST Request and generated Response data.')
   
        return response_data
    
    except Exception as e:
        print(f'An internal error occurred{str(e)}')

        app.logger.error(f'An internal error occurred {str(e)} on the server.')
        response_data = {'errorMessage':'Oops! Internal Server Error. Please try again.'}
        return response_data

@app.route('/explain_url4PhiSN', methods=['POST','OPTIONS'])
def explain_url4PhiSN():
    app.logger.info('Accessed from {request.remote_add}')
    if request.method == 'OPTIONS':
        logging.info('Received OPTION Request')
        return '',200
    data = request.json

    url = data['urls']

    h_predicted_score = data['h_predicted_score']
    s_predicted_score = data['s_predicted_score']
    n_predicted_score = data['n_predicted_score']

    expStr = phishDetector.explainerModule(url,h_predicted_score,s_predicted_score,n_predicted_score)

    response_data = {'explanation_result': expStr}
    logging.info('Processed POST Request and generated Response data for explanation.')

    return response_data


@app.route('/valAndExp_url4PhiSN', methods=['POST','OPTIONS'])
def valAndExp_url4PhiSN():
    app.logger.info('Accessed from {request.remote_add}')
    if request.method == 'OPTIONS':
        # Handle preflight OPTIONS request
        logging.info('Received OPTION Request')
        return '', 200

    data = request.json
    url = data['urls']

    logging.info(f'Received POST Request with URL(s): {url}')

    # Predict Phising / Legitimate
    urls, predicted_result, prtProb , urlCONFD_, nlpCONFD_, TTP, expFlag = phishDetector.detectorModule(url_input=url)
    expStr = phishDetector.explainerModule(urls,prtProb,urlCONFD_,nlpCONFD_)

    response_data = {'explanation_result': expStr}
    logging.info('Processed POST Request and generated Response data for explanation.')
   
    return response_data