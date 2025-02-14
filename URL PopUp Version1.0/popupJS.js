document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('shButton').addEventListener('click', validateURL);
});

function validateURL() {
    const fileReader = new FileReader();
    var url = "";
    var tmpurl = "";
    var singleInput = document.getElementById('singleUrlInput');
    var multiInput = document.getElementById('fileUrlInput');
    var expRadio = "0";

    var radioBtns = document.querySelectorAll("input[type='radio']");
    radioBtns.forEach(function (radioBtn) {
        if (radioBtn.checked) {
            var radioVal = radioBtn.value;
            if (radioVal == '1'){
                expRadio = radioVal
            }
            else if (radioVal == '0'){
                expRadio = radioVal
            }
        }
    })
    if (singleInput != null) {
        const urlPattern = /^(http:\/\/|https:\/\/|www\.)[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(\/.*)?$/;
        boolChkURL = urlPattern.test(singleInput.value);
        if (!boolChkURL){
            alert('Please Choose a valid URL Input.\nValid URL Format: [http|https|www].domain.top-level-domain[/path]');
            return; 
        }
        else {
            var raw_url = singleInput.value;
            url = encodeURIComponent(raw_url);
            sendRequest(url, raw_url, expRadio);
        }
    }
    else if (multiInput != null) {
        if (multiInput.files.length > 0) {
            tmpurl = multiInput.files[0];
            fileReader.readAsText(tmpurl);
            fileReader.onload = () => {
                var raw_url = fileReader.result;
                if (raw_url.endsWith('\n')){
                    raw_url = raw_url.substring(0,raw_url.length-1);
                }
                url = encodeURIComponent(raw_url);
                sendRequest(url, raw_url, expRadio);
            };

            fileReader.onerror = function () {
                alert(fileReader.error);
                return;
            };
        }
        else {
            alert('Please Upload a File');
            return;
        }
    }
    else {
        alert('Please Choose an option either URL Input or Upload a File');
        return;
    }
}

function sendRequest(url, raw_url, expOption) {

    var startTime = Date.now() / 1000;
    // Send the URL to the server for processing
    fetch('http://127.0.0.1:port/validate_url4PhiSN', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ urls: url, expOptions: expOption })
    })
        .then(response => response.json())
        .then(data => {
            if (data.predicted_res.length > 1) {
                var stopTime = Date.now() / 1000;
                var latency = stopTime - startTime;
                var urls = raw_url.split('\n');
                showResult(urls, data.predicted_res, data.ttp, data.prob, latency.toFixed(3));
            }
            else {
                var stopTime = Date.now() / 1000;
                var latency = stopTime - startTime;
                visitURL(raw_url, data.predicted_res, data.ttp, data.prob, latency.toFixed(3),data.explanation);
            }
        });
}
function visitURL(url, predicted_res, ttp, prob_attack, latency, explanation) {

    var inputURL;
    try { // Check if the input is a valid URL
        if (!url.startsWith("http") & !url.startsWith("www")) {
            inputURL = "https://www." + url;
        }
        else if (url.startsWith("www")) {
            inputURL = "https://" + url
        }
        else {
            inputURL = url;
        }

        let prob_ = parseFloat(prob_attack[0]);
        let prob_round = prob_.toFixed(2);

        //if (predicted_res[0] == 'Phishing') {
        if (prob_round >= 1) {
            showCustomDialog(prob_round, 'Phishing', ttp[0], url, latency, explanation)
            gotoConfirm(inputURL)
        }
        else {
            showCustomDialog(prob_round, 'Legitimate', ttp[0], url, latency, explanation)
            gotoConfirm(inputURL)
        }

    } catch (error) {
        console.error('Invalid URL:', url); // Handle invalid URL
        alert('Invalid URL. Please enter a valid URL.');
    }
}
function gotoConfirm(inputURL) {
    if (typeof chrome !== 'undefined') {

        document.getElementById('next').addEventListener('click', function () {
            chrome.tabs.update({ url: inputURL }); // Use the 'chrome' object
            window.close();
        });
    } else {
        // Handle the case where the 'chrome' object is not available
        console.error("Chrome API not available. This script is intended for Chrome extensions.");
    }

    document.getElementById('back').addEventListener('click', function () {

        var custDialog = document.getElementById("customDialog");
        custDialog.setAttribute("contenteditable", "false");
        custDialog.classList.add("disabled");
        custDialog.style.display = "none";

        var inputDiv = document.getElementById("inputDiv");
        inputDiv.setAttribute("contenteditable", "true");
        inputDiv.classList.remove("disabled");
        inputDiv.style.display = "block";
        window.location.href = chrome.runtime.getURL("home.html");
    });
}
function showCustomDialog(percentage, label, ttp, url, latency, explanation) {

    var inputDiv = document.getElementById("inputDiv");
    inputDiv.setAttribute("contenteditable", "false");
    inputDiv.classList.add("disabled");
    inputDiv.style.display = "none";

    // Display the custom dialog
    var customDialog = document.getElementById("customDialog");
    customDialog.style.display = "block";

    var messageBox = document.getElementById("messageBox");
    messageBox.style.width = "100%";
    messageBox.innerHTML = "<hr><b>WARNING!</b><br>You're visiting <b><small>" + url.substring(0, 40) + "...</small></b><br>This can be a <b><small>" + label + " website</small></b>!<hr>"+explanation;

    var ttpBox = document.getElementById("ttpBox");
    ttpBox.style.width = "100%";
    ttpBox.innerHTML = "<hr><b>Information</b><br>Time-to-process (TTP) is " + ttp + " second(s).<br>Latency is " + latency + " second(s).<hr>";
    ttpBox.style.display = "block";

    var gotoNext = document.getElementById("gotoNext");
    gotoNext.style.display = "block";

    // Simulate a severity task with a colored progress bar
    var progressBar = document.getElementById("severityProgressBar");

    var width = 0;
    var severityInterval = setInterval(function () {
        if (width >= percentage) {
            if (percentage <= 0) {
                progressBar.innerHTML = "Expecting Legitimate website.";
                progressBar.style.backgroundColor = "#808080"; // Grey
                messageBox.innerHTML = "<hr><b>Information!</b><br>You're visiting <b><small>" + url.substring(0, 40) + "...</small></b><br>This can be a <b><small>" + label + " website</small></b>.<hr>"+explanation;
            }
            else {

                // Stop incrementing but keep the progress bar displayed
                clearInterval(severityInterval);
            }
        } else {
            width++;
            progressBar.style.width = width + "%";

            // Change colors based on percentage ranges
            if ((width > percentage) && (percentage < 1)) {
                progressBar.innerHTML = "Expecting Legitimate website.";
                progressBar.style.backgroundColor = "#808080"; // Grey

            } else if (width < 25) {
                progressBar.innerHTML = "Phishing Probability [~" + width + "%]";
                progressBar.style.backgroundColor = "#949610"; // light yellow
            } else if (width < 50) {
                progressBar.innerHTML = "Phishing Probability [~" + width + "%]";
                progressBar.style.backgroundColor = "#ffff00"; // Yellow
            } else if (width < 75) {
                progressBar.innerHTML = "Phishing Probability [~" + width + "%]";
                progressBar.style.backgroundColor = "#ff9900"; // Orange
            } else if (width >= 75) {
                progressBar.innerHTML = "Phishing Probability [~" + width + "%]";
                progressBar.style.backgroundColor = "#ff3300"; // Red
            }
        }
    }, 10); // Adjust the interval based on your needs
}

function showResult(urls, predicted_res, ttp, prob_attack, latency) {
    // Open result.html in a new window or tab
    var resultPage = window.open("result.html", "_blank");

    // Check if the page is opened successfully
    if (resultPage) {

        // Set up the onload event to ensure the document is fully loaded
        resultPage.onload = function () {

            // Access the table in the existing HTML page
            var resultTable = resultPage.document.getElementById("resultTable");

            // Clear existing rows
            while (resultTable.rows.length > 0) {
                resultTable.deleteRow(0);
            }

            var dataRow = resultTable.insertRow(0);

            // Add cells with width style
            var urlCell = dataRow.insertCell(0);
            urlCell.innerText = "URL";
            urlCell.style.width = "300px"; // Set width as needed
            urlCell.style.fontWeight = "bold";

            /*var predictionCell = dataRow.insertCell(1);
            predictionCell.innerText = "Prediction";
            predictionCell.style.width = "150px"; // Set width as needed
            predictionCell.style.fontWeight = "bold";*/

            var ttpCell = dataRow.insertCell(1);
            ttpCell.innerText = "Detection Time (sec)";
            ttpCell.style.width = "150px"; // Set width as needed
            ttpCell.style.fontWeight = "bold";
            ttpCell.style.textAlign = "right";

            var probCell = dataRow.insertCell(2);
            probCell.innerText = "Potential Threat \n( >1% is not safe)";
            probCell.style.width = "150px";
            probCell.style.fontWeight = "bold";
            probCell.style.textAlign = "right";

            // Add the table content to the result page
            for (let i = 0; i < urls.length; i++) {
                dataRow = resultTable.insertRow(i + 1);

                //var predicted_label = 'Phishing';
                //if (prob_attack[i] < 1){
                //    predicted_label = 'Legitimate';
                //}

                // Add cells with width style
                urlCell = dataRow.insertCell(0);
                var link = document.createElement("a");
                link.setAttribute("href", urls[i])
                link.className = "someCSSclass";
                var linkText = document.createTextNode(urls[i].split('?')[0].substring(0, 100));
                link.appendChild(linkText);
                urlCell.appendChild(link);
                urlCell.style.width = "300px"; // Set width as needed

                /*
                predictionCell = dataRow.insertCell(1);
                predictionCell.innerText = predicted_label;
                predictionCell.style.width = "150px"; // Set width as needed
                */

                ttpCell = dataRow.insertCell(1);
                ttpCell.innerText = parseFloat(ttp[i]).toFixed(3);
                ttpCell.style.width = "150px"; // Set width as needed
                ttpCell.style.textAlign = "right";

                probCell = dataRow.insertCell(2);
                probCell.innerText = parseFloat(prob_attack[i]).toFixed(2) + '%';
                probCell.style.width = "150px";
                probCell.style.textAlign = "right";

            }
        };
    } else {
        alert("Unable to open the result page. Please check your browser settings.");
    }
}
