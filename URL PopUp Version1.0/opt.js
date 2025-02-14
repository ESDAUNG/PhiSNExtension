document.addEventListener('DOMContentLoaded', function () {
    var radioBtns = document.querySelectorAll("input[type='radio']");
    radioBtns.forEach(function (radioBtn) {
        if (radioBtn.checked) {
            var radioVal = radioBtn.value;
            if (radioVal == 'singleOption') {
                document.getElementById("dynForm").innerHTML = 
                '<input type="text" id="singleUrlInput"><br>\
                <input type="radio" name="expOption" id="expTrue" value="1" checked> Enable explanation\
                <input type="radio" name="expOption" id="expFalse" value="0"> Disable explanation<br>\
                <label for="trueInfo" id="exptxt" contenteditable="true">*Note: Explanation will take a few seconds.</label>\
                <form id="inputForm">\
                </form>';
            }
            else if (radioVal == 'groupOption') {
                document.getElementById("dynForm").innerHTML = '<input type="file" id="fileUrlInput">';
            }
        }
    })
});

var radioBtns = document.querySelectorAll("input[type='radio']");
var grpLabel = document.getElementById("grptxt");
radioBtns.forEach(function (radioBtn) {
    radioBtn.addEventListener("change", function () {
        if (radioBtn.checked) {
            var radioVal = radioBtn.value;
            if (radioVal == 'singleOption') {

                grpLabel.setAttribute("contenteditable", "true");
                grpLabel.classList.add("disabled");
                grpLabel.style.display = "none";
                document.getElementById("dynForm").innerHTML =  
                '<input type="text" id="singleUrlInput"><br>\
                <input type="radio" name="expOption" id="expTrue" value="expTrueOption" checked> Enable explanation\
                <input type="radio" name="expOption" id="expFalse" value="expFalseOption"> Disable explanation<br>\
                <label for="trueInfo" id="exptxt" contenteditable="true">*Note: Explanation will take a few seconds.</label>\
                <form id="inputForm">\
                </form>';
            }
            else if (radioVal == 'groupOption') {

                grpLabel.setAttribute("contenteditable", "true");
                grpLabel.classList.remove("disabled");
                grpLabel.style.display = "block";
                document.getElementById("dynForm").innerHTML = '<input type="file" id="fileUrlInput">';
            }
        }
    });
});
