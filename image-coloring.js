$(document).ready(function() {
    function populate(l, data) {
        for (var x in data) {
            for (var y in data[x]) {
                if (l.hasOwnProperty(y)) {
                    l[y].push(data[x][y]);
                }
            }
        }
        return l;
    }

    $.getJSON("micro-epochs.json", function(data) {
        var cnt = Object.keys(data).length;
        $("#epoch-count").html(cnt.toString() + "<small>/1000</small>");
        var l = {
            "val_color_model_acc": [],
            "val_clf_model_acc": [],
            "color_model_acc": [],
            "clf_model_acc": []
        }
        l = populate(l, data);
        labels = []
        for (var x = 1; x <= cnt; x++) {
            labels.push(x);
        }
        var trainCTX = $("#trainingChart");
        var trainData = {
            labels: labels,
            datasets: [{
                    label: "Classification Accuracy",
                    backgroundColor: "#FF980066",
                    borderColor: "#FF9800",
                    borderWidth: 2,
                    pointBorderWidth: 1,
                    fill: "start",
                    data: l["clf_model_acc"]
                },
                {
                    label: "Coloring Accuracy",
                    backgroundColor: "#2196F366",
                    borderColor: "#2196F3",
                    borderWidth: 2,
                    pointBorderWidth: 1,
                    fill: "end",
                    data: l["color_model_acc"]
                }
            ]
        }
        var options = {
            scales: {
                yAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: "Accuracy"
                    }
                }],
                xAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: "Epochs"
                    }
                }]
            }
        }
        var trainChart = new Chart(trainCTX, {
            type: "line",
            data: trainData,
            options: options
        });

        var validationCTX = $("#validationChart");
        var validData = {
            labels: labels,
            datasets: [{
                    label: "Classification Accuracy",
                    backgroundColor: "#FF980066",
                    borderColor: "#FF9800",
                    borderWidth: 2,
                    pointBorderWidth: 1,
                    fill: "start",
                    data: l["val_clf_model_acc"]
                },
                {
                    label: "Coloring Accuracy",
                    backgroundColor: "#2196F366",
                    borderColor: "#2196F3",
                    borderWidth: 2,
                    pointBorderWidth: 1,
                    fill: "end",
                    data: l["val_color_model_acc"]
                }
            ]
        }
        var validationChart = new Chart(validationCTX, {
            type: "line",
            data: validData,
            options: options
        });
    });

    $("#main-nav a").on('click', function(event) {
        if (this.hash !== "") {
            event.preventDefault();
            var hash = this.hash;
            $('html, body').animate({
                scrollTop: $(hash).offset().top
            }, 1000, function() {
                window.location.hash = hash;
            });
        }
    });
});
