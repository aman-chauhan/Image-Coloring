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

    // imgs = ["badlands", "cliff", "corridor", "diner", "embankment", "kitchen", "office", "skyscraper", "staircase", "street", "tunnel", "utility_room"]
    preds = {}
    $.when($.ajax({
        url: "docs/badlands_prediction.json",
        async: true,
        dataType: "json",
        success: function(data) {
            preds["badlands"] = data;
        }
    }), $.ajax({
        url: "docs/cliff_prediction.json",
        async: true,
        dataType: "json",
        success: function(data) {
            preds["cliff"] = data;
        }
    }), $.ajax({
        url: "docs/corridor_prediction.json",
        async: true,
        dataType: "json",
        success: function(data) {
            preds["corridor"] = data;
        }
    }), $.ajax({
        url: "docs/diner_prediction.json",
        async: true,
        dataType: "json",
        success: function(data) {
            preds["diner"] = data;
        }
    }), $.ajax({
        url: "docs/embankment_prediction.json",
        async: true,
        dataType: "json",
        success: function(data) {
            preds["embankment"] = data;
        }
    }), $.ajax({
        url: "docs/kitchen_prediction.json",
        async: true,
        dataType: "json",
        success: function(data) {
            preds["kitchen"] = data;
        }
    }), $.ajax({
        url: "docs/office_prediction.json",
        async: true,
        dataType: "json",
        success: function(data) {
            preds["office"] = data;
        }
    }), $.ajax({
        url: "docs/skyscraper_prediction.json",
        async: true,
        dataType: "json",
        success: function(data) {
            preds["skyscraper"] = data;
        }
    }), $.ajax({
        url: "docs/staircase_prediction.json",
        async: true,
        dataType: "json",
        success: function(data) {
            preds["staircase"] = data;
        }
    }), $.ajax({
        url: "docs/street_prediction.json",
        async: true,
        dataType: "json",
        success: function(data) {
            preds["street"] = data;
        }
    }), $.ajax({
        url: "docs/tunnel_prediction.json",
        async: true,
        dataType: "json",
        success: function(data) {
            preds["tunnel"] = data;
        }
    }), $.ajax({
        url: "docs/utility_room_prediction.json",
        async: true,
        dataType: "json",
        success: function(data) {
            preds["utility_room"] = data;
        }
    })).then(function() {
        for (var x in preds) {
            $("#" + x + "-prediction").append('<li class="list-group-item text-center"><strong>Top 5 Scene Prediction</strong></li>');
            for (var y in preds[x]) {
                $("#" + x + "-prediction").append('<li class="list-group-item text-center">' + y + ' <span class="badge badge-primary">' + preds[x][y] + '</span>' + '</li>');
            }
        }
    });
});
