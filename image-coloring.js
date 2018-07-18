$(document).ready(function() {
    Chart.defaults.global.responsive = true;

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

    function shuffleArray(array) {
        for (var i = array.length - 1; i > 0; i--) {
            var j = Math.floor(Math.random() * (i + 1));
            var temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
        return array;
    }

    $.getJSON("ic-epochs.json", function(data) {
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

    imgs = ["badlands", "cliff", "corridor", "diner", "embankment", "kitchen", "office", "skyscraper", "staircase", "street", "tunnel", "utility_room"]
    imgs = shuffleArray(imgs)
    for (var x in imgs) {
        var card = $("<div />", {
            "class": "card text-center"
        });


        var header = $("<div />", {
            "class": "card-header"
        });
        $('<h4 class="card-title">' + imgs[x] + '</h4>').appendTo(header);
        var headerlist = $("<ul />", {
            "class": "nav nav-pills nav-fill card-header-tabs mb-1",
            id: imgs[x] + "Tab",
            role: "tablist"
        });
        $('<li class="nav-item"><a class="nav-link active" id="' + imgs[x] + '-input-pill" data-toggle="tab" href="#' + imgs[x] + '-input" role="pill" aria-controls="' + imgs[x] + '-input" aria-selected="true">Gray</a></li>').appendTo(headerlist);
        $('<li class="nav-item"><a class="nav-link" id="' + imgs[x] + '-map-pill" data-toggle="tab" href="#' + imgs[x] + '-map" role="pill" aria-controls="' + imgs[x] + '-map" aria-selected="false">Map</a></li>').appendTo(headerlist);
        $('<li class="nav-item"><a class="nav-link" id="' + imgs[x] + '-output-pill" data-toggle="tab" href="#' + imgs[x] + '-output" role="pill" aria-controls="' + imgs[x] + '-output" aria-selected="false">Color</a></li>').appendTo(headerlist);
        $('<li class="nav-item"><a class="nav-link" id="' + imgs[x] + '-pill" data-toggle="tab" href="#' + imgs[x] + '" role="pill" aria-controls="' + imgs[x] + '" aria-selected="false">Truth</a></li>').appendTo(headerlist);
        headerlist.appendTo(header);
        header.appendTo(card);


        var body = $("<div />", {
            "class": "card-body p-0"
        });
        var tabcontent = $("<div />", {
            "class": "tab-content",
            id: imgs[x] + "-tabContent"
        });
        $('<div class="tab-pane fade show active text-center" id="' + imgs[x] + '-input" role="tabpanel" aria-labelledby="' + imgs[x] + '-input-tab"><img src="docs/' + imgs[x] + '_input.png" class="img-fluid" alt="' + imgs[x] + ' Input"></div>').appendTo(tabcontent);
        $('<div class="tab-pane fade text-center" id="' + imgs[x] + '-map" role="tabpanel" aria-labelledby="' + imgs[x] + '-map-tab"><img src="docs/' + imgs[x] + '_map.png" class="img-fluid" alt="' + imgs[x] + ' Map"></div>').appendTo(tabcontent);
        $('<div class="tab-pane fade text-center" id="' + imgs[x] + '-output" role="tabpanel" aria-labelledby="' + imgs[x] + '-output-tab"><img src="docs/' + imgs[x] + '_output.png" class="img-fluid" alt="' + imgs[x] + ' Output"></div>').appendTo(tabcontent);
        $('<div class="tab-pane fade text-center" id="' + imgs[x] + '" role="tabpanel" aria-labelledby="' + imgs[x] + '-tab"><img src="docs/' + imgs[x] + '.jpg" class="img-fluid" alt="' + imgs[x] + '"></div>').appendTo(tabcontent);
        tabcontent.appendTo(body);
        body.appendTo(card);

        $('<ul class="list-group list-group-flush" id="' + imgs[x] + '-prediction"></ul>').appendTo(card);

        card.appendTo("#previewColumns");

        $('#' + imgs[x] + 'Tab a').on('click', function(e) {
            e.preventDefault();
            $(this).tab('show');
        });

        $('#' + imgs[x] + 'Tab li:first-child a').tab('show');
    }

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
