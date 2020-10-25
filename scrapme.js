var g = require('google-play-scraper').memoized();
var arrg = []; //list of appids
var reviews = []; //arr of review-score pairs
var id_rev = []; //dict of appID - review pair

function pushReview(xys) {
    xys.forEach(xy => {
        if(xy != undefined && xy != null && xy.text != null && xy.score != null) {
            reviews.push([
                xy.text, 0.0+xy.score+0.0
            ]);
        }
    });
    console.log(reviews.length);
}

function getReviewsForApps() {
    arrg.forEach(e => {
        g.reviews({
            sort: g.sort.NEWEST,
            appId: e
        }).then(v => pushReview(v), console.log);
    });
    saveToCSV();
}

function saveToCSV() {
    var stringify = require('csv-stringify');
    var fs = require('fs');
    stringify(reviews, function(err, output) {
        fs.writeFile('financeapps.csv', output, 'utf8', function(err) {
          if (err) {
            console.log('Some error occured - file either not saved or corrupted file saved.');
          } else {
            console.log('It\'s saved!');
          }
        });
      });
}

g.list({
        category: g.category.FINANCE,
        collection: g.collection.TOP_FREE,
        num: 100
    }).then(
        function(v){
            arrg=v.map(x => x.appId);
            console.log("Fetched "+arrg.length)
            getReviewsForApps();
        },
        console.log
    );

setTimeout(saveToCSV, 15000);