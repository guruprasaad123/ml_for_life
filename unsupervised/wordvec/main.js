const wordVectors = ml5.word2vec('./data/arts.json', modelLoaded);

// When the model is loaded
function modelLoaded() {
  console.log('Model Loaded!');
}

// Find the closest word to 'rainbow'
wordVectors.nearest('earth', function(err, results) {
  console.log(results);
});

