//version of tf library that tells where to run calculations
require('@tensorflow/tfjs-node'); 
// makes tf library available 
const tf = require('@tensorflow/tfjs'); 
const loadCSV = require('../load-csv.js'); 
const LinearRegression = require('./linear-regression.js'); 
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
  shuffle: true, 
  splitTest: 50, 
  dataColumns: ['horsepower', 'weight', 'displacement'], 
  labelColumns: ['mpg']
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1, 
  iterations: 3, 
  batchSize: 10
}); 

regression.train(); 
const r2 = regression.test(testFeatures, testLabels);

plot ({
  x: regression.MSEHistory.reverse(), 
  xLabel: 'Iterations', 
  yLabel: 'Mean Squared Error'
});
console.log('r2: ', r2);

regression
  .predict([ 
    [120, 2, 380], 
    [135, 2.1, 420]
  ])
  .print();
