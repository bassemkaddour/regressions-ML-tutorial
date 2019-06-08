const tf = require('@tensorflow/tfjs'); 
// const _ = require('lodash'); 

class LogisticRegression {
  constructor(features, labels, options) {
    // remember, tensors are immutable.
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels); 
    // cross entropy eq and MSE eq can be often referred to as cost functions 
    this.costHistory = []; 

    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000, decisionBoundary: 0.5 }, 
      options
      );
    
    this.weights = tf.zeros([this.features.shape[1], 1]);  
  }
  
  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).sigmoid();
    const differences = currentGuesses.sub(labels); 

    const gradients = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    this.weights = this.weights.sub(gradients.mul(this.options.learningRate));
  }

  train() {
    const { batchSize } = this.options; 
    const batchQuantity = Math.floor(this.features.shape[0] / batchSize); 

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * batchSize;
        const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1]); 
        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

        this.gradientDescent(featureSlice, labelSlice);
      }

      this.recordCost();
      this.updateLearningRate(); 
    }
  }

  predict(observations) {
    return this.processFeatures(observations)
      .matMul(this.weights)
      .sigmoid()
      .greater(this.options.decisionBoundary)
      .cast('float32'); 
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures); 
    const predictionCount = predictions.shape[0]; 
    testLabels = tf.tensor(testLabels); 
    
    const incorrect = predictions
      .sub(testLabels)
      .abs()
      .sum()
      // use get because sum returns a tensor
      .get(); 
    
    return (predictionCount - incorrect) / predictionCount; 
  }

  processFeatures(features) {
    features = tf.tensor(features);
    
    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }
    
    features = tf
      .ones([features.shape[0], 1])
      .concat(features, 1);

    return features;
  }

  standardize(features) {
    // the mean and variance are taken from the training set, 
    // but when they are applied to the test set, the same 
    // mean and variance should be used (instead of finding new
    // ones  the test set)
    const { mean, variance } = tf.moments(features, 0);

    this.mean = mean; 
    this.variance = variance; 

    return features.sub(mean).div(variance.pow(0.5));
  }

  recordCost() {
    // using the vectorized eq for cross entropy
    const guesses = this.features.matMul(this.weights).sigmoid(); 

    const termOne = this.labels
      .transpose()
      .matMul(guesses.log());

    const termTwo = this.labels	
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(
        guesses
          .mul(-1)
          .add(1)
          .log()
      );

    const cost = termOne
      .add(termTwo)
      .div(this.features.shape[0])
      .mul(-1)
      .get(0, 0); 

    this.costHistory.unshift(cost); 
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return; 
    }

    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05; 
    }
  }
}

module.exports = LogisticRegression; 
