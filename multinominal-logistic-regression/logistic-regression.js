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
    
    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
  }
  
  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).softmax();
    const differences = currentGuesses.sub(labels); 
    
    const gradients = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    return this.weights.sub(gradients.mul(this.options.learningRate));
  }

  train() {
    const { batchSize } = this.options; 
    const batchQuantity = Math.floor(this.features.shape[0] / batchSize); 
    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * batchSize;
        
        this.weights = tf.tidy(() => {
          const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1]); 
          const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);
  
          return this.gradientDescent(featureSlice, labelSlice);
        });
      }
      
      this.recordCost();
      this.updateLearningRate(); 
    }
  }

  predict(observations) {
    return this.processFeatures(observations)
      .matMul(this.weights)
      .softmax()
      .argMax(1);  
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures); 
    const predictionCount = predictions.shape[0]; 
    testLabels = tf.tensor(testLabels).argMax(1); 
    
    const incorrect = predictions
      .notEqual(testLabels)
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

    // columns of all 0s will lead to division of a variance of 0 
    // ideally we would remove these columns since they are not 
    // providing any beneift to the analysis, but it is an easier 
    // solution to just change any division of 0 to division by 1
    // so filler will find any 0 value and create a tensor with a 1
    // in that position, while any non-zero value will have a 0 in 
    // its position. we can add this to our variance to replace all 
    // 0 variance columns with 1 variance columns. 
    const filler = variance.cast('bool').logicalNot().cast('float32'); 

    this.mean = mean; 
    this.variance = variance.add(filler); 

    return features.sub(mean).div(this.variance.pow(0.5));
  }

  recordCost() {
    // using the vectorized eq for cross entropy

    const cost = tf.tidy(() => {
      const guesses = this.features.matMul(this.weights).sigmoid(); 
  
      const termOne = this.labels
        .transpose()
        .matMul(guesses.add(1e-7).log());
  
      const termTwo = this.labels	
        .mul(-1)
        .add(1)
        .transpose()
        .matMul(
          guesses
            .mul(-1)
            .add(1)
            .add(1e-7) // avoid log0
            .log()
        );
  
      return termOne
        .add(termTwo)
        .div(this.features.shape[0])
        .mul(-1)
        .get(0, 0); 
    });

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
