/**
 * A module simple neural network implementation
 * @module
 */

import { MathUtils } from "./math_utils.ts";

/**
 * Enum representing the available activation functions for the neural network.
 */
enum ActivationFunction {
  TANH,
  SIGMOID,
  RELU,
  LINEAR,
  TANH_DERIVATIVE,
  SOFTMAX,
}

interface NeuralNetworkConfig {
  /**
   * The size of the input layer.
   */
  inputSize: number;
  /**
   * The size of the hidden layer.
   */
  hiddenSize: number;
  /**
   * The size of the output layer.
   */
  outputSize: number;
  /**
   * The learning rate of the neural network.
   */
  learningRate: number;
  /**
   * The activation function to use in the neural network.
   */
  activationFunction: ActivationFunction;
}

/**
 * Class representing a neural network.
 * The neural network is a feedforward neural network with one hidden layer.
 * The neural network uses the Adam optimizer for training.
 * The neural network supports the following activation functions:
 * - Sigmoid
 * - Tanh
 * - ReLU
 * - Linear
 * - Tanh derivative
 * - Softmax
 * The neural network uses mean squared error as the loss function.
 * The neural network uses the following hyperparameters:
 * - beta1 = 0.9
 * - beta2 = 0.999
 * - epsilon = 1e-8
 * @example
 * const config = {
 *  inputSize: 2,
 *  hiddenSize: 2,
 *  outputSize: 2,
 *  learningRate: 0.01,
 *  activationFunction: ActivationFunction.TANH,
 * };
 *
 * const nn = new NeuralNetwork(config, ActivationFunction.TANH);
 */
class NeuralNetwork {
  private inputSize: number;
  private hiddenSize: number;
  private outputSize: number;
  private learningRate: number;
  private activationFunction: ActivationFunction;

  private weights: {
    inputToHidden: number[][];
    hiddenToOutput: number[][];
  };

  /**
   * The optimizer parameters for the Adam optimizer.
   */
  private optimizerParams: {
    inputToHiddenM: number[][];
    inputToHiddenV: number[][];
    hiddenToOutputM: number[][];
    hiddenToOutputV: number[][];
    t: number;
  };

  private beta1 = 0.9;
  private beta2 = 0.999;
  private epsilon = 1e-8;

  constructor(
    config: NeuralNetworkConfig,
    activationFunction: ActivationFunction,
  ) {
    this.inputSize = config.inputSize;
    this.hiddenSize = config.hiddenSize;
    this.outputSize = config.outputSize;
    this.learningRate = config.learningRate;
    this.activationFunction = activationFunction;

    // Initialize the weights and optimizer parameters of the neural network with random values
    this.weights = {
      inputToHidden: Array.from(
        { length: this.inputSize },
        () => Array.from({ length: this.hiddenSize }, () => Math.random()),
      ),
      hiddenToOutput: Array.from(
        { length: this.hiddenSize },
        () => Array.from({ length: this.outputSize }, () => Math.random()),
      ),
    };
    this.optimizerParams = {
      inputToHiddenM: Array.from(
        { length: this.inputSize },
        () => Array.from({ length: this.hiddenSize }, () => 0),
      ),
      inputToHiddenV: Array.from(
        { length: this.inputSize },
        () => Array.from({ length: this.hiddenSize }, () => 0),
      ),
      hiddenToOutputM: Array.from(
        { length: this.hiddenSize },
        () => Array.from({ length: this.outputSize }, () => 0),
      ),
      hiddenToOutputV: Array.from(
        { length: this.hiddenSize },
        () => Array.from({ length: this.outputSize }, () => 0),
      ),
      t: 0,
    };
  }

  private activate(x: number): number {
    switch (this.activationFunction) {
      case ActivationFunction.SIGMOID:
        return MathUtils.sigmoid(x);
      case ActivationFunction.TANH:
        return MathUtils.tanh(x);
      case ActivationFunction.RELU:
        return Math.max(0, x); // ReLU activation function
      case ActivationFunction.LINEAR:
        return x; // Linear activation function
      case ActivationFunction.TANH_DERIVATIVE:
        return 1 - MathUtils.tanh(x) * MathUtils.tanh(x); // Derivative of tanh
      case ActivationFunction.SOFTMAX:
        // Softmax will be applied during the feedforward step
        return x;
      default:
        return MathUtils.tanh(x);
    }
  }

  public feedforward(inputs: number[]): number[] {
    const hiddenOutputs: number[] = Array(this.hiddenSize).fill(0);

    // Calculate the outputs of the hidden layer
    for (let i = 0; i < this.hiddenSize; i++) {
      let sum = 0;
      for (let j = 0; j < this.inputSize; j++) {
        sum += inputs[j] * this.weights.inputToHidden[j][i];
      }
      hiddenOutputs[i] = this.activate(sum);
    }

    const outputs: number[] = Array(this.outputSize).fill(0);

    // Calculate the outputs of the output layer
    for (let i = 0; i < this.outputSize; i++) {
      let sum = 0;
      for (let j = 0; j < this.hiddenSize; j++) {
        sum += hiddenOutputs[j] * this.weights.hiddenToOutput[j][i];
      }
      outputs[i] = this.activate(sum);
    }

    // Apply softmax activation for the output layer
    if (this.activationFunction === ActivationFunction.SOFTMAX) {
      let expSum = 0;
      for (let i = 0; i < this.outputSize; i++) {
        expSum += Math.exp(outputs[i]);
      }

      for (let i = 0; i < this.outputSize; i++) {
        outputs[i] = Math.exp(outputs[i]) / expSum;
      }
    }

    return outputs;
  }

  /**
   * Backpropagation algorithm to train the neural network.
   * @param inputs The input data for the neural network.
   * @param targets The target data for the neural network.
   */
  public backpropagation(inputs: number[], targets: number[]): void {
    const hiddenOutputs: number[] = Array(this.hiddenSize).fill(0);
    const outputs: number[] = Array(this.outputSize).fill(0);

    // Calculate the outputs of the hidden layer and the final output
    for (let i = 0; i < this.hiddenSize; i++) {
      let sum = 0;
      for (let j = 0; j < this.inputSize; j++) {
        sum += inputs[j] * this.weights.inputToHidden[j][i];
      }
      hiddenOutputs[i] = this.activate(sum);
    }

    for (let i = 0; i < this.outputSize; i++) {
      let sum = 0;
      for (let j = 0; j < this.hiddenSize; j++) {
        sum += hiddenOutputs[j] * this.weights.hiddenToOutput[j][i];
      }
      outputs[i] = this.activate(sum);
    }

    // Calculate the output error
    const outputErrors: number[] = Array(this.outputSize).fill(0);
    for (let i = 0; i < this.outputSize; i++) {
      outputErrors[i] = targets[i] - outputs[i];
    }

    // Calculate the hidden layer error
    const hiddenErrors: number[] = Array(this.hiddenSize).fill(0);
    for (let i = 0; i < this.hiddenSize; i++) {
      let sum = 0;
      for (let j = 0; j < this.outputSize; j++) {
        sum += outputErrors[j] * this.weights.hiddenToOutput[i][j];
      }
      hiddenErrors[i] = hiddenOutputs[i] * (1 - hiddenOutputs[i]) * sum;
    }

    // Update the weights from the hidden layer to the output
    this.optimizerParams.t++;
    for (let i = 0; i < this.hiddenSize; i++) {
      for (let j = 0; j < this.outputSize; j++) {
        this.optimizerParams.hiddenToOutputM[i][j] =
          this.beta1 * this.optimizerParams.hiddenToOutputM[i][j] +
          (1 - this.beta1) * outputErrors[j] * hiddenOutputs[i];
        this.optimizerParams.hiddenToOutputV[i][j] =
          this.beta2 * this.optimizerParams.hiddenToOutputV[i][j] +
          (1 - this.beta2) * outputErrors[j] * outputErrors[j];
        const correctedM = this.optimizerParams.hiddenToOutputM[i][j] /
          (1 - Math.pow(this.beta1, this.optimizerParams.t));
        const correctedV = this.optimizerParams.hiddenToOutputV[i][j] /
          (1 - Math.pow(this.beta2, this.optimizerParams.t));
        this.weights.hiddenToOutput[i][j] += (this.learningRate * correctedM) /
          (Math.sqrt(correctedV) + this.epsilon);
      }
    }

    // Update the weights from the input to the hidden layer
    for (let i = 0; i < this.inputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        this.optimizerParams.inputToHiddenM[i][j] =
          this.beta1 * this.optimizerParams.inputToHiddenM[i][j] +
          (1 - this.beta1) * hiddenErrors[j] * inputs[i];
        this.optimizerParams.inputToHiddenV[i][j] =
          this.beta2 * this.optimizerParams.inputToHiddenV[i][j] +
          (1 - this.beta2) * hiddenErrors[j] * hiddenErrors[j];
        const correctedM = this.optimizerParams.inputToHiddenM[i][j] /
          (1 - Math.pow(this.beta1, this.optimizerParams.t));
        const correctedV = this.optimizerParams.inputToHiddenV[i][j] /
          (1 - Math.pow(this.beta2, this.optimizerParams.t));
        this.weights.inputToHidden[i][j] += (this.learningRate * correctedM) /
          (Math.sqrt(correctedV) + this.epsilon);
      }
    }
  }

  /**
   * Train the neural network using the Adam optimizer.
   * @param trainingData The training data for the neural network.
   * @param validationData The validation data for the neural network.
   * @param numberOfIterations The number of iterations to train the neural network.
   * @example
   * const trainingData = [
   * [[0, 0], [0]],
   * [[0, 1], [1]],
   * [[1, 0], [1]],
   * [[1, 1], [0]],
   * ];
   * const validationData = [
   * [[0, 0], [0]],
   * [[0, 1], [1]],
   * [[1, 0], [1]],
   * [[1, 1], [0]],
   * ];
   * nn.train(trainingData, validationData, 1000);
   */
  public train(
    trainingData: Array<[number[], number[]]>,
    validationData: Array<[number[], number[]]>,
    numberOfIterations: number,
  ): void {
    let bestValidationLoss = Number.MAX_VALUE;
    let bestWeightsInputToHidden: number[][] = [];
    let bestWeightsHiddenToOutput: number[][] = [];

    for (let i = 0; i < numberOfIterations; i++) {
      const randomIndex = Math.floor(Math.random() * trainingData.length);
      const [randomInputs, randomTargets] = trainingData[randomIndex];
      this.backpropagation(randomInputs, randomTargets);

      // Evaluate on validation set periodically
      if ((i + 1) % 100 === 0) {
        const validationLoss = this.calculateLoss(validationData);
        if (validationLoss < bestValidationLoss) {
          bestValidationLoss = validationLoss;
          bestWeightsInputToHidden = this.weights.inputToHidden;
          bestWeightsHiddenToOutput = this.weights.hiddenToOutput;
        }
      }
    }

    // Restore best weights
    this.weights.inputToHidden = bestWeightsInputToHidden;
    this.weights.hiddenToOutput = bestWeightsHiddenToOutput;
  }

  /**
   * Calculate the loss of the neural network using the mean squared error loss function.
   * @param data The data to calculate the loss on.
   * @returns The loss of the neural network.
   * @example
   * const data = [
   * [[0, 0], [0]],
   * [[0, 1], [1]],
   * [[1, 0], [1]],
   * [[1, 1], [0]],
   * ];
   * const loss = nn.calculateLoss(data);
   * console.log(loss);
   * // Output: 0.25 (hypothetical value)
   */
  public calculateLoss(data: Array<[number[], number[]]>): number {
    let totalLoss = 0;
    for (const [inputs, targets] of data) {
      const outputs = this.feedforward(inputs);
      let instanceLoss = 0;
      for (let i = 0; i < this.outputSize; ++i) {
        instanceLoss += Math.pow(targets[i] - outputs[i], 2); // Using mean squared error
      }
      totalLoss += instanceLoss / this.outputSize;
    }
    return totalLoss / data.length;
  }
}

export { ActivationFunction, NeuralNetwork };
export type { NeuralNetworkConfig };
