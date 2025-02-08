import { ActivationFunction, NeuralNetwork } from "../src/nn.ts";

const nn = new NeuralNetwork(
  {
    inputSize: 2,
    hiddenSize: 2,
    outputSize: 1,
    learningRate: 0.01,
    activationFunction: ActivationFunction.SIGMOID,
  },
  ActivationFunction.SIGMOID,
);

const trainingData = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

const validationData = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

nn.train(
  trainingData.map(({ input, output }) => [input, output]),
  validationData.map(({ input, output }) => [input, output]),
  100000,
);

// calculate the accuracy in %
let accuracy = 0;
validationData.forEach(({ input, output }) => {
  const prediction = nn.feedforward(input)[0];
  const roundedPrediction = prediction < 0.5 ? 0 : 1;
  if (roundedPrediction === output[0]) {
    accuracy++;
  }
});
accuracy /= validationData.length;

console.log(`Accuracy: ${accuracy * 100}%`);

validationData.forEach(({ input, output }) => {
  const prediction = nn.feedforward(input)[0];
  console.log(
    `Input: ${input[0]} ${input[1]} Output: ${
      output[0]
    } Prediction: ${prediction}`,
  );
});
