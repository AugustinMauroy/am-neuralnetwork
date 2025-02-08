import { ActivationFunction, NeuralNetwork } from "../src/nn.ts";

// input is an angle in degrees
// ouput is the quadrant the angle

const nn = new NeuralNetwork(
  {
    inputSize: 1,
    hiddenSize: 1,
    outputSize: 1,
    learningRate: 0.01,
    activationFunction: ActivationFunction.SIGMOID,
  },
  ActivationFunction.SIGMOID,
);

const trainingData = [] as { input: number[]; output: number[] }[];

for (let i = 0; i < 360; i++) {
  const input = [i];
  const output = [Math.floor(i / 90)];
  trainingData.push({ input, output });
}

const validationData = [] as { input: number[]; output: number[] }[];

for (let i = 0; i < 360; i++) {
  const input = [i];
  const output = [Math.floor(i / 90)];
  validationData.push({ input, output });
}

nn.train(
  trainingData.map(({ input, output }) => [input, output]),
  validationData.map(({ input, output }) => [input, output]),
  100000,
);

// calculate the accuracy in %

let accuracy = 0;

validationData.forEach(({ input, output }) => {
  const prediction = nn.feedforward(input)[0];
  const roundedPrediction = Math.round(prediction);
  if (roundedPrediction === output[0]) {
    accuracy++;
  }
});

accuracy /= validationData.length;

console.log(`Accuracy: ${accuracy * 100}%`);

validationData.forEach(({ input, output }) => {
  const prediction = nn.feedforward(input)[0];
  console.log(
    `Input: ${input[0]} Output: ${output[0]} Prediction: ${prediction}`,
  );
});
