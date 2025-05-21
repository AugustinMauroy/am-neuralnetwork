# Neural Network

[![JSR](https://jsr.io/badges/@am/neuralnetwork)](https://jsr.io/@am/neuralnetwork)

`@am/neuralnetwork` is a simple neural network library written in TypeScript. It
is designed to be easy to use and understand, while still being powerful enough
to be useful in real-world applications.

Also, it's cross-runtime compatible, meaning you can use it in both [Node.js](https://nodejs.org/) and [Deno](https://deno.land/), [bun](https://bun.sh) as well as in the browser. It

## How to construct a neural network

Constructing a neural network with `@am/neuralnetwork` involves a few key steps:

1.  **Define the Model**: Instantiate the `Model` class.
    ```typescript
    import { Model } from "@am/neuralnetwork";
    const model = new Model();
    ```
2.  **Add Layers**: Add layers to the model. This includes dense layers for computation and activation layers like ReLU or Sigmoid.
    ```typescript
    import { Dense, ReLU, Sigmoid } from "@am/neuralnetwork/layers";
    // Example:
    model.addLayer(new Dense(2, 2)); // Input to hidden layer
    model.addLayer(new ReLU());      // Activation for hidden layer
    model.addLayer(new Dense(2, 1)); // Hidden to output layer
    model.addLayer(new Sigmoid());   // Activation for output layer
    ```
3.  **Compile the Model**: Configure the learning process by specifying an optimizer, a loss function, and optionally, metrics.
    ```typescript
    import { Adam } from "@am/neuralnetwork/optimizers";
    import { MeanSquaredError } from "@am/neuralnetwork/losses";
    model.compile(
        new Adam(0.01),
        new MeanSquaredError(),
        ["accuracy"], // Optional metrics
    );
    ```
4.  **Prepare Training Data**: Create arrays for your input samples (`trainingData`) and their corresponding output labels (`trainingLabels`).
    ```typescript
    const trainingData = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const trainingLabels = [[0], [1], [1], [0]];
    ```
5.  **Train the Model**: Use the `fit` method to train the model on your data.
    ```typescript
    await model.fit(trainingData, trainingLabels, 10000, 4, false); // epochs, batchSize, shuffle
    ```
6.  **Make Predictions**: Use the `predict` method with new input data to get predictions from the trained model.
    ```typescript
    const newData = [[0, 0], [0, 1]];
    const predictions = model.predict(newData);
    ```
7.  **Evaluate the Model**: (Conceptually) Use the `evaluate` method with validation data and labels to assess the model's performance.
    ```typescript
    const validationData = [[0, 0], [0, 1]];
    const validationLabels = [[0], [1]];
    const evaluation = model.evaluate(validationData, validationLabels);
    ```

## Simple xor example

Here's a simple example of how to use `@am/neuralnetwork` to create a neural network that learns the XOR function. This example demonstrates the steps outlined above.

```typescript
import { Model } from "@am/neuralnetwork";
import { Dense } from "@am/neuralnetwork/layers";
import { ReLU, Sigmoid } from "@am/neuralnetwork/layers";
import { Adam } from "@am/neuralnetwork/optimizers";
import { MeanSquaredError } from "@am/neuralnetwork/losses";

// 1. Define the model
const model = new Model();

// 2. Add layers
// XOR problem: 2 input neurons, e.g., 2 hidden neurons, 1 output neuron
model.addLayer(new Dense(2, 2)); // Input layer (2 features) to hidden layer (2 neurons)
model.addLayer(new ReLU()); // Activation for hidden layer
model.addLayer(new Dense(2, 1)); // Hidden layer (2 neurons) to output layer (1 neuron)
model.addLayer(new Sigmoid()); // Sigmoid activation for binary output

// 3. Compile the model
// The Adam optimizer and MeanSquaredError loss are available
model.compile(
	new Adam(0.01), // Adam optimizer with a learning rate of 0.01
	new MeanSquaredError(), // Mean Squared Error loss function
	["accuracy"], // Placeholder for metrics, as evaluation logic is not fully implemented
);

// 4. Prepare training data (conceptual)
// trainingData would be an array of input samples, e.g., number[][]
// trainingLabels would be an array of corresponding output labels, e.g., number[][]
const trainingData = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1],
];
const trainingLabels = [[0], [1], [1], [0]];

// 5. Train the model (conceptual, as `fit` is not fully implemented)
console.log(
	"Starting model training (conceptual, `fit` may not be fully implemented)...",
);
await model.fit(trainingData, trainingLabels, 10000, 4, false);
console.log("Model training finished (conceptually).");

// 6. Make predictions
// someNewData would be an array of new input samples, e.g., number[][]
const someNewData = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1],
];
if (someNewData.length > 0) {
	const predictions = model.predict(someNewData);
	console.log("Predictions for XOR inputs:");
	someNewData.forEach((input, index) => {
		console.log(
			`Input: [${input.join(", ")}], Output: ${predictions[index]}, Expected: ${
				trainingLabels[index]
			}`,
		);
	});
} else {
	console.log("No data provided for prediction in this example.");
}

// 7. Evaluate the model (conceptual, as `evaluate` is not fully implemented)
const validationData: number[][] = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1],
];
const validationLabels: number[][] = [[0], [1], [1], [0]];
if (validationData.length > 0) {
	const evaluation = model.evaluate(validationData, validationLabels);
	console.log("Evaluation:", evaluation);
}
```

If you want more examples, check out the [examples](https://github.com/AugustinMauroy/am-neuralnetwork/tree/main/examples) folder.