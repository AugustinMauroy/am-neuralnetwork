import { Model } from "../src/core/mod.ts";
import { Dense } from "../src/layers/mod.ts";
import { ReLU, Sigmoid } from "../src/layers/mod.ts";
import { Adam } from "../src/optimizes/mod.ts";
import { MeanSquaredError } from "../src/losses/mse.ts";

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
