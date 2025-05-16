import { Model } from "../src/core/mod.ts";
import { Dense, ReLU, Softmax } from "../src/layers/mod.ts";
import { SGD } from "../src/optimizes/mod.ts";
import { MeanSquaredError } from "../src/losses/mod.ts";

// Helper function to generate data
function generateAngleData(numSamplesPerQuadrant: number): {
	data: number[][];
	labels: number[][];
} {
	const data: number[][] = [];
	const labels: number[][] = [];
	const PI = Math.PI;

	const quadrantRanges = [
		{ min: 0.01, max: PI / 2, id: 0 }, // Quadrant 1 (0 to 90 deg)
		{ min: PI / 2 + 0.01, max: PI, id: 1 }, // Quadrant 2 (90 to 180 deg)
		{ min: PI + 0.01, max: (3 * PI) / 2, id: 2 }, // Quadrant 3 (180 to 270 deg)
		{ min: (3 * PI) / 2 + 0.01, max: 2 * PI, id: 3 }, // Quadrant 4 (270 to 360 deg)
	];

	for (const range of quadrantRanges) {
		for (let i = 0; i < numSamplesPerQuadrant; i++) {
			const angle = Math.random() * (range.max - range.min) + range.min;
			data.push([Math.cos(angle), Math.sin(angle)]);
			const oneHotLabel = [0, 0, 0, 0];
			oneHotLabel[range.id] = 1;
			labels.push(oneHotLabel);
		}
	}
	return { data, labels };
}

// 1. Define the model
const model = new Model();

// 2. Add layers
// Input: [cos(angle), sin(angle)] -> 2 features
// Output: [q1_prob, q2_prob, q3_prob, q4_prob] -> 4 classes
model.addLayer(new Dense(2, 16)); // Input layer (2 features) to hidden layer (16 neurons)
model.addLayer(new ReLU()); // Activation for hidden layer
model.addLayer(new Dense(16, 4)); // Hidden layer (16 neurons) to output layer (4 neurons for 4 quadrants)
model.addLayer(new Softmax()); // Softmax activation for multi-class probabilities

// 3. Compile the model
model.compile(
	new SGD(0.01), // SGD optimizer with a learning rate of 0.01
	new MeanSquaredError(), // Mean Squared Error loss function
	["accuracy"], // Using 'accuracy' as a metric key (conceptual)
);

// 4. Prepare training data
const { data: trainingData, labels: trainingLabels } = generateAngleData(20); // 20 samples per quadrant

// 5. Train the model (conceptual)
console.log(
	"Starting model training for quadrant classification (conceptual, `fit` may not fully update weights)...",
);
// Using a moderate number of epochs
await model.fit(trainingData, trainingLabels, 200, 8, true); // 200 epochs, batch size 8, debug epochs
console.log("Model training finished (conceptually).");

// 6. Make predictions
// Test with a few specific angles
const testAnglesRad = [
	Math.PI / 4, // Q1 (45 deg)
	(3 * Math.PI) / 4, // Q2 (135 deg)
	(5 * Math.PI) / 4, // Q3 (225 deg)
	(7 * Math.PI) / 4, // Q4 (315 deg)
	0.1, // Q1
	Math.PI - 0.1, // Q2
	Math.PI + 0.1, // Q3
	2 * Math.PI - 0.1, // Q4
];

const testData: number[][] = testAnglesRad.map((angle) => [
	Math.cos(angle),
	Math.sin(angle),
]);
const quadrantNames = ["Quadrant 1", "Quadrant 2", "Quadrant 3", "Quadrant 4"];

if (testData.length > 0) {
	const predictions = model.predict(testData);
	console.log("\nPredictions for test angles:");
	testData.forEach((input, index) => {
		const predictedProbs = predictions[index];
		const predictedQuadrantIndex = predictedProbs.indexOf(
			Math.max(...predictedProbs),
		);
		console.log(
			`Angle (cos, sin): [${input[0].toFixed(3)}, ${input[1].toFixed(
				3,
			)}] (Actual ~${
				quadrantNames[Math.floor(testAnglesRad[index] / (Math.PI / 2)) % 4]
			}) -> Predicted: ${quadrantNames[predictedQuadrantIndex]} (Probs: ${predictedProbs
				.map((p) => p.toFixed(3))
				.join(", ")})`,
		);
	});
} else {
	console.log("No data provided for prediction in this example.");
}

// 7. Evaluate the model (conceptual)
const { data: validationData, labels: validationLabels } = generateAngleData(5); // 5 validation samples per quadrant
if (validationData.length > 0) {
	const evaluation = model.evaluate(validationData, validationLabels);
	console.log("\nEvaluation (conceptual):", evaluation);
}

console.log(
	"\nNote: True model training requires full implementation of weight updates in layers and the Model.fit method.",
);
console.log(
	"Predictions will likely be inaccurate as weights are not properly trained.",
);
