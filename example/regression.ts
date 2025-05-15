import { Model } from "../src/core/mod.ts";
import { Dense, ReLU } from "../src/layers/mod.ts";
import { SGD } from "../src/optimizes/mod.ts";
import { MeanSquaredError } from "../src/losses/mod.ts";

// 1. Define the model
const model = new Model();

// 2. Add layers
// Simple regression: 1 input feature, 1 output feature
// Model: Input(1) -> Dense(8) -> ReLU -> Dense(1) -> Output
model.addLayer(new Dense(1, 8)); // Input layer (1 feature) to hidden layer (8 neurons)
model.addLayer(new ReLU()); // Activation for hidden layer
model.addLayer(new Dense(8, 1)); // Hidden layer (8 neurons) to output layer (1 neuron)
// No activation for the output layer for linear regression output

// 3. Compile the model
model.compile(
  new SGD(0.001), // SGD optimizer with a learning rate of 0.001
  new MeanSquaredError(), // Mean Squared Error loss function
  ["mse"], // Using 'mse' as a metric key (conceptual)
);

// 4. Prepare training data
// Aiming to learn a simple linear relationship, e.g., y = 2x + 1
// Data needs to be in number[][] format
const trainingData: number[][] = [[1], [2], [3], [4], [5], [6], [7], [8]];
const trainingLabels: number[][] = [[3], [5], [7], [9], [11], [13], [15], [17]];

// 5. Train the model (conceptual)
console.log(
  "Starting model training (conceptual, `fit` may not fully update weights)...",
);
// Using more epochs due to potentially slow convergence of SGD without full implementation
await model.fit(trainingData, trainingLabels, 1000, 2, false);
console.log("Model training finished (conceptually).");

// 6. Make predictions
const testData: number[][] = [[9], [10], [0.5]];
if (testData.length > 0) {
  const predictions = model.predict(testData);
  console.log("\nPredictions for test inputs:");
  testData.forEach((input, index) => {
    // Calculate expected output based on y = 2x + 1
    const expected = (2 * input[0] + 1).toFixed(4);
    console.log(
      `Input: [${input.join(", ")}], Predicted Output: ${
        predictions[index].map((p) => p.toFixed(4)).join(", ")
      }, Expected Output: ${expected}`,
    );
  });
} else {
  console.log("No data provided for prediction in this example.");
}

// 7. Evaluate the model (conceptual)
const validationData: number[][] = [[11], [12]];
const validationLabels: number[][] = [[23], [25]]; // Expected for y = 2x + 1
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
