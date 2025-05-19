import { readFileSync } from "node:fs";

import { Model } from "../src/core/mod.ts";
import { Dense, ReLU, Softmax } from "../src/layers/mod.ts";
import { Adam } from "../src/optimizes/mod.ts";
import { MeanSquaredError } from "../src/losses/mod.ts";

// Helper function to parse CSV data and prepare it for the model
function loadIrisData(filePath: string): {
    features: number[][];
    labels: number[][];
    classNames: string[];
} {
    const csvString = readFileSync(filePath, "utf-8");
    const lines = csvString.trim().split("\n");
    const header = lines.shift()?.split(",").map((s) => s.replace(/"/g, ""));
    if (!header) throw new Error("CSV header not found");

    const features: number[][] = [];
    const rawLabels: string[] = [];
    const classNames = ["Setosa", "Versicolor", "Virginica"]; // Predefined class order

    //lines.forEach((line) => {
    for (const line of lines) {
        const values = line.split(",");
        features.push(values.slice(0, 4).map(Number));
        rawLabels.push(values[4].replace(/"/g, ""));
    };

    // One-hot encode labels
    const labels: number[][] = rawLabels.map((label) => {
        const oneHot = [0, 0, 0];
        const index = classNames.indexOf(label);
        if (index !== -1) {
            oneHot[index] = 1;
        }
        return oneHot;
    });

    return { features, labels, classNames };
}

// 1. Prepare data
// Load the full dataset from the CSV file
const dataPath = "./example/iris.csv"; // Relative path to the CSV file
const {
    features: allFeatures,
    labels: allLabels,
    classNames,
} = loadIrisData(dataPath);

// Shuffle the data (optional but good practice)
const combined = allFeatures.map((feature, i) => ({
    feature,
    label: allLabels[i],
}));
combined.sort(() => Math.random() - 0.5);
const trainingData = combined.map((d) => d.feature);
const trainingLabels = combined.map((d) => d.label);

// Simple 80/20 split for training and validation
const splitIndex = Math.floor(trainingData.length * 0.8);
const trainFeatures = trainingData.slice(0, splitIndex);
const trainLabels = trainingLabels.slice(0, splitIndex);
const valFeatures = trainingData.slice(splitIndex);
const valLabels = trainingLabels.slice(splitIndex);

// 2. Define the model
const model = new Model();
model.addLayer(new Dense(4, 10)); // Input layer (4 features) to hidden layer (10 neurons)
model.addLayer(new ReLU()); // Activation for hidden layer
model.addLayer(new Dense(10, 3)); // Hidden layer (10 neurons) to output layer (3 classes)
model.addLayer(new Softmax()); // Softmax activation for multi-class probabilities

// 3. Compile the model
model.compile(
    new Adam(0.005), // Adam optimizer with a slightly smaller learning rate
    new MeanSquaredError(), // Mean Squared Error loss function
    ["accuracy"],
);

// 4. Train the model
console.log("Starting Iris model training...");
await model.fit(trainFeatures, trainLabels, 5, 5);
console.log("Model training finished.");

// 5. Make predictions,
if (valFeatures.length > 0) {
    console.log("\nPredictions on validation data (first 5 samples):");
    const predictions = model.predict(valFeatures);
    predictions.slice(0, 5).forEach((prediction, index) => {
        const predictedProbs = prediction;
        const actualLabelOneHot = valLabels[index];

        const predictedClassIndex = predictedProbs.indexOf(
            Math.max(...predictedProbs),
        );
        const actualClassIndex = actualLabelOneHot.indexOf(1);

        console.log(
            `Input: [${valFeatures[index].join(", ")}] -> Predicted: ${
                classNames[predictedClassIndex]
            } (Probs: ${predictedProbs
                .map((p) => p.toFixed(3))
                .join(", ")}), Actual: ${classNames[actualClassIndex]}`,
        );
    });
}

// 6. Evaluate the model
if (valFeatures.length > 0) {
    const evaluation = model.evaluate(valFeatures, valLabels);
    console.log("\nModel Evaluation (conceptual):", evaluation);
}