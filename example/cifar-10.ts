import { readFileSync } from "node:fs";
import { Model } from "../src/core/mod.ts";
import { Dense, ReLU, Softmax } from "../src/layers/mod.ts";
import { Adam } from "../src/optimizes/mod.ts";
import { MeanSquaredError } from "../src/losses/mod.ts";

const CIFAR_IMAGE_HEIGHT = 32;
const CIFAR_IMAGE_WIDTH = 32;
const CIFAR_IMAGE_CHANNELS = 3;
const CIFAR_IMAGE_SIZE =
	CIFAR_IMAGE_HEIGHT * CIFAR_IMAGE_WIDTH * CIFAR_IMAGE_CHANNELS;
const CIFAR_NUM_CLASSES = 10;

function loadIdxLabels(filePath: string, numClasses: number): number[][] {
	const buffer = readFileSync(filePath);
	// const magic = buffer.readInt32BE(0); // 0x00000801 for labels
	const numItems = buffer.readInt32BE(4);
	const labelsData = buffer.subarray(8);

	const labels: number[][] = [];
	for (let i = 0; i < numItems; i++) {
		const label = labelsData.readUInt8(i);
		const oneHotLabel = Array(numClasses).fill(0);
		oneHotLabel[label] = 1;
		labels.push(oneHotLabel);
	}
	return labels;
}

function loadIdxImages(
	filePath: string,
	imageHeight: number,
	imageWidth: number,
	numChannels: number,
): number[][] {
	const buffer = readFileSync(filePath);
	// const magic = buffer.readInt32BE(0); // 0x00000803 for images
	const numItemsInHeader = buffer.readInt32BE(4);
	// const numRows = buffer.readInt32BE(8);
	// const numCols = buffer.readInt32BE(12);
	const imagesData = buffer.subarray(16);

	const imageSize = imageHeight * imageWidth * numChannels;
	const images: number[][] = [];

	// Calculate the actual number of items that can be processed based on buffer size
	const numItemsToProcess = Math.floor(imagesData.length / imageSize);

	if (numItemsInHeader !== numItemsToProcess) {
		console.warn(
			`[loadIdxImages] Warning: Header in file "${filePath}" indicates ${numItemsInHeader} items, but actual data size allows for ${numItemsToProcess} items. Using ${numItemsToProcess}. The file might be corrupted or truncated.`,
		);
	}
	// If numItemsToProcess is 0 and numItemsInHeader was > 0, it means the file is too small for even one image.

	for (let i = 0; i < numItemsToProcess; i++) {
		// Use numItemsToProcess instead of numItemsInHeader
		const imageData: number[] = [];
		const offset = i * imageSize;
		for (let j = 0; j < imageSize; j++) {
			// Normalize pixel values to [0, 1]
			imageData.push(imagesData.readUInt8(offset + j) / 255.0);
		}
		images.push(imageData);
	}
	return images;
}

console.log("CIFAR-10 Example using a Multi-Layer Perceptron (MLP)");
console.log(
	"Note: MLPs have limitations for image data. CNNs are preferred for better accuracy.",
);
console.log(
	"This example might be slow and memory-intensive due to the dataset size and MLP structure.",
);

// 1. Load CIFAR-10 Data
// Update these paths if your files are located elsewhere
const trainImagesPath = "./example/cifar-10/train-images.idx3-ubyte";
const trainLabelsPath = "./example/cifar-10/train-labels.idx1-ubyte";
const testImagesPath = "./example/cifar-10/t10k-images.idx3-ubyte";
const testLabelsPath = "./example/cifar-10/t10k-labels.idx1-ubyte";

let trainingData: number[][];
let trainingLabels: number[][];
let testData: number[][];
let testLabels: number[][];

try {
	console.log("Loading training images...");
	trainingData = loadIdxImages(
		trainImagesPath,
		CIFAR_IMAGE_HEIGHT,
		CIFAR_IMAGE_WIDTH,
		CIFAR_IMAGE_CHANNELS,
	);
	console.log(`Loaded ${trainingData.length} training images.`);

	console.log("Loading training labels...");
	trainingLabels = loadIdxLabels(trainLabelsPath, CIFAR_NUM_CLASSES);
	console.log(`Loaded ${trainingLabels.length} training labels.`);

	console.log("Loading test images...");
	testData = loadIdxImages(
		testImagesPath,
		CIFAR_IMAGE_HEIGHT,
		CIFAR_IMAGE_WIDTH,
		CIFAR_IMAGE_CHANNELS,
	);
	console.log(`Loaded ${testData.length} test images.`);

	console.log("Loading test labels...");
	testLabels = loadIdxLabels(testLabelsPath, CIFAR_NUM_CLASSES);
	console.log(`Loaded ${testLabels.length} test labels.`);
} catch (error) {
	console.error(
		"Error loading CIFAR-10 data. Make sure the .idx*-ubyte files are in the 'example/cifar-10/' directory.",
		error,
	);
	process.exit(1);
}

// For demonstration, let's use a small subset of the data to speed up training.
// Remove or adjust these lines to use the full dataset.
const subsetSizeTrain = 1000; // e.g., 1000 training samples
const subsetSizeTest = 200; // e.g., 200 test samples

console.warn(
	`Using a subset of data: ${subsetSizeTrain} for training, ${subsetSizeTest} for testing.`,
);
trainingData = trainingData.slice(0, subsetSizeTrain);
trainingLabels = trainingLabels.slice(0, subsetSizeTrain);
testData = testData.slice(0, subsetSizeTest);
testLabels = testLabels.slice(0, subsetSizeTest);

// 2. Define the Model (MLP)
const model = new Model();
model.addLayer(new Dense(CIFAR_IMAGE_SIZE, 128)); // Input layer (32*32*3=3072 features) to hidden layer (128 neurons)
model.addLayer(new ReLU());
model.addLayer(new Dense(128, CIFAR_NUM_CLASSES)); // Hidden layer to output layer (10 classes)
model.addLayer(new Softmax());

// 3. Compile the Model
model.compile(
	new Adam(0.001), // Adam optimizer
	new MeanSquaredError(), // Using MSE as it's available, though CrossEntropy is typical for classification
	["accuracy"], // Metric
);

// 4. Train the Model
console.log("Starting model training...");
const epochs = 10; // Adjust epochs as needed
const batchSize = 32; // Adjust batch size based on memory and performance
try {
	await model.fit(trainingData, trainingLabels, epochs, batchSize, true); // Enable debugEpochEnabled
	console.log("Model training finished.");
} catch (e) {
	console.error("Error during model training:", e);
}

// 5. Evaluate the Model
if (testData.length > 0 && testLabels.length > 0) {
	console.log("\nEvaluating model...");
	try {
		const evaluation = model.evaluate(testData, testLabels);
		console.log(
			"Model Evaluation (conceptual, built-in accuracy is for binary):",
			evaluation,
		);

		// Manual accuracy calculation for multi-class
		const predictions = model.predict(testData);
		let correctPredictions = 0;
		for (let i = 0; i < predictions.length; i++) {
			const predictedClassIndex = predictions[i].indexOf(
				Math.max(...predictions[i]),
			);
			const actualClassIndex = testLabels[i].indexOf(1);
			if (predictedClassIndex === actualClassIndex) {
				correctPredictions++;
			}
		}
		const manualAccuracy = correctPredictions / predictions.length;
		console.log(
			`Manual Multi-class Accuracy on Test Set: ${(manualAccuracy * 100).toFixed(2)}%`,
		);
	} catch (e) {
		console.error("Error during model evaluation:", e);
	}
}

// 6. Make some predictions (optional)
const cifarClassNames = [
	"airplane",
	"automobile",
	"bird",
	"cat",
	"deer",
	"dog",
	"frog",
	"horse",
	"ship",
	"truck",
];

if (testData.length > 5) {
	console.log("\nSample Predictions (first 5 test images):");
	const samplePredictions = model.predict(testData.slice(0, 5));
	samplePredictions.forEach((prediction, index) => {
		const predictedClassIndex = prediction.indexOf(Math.max(...prediction));
		const actualClassIndex = testLabels[index].indexOf(1);
		console.log(
			`Sample ${index + 1}: Predicted: ${cifarClassNames[predictedClassIndex]}, Actual: ${cifarClassNames[actualClassIndex]}`,
		);
	});
}

console.log("\n--- CIFAR-10 MLP Example End ---");
console.log(
	"Reminder: For better performance on CIFAR-10, consider using Convolutional Neural Networks (CNNs).",
);
