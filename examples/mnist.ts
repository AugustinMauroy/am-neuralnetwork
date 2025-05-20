import { readFileSync } from "node:fs";
import { Model } from "../src/core/mod.ts";
import { Dense, ReLU, Softmax } from "../src/layers/mod.ts";
import { Adam } from "../src/optimizes/mod.ts";
import { CrossEntropyLoss } from "../src/losses/mod.ts";

const MNIST_IMAGE_MAGIC_NUMBER = 2051;
const MNIST_LABEL_MAGIC_NUMBER = 2049;
const MNIST_IMAGE_HEIGHT = 28;
const MNIST_IMAGE_WIDTH = 28;
const MNIST_IMAGE_SIZE = MNIST_IMAGE_HEIGHT * MNIST_IMAGE_WIDTH;
const MNIST_NUM_CLASSES = 10;

/**
 * Loads MNIST images from an IDX3-UBYTE file.
 * @param filePath Path to the IDX3-UBYTE file.
 * @returns A 2D array of numbers, where each inner array is a flattened image (784 pixels), normalized to [0, 1].
 */
function loadMnistImages(filePath: string): number[][] {
	const buffer = readFileSync(filePath);
	const dataView = new DataView(buffer.buffer);

	const magicNumber = dataView.getInt32(0, false); // false for big-endian
	if (magicNumber !== MNIST_IMAGE_MAGIC_NUMBER) {
		throw new Error(
			`Invalid magic number for images file: ${magicNumber}. Expected ${MNIST_IMAGE_MAGIC_NUMBER}`,
		);
	}

	const numImages = dataView.getInt32(4, false);
	const numRows = dataView.getInt32(8, false);
	const numCols = dataView.getInt32(12, false);

	if (numRows !== MNIST_IMAGE_HEIGHT || numCols !== MNIST_IMAGE_WIDTH) {
		throw new Error(
			`Unexpected image dimensions: ${numRows}x${numCols}. Expected ${MNIST_IMAGE_HEIGHT}x${MNIST_IMAGE_WIDTH}`,
		);
	}

	const images: number[][] = [];
	let offset = 16; // Start of image data

	for (let i = 0; i < numImages; i++) {
		const imageData = new Array(MNIST_IMAGE_SIZE);
		for (let j = 0; j < MNIST_IMAGE_SIZE; j++) {
			imageData[j] = dataView.getUint8(offset++) / 255.0; // Normalize pixel to [0, 1]
		}
		images.push(imageData);
	}
	return images;
}

/**
 * Loads MNIST labels from an IDX1-UBYTE file.
 * @param filePath Path to the IDX1-UBYTE file.
 * @param numClasses The number of classes for one-hot encoding.
 * @returns A 2D array of numbers, where each inner array is a one-hot encoded label.
 */
function loadMnistLabels(filePath: string, numClasses: number): number[][] {
	const buffer = readFileSync(filePath);
	const dataView = new DataView(buffer.buffer);

	const magicNumber = dataView.getInt32(0, false);
	if (magicNumber !== MNIST_LABEL_MAGIC_NUMBER) {
		throw new Error(
			`Invalid magic number for labels file: ${magicNumber}. Expected ${MNIST_LABEL_MAGIC_NUMBER}`,
		);
	}

	const numLabels = dataView.getInt32(4, false);
	const labels: number[][] = [];
	let offset = 8; // Start of label data

	for (let i = 0; i < numLabels; i++) {
		const label = dataView.getUint8(offset++);
		const oneHotLabel = Array(numClasses).fill(0);
		if (label < numClasses) {
			oneHotLabel[label] = 1;
		} else {
			console.warn(
				`[loadMnistLabels] Warning: Label ${label} out of bounds for numClasses ${numClasses}`,
			);
		}
		labels.push(oneHotLabel);
	}
	return labels;
}

function displayImageInTerminal(image: number[]): void {
	const width = Math.sqrt(image.length);
	const height = width;
	let output = "";

	for (let i = 0; i < height; i++) {
		for (let j = 0; j < width; j++) {
			const pixelValue = image[i * width + j];
			const char = pixelValue > 0.5 ? "#" : " ";
			output += char;
		}
		output += "\n";
	}
	console.log(output);
}

console.log("MNIST Example using a Multi-Layer Perceptron (MLP)");

// 1. Load MNIST Data
// Make sure you have the MNIST IDX files in the 'examples/mnist-dataset/' directory.
// You can use the MNIST-fetch.sh script if it downloads IDX files, or find them online.
// Common file names:
// train-images-idx3-ubyte
// train-labels-idx1-ubyte
// t10k-images-idx3-ubyte
// t10k-labels-idx1-ubyte

const TRAIN_IMAGES_PATH = "./examples/mnist-dataset/train-images.idx3-ubyte";
const TRAIN_LABELS_PATH = "./examples/mnist-dataset/train-labels.idx1-ubyte";
const TEST_IMAGES_PATH = "./examples/mnist-dataset/t10k-images.idx3-ubyte";
const TEST_LABELS_PATH = "./examples/mnist-dataset/t10k-labels.idx1-ubyte";

let trainingData: number[][];
let trainingLabels: number[][];
let testData: number[][];
let testLabels: number[][];

try {
	console.log("Loading MNIST training data...");
	trainingData = loadMnistImages(TRAIN_IMAGES_PATH);
	trainingLabels = loadMnistLabels(TRAIN_LABELS_PATH, MNIST_NUM_CLASSES);
	console.log(
		`Loaded ${trainingData.length} training images and ${trainingLabels.length} training labels.`,
	);

	console.log("Loading MNIST test data...");
	testData = loadMnistImages(TEST_IMAGES_PATH);
	testLabels = loadMnistLabels(TEST_LABELS_PATH, MNIST_NUM_CLASSES);
	console.log(
		`Loaded ${testData.length} test images and ${testLabels.length} test labels.`,
	);
} catch (error) {
	console.error(
		"Error loading MNIST data. Make sure the IDX files are in the 'examples/mnist-dataset/' directory and are not corrupted.",
		error,
	);
	process.exit(1); // Exit if data loading fails
}

// For demonstration, let's use a small subset of the data to speed up training.
// Remove or adjust these lines to use the full dataset.
const subsetSizeTrain = 1000; // e.g., 1000 training samples
const subsetSizeTest = 200; // e.g., 200 test samples

if (trainingData.length > subsetSizeTrain && testData.length > subsetSizeTest) {
	console.warn(
		`Using a subset of data: ${subsetSizeTrain} for training, ${subsetSizeTest} for testing.`,
	);
	trainingData = trainingData.slice(0, subsetSizeTrain);
	trainingLabels = trainingLabels.slice(0, subsetSizeTrain);
	testData = testData.slice(0, subsetSizeTest);
	testLabels = testLabels.slice(0, subsetSizeTest);
} else {
	console.warn(
		"Dataset size is smaller than subset sizes, using full loaded data.",
	);
}

// 2. Define the Model (MLP)
const model = new Model();
model.addLayer(new Dense(MNIST_IMAGE_SIZE, 128)); // Input layer (28*28=784 features) to hidden layer (128 neurons)
model.addLayer(new ReLU());
model.addLayer(new Dense(128, MNIST_NUM_CLASSES)); // Hidden layer to output layer (10 classes for digits 0-9)
model.addLayer(new Softmax()); // Softmax for multi-class probability output

// 3. Compile the Model
model.compile(
	new Adam(0.001), // Adam optimizer
	new CrossEntropyLoss(), // Cross-entropy loss for multi-class classification
	["accuracy"], // Metric
);

// 4. Train the Model
console.log("Starting model training...");
const epochs = 5; // Adjust epochs as needed
const batchSize = 32; // Adjust batch size based on memory and performance
await model.fit(trainingData, trainingLabels, epochs, batchSize, true); // Enable debugEpochEnabled

// 5. Evaluate the Model
if (testData.length > 0 && testLabels.length > 0) {
	console.log("\nEvaluating model...");
	const evaluation = model.evaluate(testData, testLabels);
	console.log(
		"Model Evaluation (Note: built-in accuracy in model.evaluate is for binary classification):",
		evaluation,
	);

	// Manual accuracy calculation for multi-class
	const predictions = model.predict(testData);
	let correctPredictions = 0;
	for (let i = 0; i < predictions.length; i++) {
		const predictedLabel = predictions[i].indexOf(Math.max(...predictions[i]));
		const actualLabel = testLabels[i].indexOf(1);
		if (predictedLabel === actualLabel) {
			correctPredictions++;
		}
	}
	const multiClassAccuracy = correctPredictions / testData.length;
	console.log(
		`Multi-class Accuracy on Test Set: ${multiClassAccuracy.toFixed(4)}`,
	);
}

// 6. Use the Model for Predictions (Example for a few test samples)
if (testData.length > 0) {
	console.log("\nExample predictions on test data (first 10 samples):");
	const numSamplesToDisplay = Math.min(10, testData.length);
	const samplePredictions = model.predict(
		testData.slice(0, numSamplesToDisplay),
	);

	for (let i = 0; i < numSamplesToDisplay; i++) {
		const prediction = samplePredictions[i];
		const predictedLabel = prediction.indexOf(Math.max(...prediction));
		const actualLabel = testLabels[i].indexOf(1); // Get the index of the '1' in one-hot encoded label

		console.log(
			`Test Sample ${i + 1}: Predicted Label: ${predictedLabel}, Actual Label: ${actualLabel}`,
		);
		displayImageInTerminal(testData[i]);
	}
}

console.log("\nMNIST MLP example finished.");
