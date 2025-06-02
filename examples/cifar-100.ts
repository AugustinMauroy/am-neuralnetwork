import { readFileSync } from "node:fs";

import { Model } from "../src/core/mod.ts";
import { Dense, ReLU, Softmax } from "../src/layers/mod.ts";
import { Adam } from "../src/optimizes/mod.ts";
import { CrossEntropyLoss } from "../src/losses/mod.ts";

const CIFAR100_IMAGE_HEIGHT = 32;
const CIFAR100_IMAGE_WIDTH = 32;
const CIFAR100_IMAGE_CHANNELS = 3;
const CIFAR100_IMAGE_SIZE =
	CIFAR100_IMAGE_HEIGHT * CIFAR100_IMAGE_WIDTH * CIFAR100_IMAGE_CHANNELS;
const CIFAR100_NUM_CLASSES = 100; // Renamed from CIFAR100_NUM_FINE_CLASSES
// const CIFAR100_NUM_COARSE_CLASSES = 20; // Coarse labels are also available

const CIFAR100_COARSE_LABEL_BYTES = 1;
const CIFAR100_FINE_LABEL_BYTES = 1;
// Each record: <1 x coarse label><1 x fine label><3072 x pixel>
const CIFAR100_RECORD_SIZE =
	CIFAR100_COARSE_LABEL_BYTES + CIFAR100_FINE_LABEL_BYTES + CIFAR100_IMAGE_SIZE;

// Function to load a CIFAR-100 data file (train.bin or test.bin)
function loadCifarFile(
	// Renamed from loadCifar100File
	filePath: string,
	numClasses: number,
): { images: number[][]; labels: number[][] } {
	const buffer = readFileSync(filePath);
	const numRecords = buffer.length / CIFAR100_RECORD_SIZE;

	if (buffer.length % CIFAR100_RECORD_SIZE !== 0) {
		console.warn(
			`[loadCifarFile] Warning: File "${filePath}" size (${buffer.length}) is not a multiple of record size (${CIFAR100_RECORD_SIZE}). File might be corrupted or incomplete. Expected records: ${
				buffer.length / CIFAR100_RECORD_SIZE
			}`,
		);
	}

	const images: number[][] = [];
	const labels: number[][] = []; // We'll use fine labels for this example

	for (let i = 0; i < numRecords; i++) {
		const offset = i * CIFAR100_RECORD_SIZE;

		// Extract fine label (second byte)
		const fineLabel = buffer.readUInt8(offset + 1);
		const oneHotLabel = Array(numClasses).fill(0);
		if (fineLabel < numClasses) {
			oneHotLabel[fineLabel] = 1;
		} else {
			console.warn(
				`[loadCifarFile] Warning: Fine label ${fineLabel} out of bounds for numClasses ${numClasses} in file ${filePath}`,
			);
		}
		labels.push(oneHotLabel);

		// Extract image data
		const imageData = new Array(CIFAR100_IMAGE_SIZE);
		for (let j = 0; j < CIFAR100_IMAGE_SIZE; j++) {
			// Normalize pixel values to [0, 1]
			imageData[j] = buffer.readUInt8(offset + 2 + j) / 255.0;
		}
		images.push(imageData);
	}
	return { images, labels };
}

// Function to load label names
function loadLabelNames(filePath: string): string[] {
	try {
		const content = readFileSync(filePath, "utf-8");
		return content
			.trim()
			.split("\n")
			.map((name) => name.trim())
			.filter((name) => name.length > 0);
	} catch (e) {
		console.error(`Error loading label names from ${filePath}:`, e);
		// Fallback label names, similar to cifar-10.ts
		return Array.from({ length: CIFAR100_NUM_CLASSES }, (_, i) => `Class ${i}`);
	}
}

function displayImageInTerminal(
	data: number[],
	width: number,
	height: number,
): void {
	console.log("\x1b[2J"); // Clear the terminal
	console.log("\x1b[0;0H"); // Move the cursor to the top-left corner

	for (let y = 0; y < height; y++) {
		let row = "";
		for (let x = 0; x < width; x++) {
			const r = Math.floor(data[y * width + x] * 255);
			const g = Math.floor(data[1024 + y * width + x] * 255);
			const b = Math.floor(data[2048 + y * width + x] * 255);
			row += `\x1b[48;2;${r};${g};${b}m  `;
		}
		row += "\x1b[0m"; // Reset the color
		console.log(row);
	}
	console.log("\x1b[0m\n"); // Reset the color and add a new line
}

console.log("CIFAR-100 Example using a Multi-Layer Perceptron (MLP)");
console.log(
	"Note: MLPs have limitations for image data. CNNs are preferred for better accuracy.",
);
console.log(
	"This example might be slow and memory-intensive due to the dataset size and MLP structure.",
);

// 1. Load CIFAR-100 Data
// Ensure you have train.bin, test.bin, fine_label_names.txt, and coarse_label_names.txt
// in the 'example/cifar-100/' directory.
const LABEL_NAMES_PATH = "./examples/cifar-100-binary/fine_label_names.txt"; // Renamed from FINE_LABEL_NAMES_PATH
// const COARSE_LABEL_NAMES_PATH = "./examples/cifar-100/coarse_label_names.txt";
const TRAIN_FILE_PATH = "./examples/cifar-100-binary/train.bin";
const TEST_FILE_PATH = "./examples/cifar-100-binary/test.bin";

let trainingData: number[][];
let trainingLabels: number[][];
let testData: number[][];
let testLabels: number[][];
let labelNames: string[]; // Renamed from fineLabelNames

try {
	console.log("Loading CIFAR-100 label names...");
	labelNames = loadLabelNames(LABEL_NAMES_PATH); // Use renamed variable
	if (
		labelNames.length !== CIFAR100_NUM_CLASSES && // Use renamed constant
		labelNames.length > 0
	) {
		console.warn(
			`Expected ${CIFAR100_NUM_CLASSES} label names, but found ${labelNames.length}. Using provided names.`, // Use renamed constant
		);
	} else if (labelNames.length === 0) {
		// Fallback already handled by loadLabelNames, this ensures we use the fallback if it was triggered
		console.warn(
			"No label names found or error during loading. Using generic names.",
		);
		// If loadLabelNames returned an empty array due to an error and then the fallback,
		// this re-assigns it. If it successfully loaded an empty file, it also assigns generic names.
		// The updated loadLabelNames should already provide the fallback.
		if (labelNames.length === 0) {
			// Check again if it's still empty after loadLabelNames's attempt
			labelNames = Array.from(
				{ length: CIFAR100_NUM_CLASSES }, // Use renamed constant
				(_, i) => `Class ${i}`,
			);
		}
	}
	console.log(`Loaded ${labelNames.length} label names.`); // Use renamed variable

	console.log("Loading CIFAR-100 training data...");
	const trainDataset = loadCifarFile(
		// Use renamed function
		TRAIN_FILE_PATH,
		CIFAR100_NUM_CLASSES, // Use renamed constant
	);
	trainingData = trainDataset.images;
	trainingLabels = trainDataset.labels;
	console.log(
		`Loaded ${trainingData.length} training images and ${trainingLabels.length} training labels.`,
	);

	console.log("Loading CIFAR-100 test data...");
	const testDataset = loadCifarFile(
		// Use renamed function
		TEST_FILE_PATH,
		CIFAR100_NUM_CLASSES, // Use renamed constant
	);
	testData = testDataset.images;
	testLabels = testDataset.labels;
	console.log(
		`Loaded ${testData.length} test images and ${testLabels.length} test labels.`,
	);
} catch (error) {
	console.error(
		"Error loading CIFAR-100 data. Make sure the .bin files and label name files are in the 'example/cifar-100/' directory and are not corrupted.",
		error,
	);
	process.exit(1);
}

// For demonstration, let's use a small subset of the data to speed up training.
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
model.addLayer(new Dense(CIFAR100_IMAGE_SIZE, 256)); // Input layer to hidden layer (e.g., 256 neurons)
model.addLayer(new ReLU());
model.addLayer(new Dense(256, 128)); // Second hidden layer
model.addLayer(new ReLU());
model.addLayer(new Dense(128, CIFAR100_NUM_CLASSES)); // Hidden layer to output layer (100 classes) // Use renamed constant
model.addLayer(new Softmax());

// 3. Compile the Model
model.compile(new Adam(0.001), new CrossEntropyLoss(), ["accuracy"]);

// 4. Train the Model
console.log("Starting model training...");
const epochs = 5; // Adjust epochs as needed, more epochs will take longer
const batchSize = 32; // Adjust batch size as needed
await model.fit(trainingData, trainingLabels, epochs, batchSize, true);

// 5. Evaluate the Model
if (testData.length > 0 && testLabels.length > 0) {
	console.log("\nEvaluating model...");
	const evaluation = model.evaluate(testData, testLabels);
	console.log("Model Evaluation:", evaluation);

	// Example of detailed prediction check for a few samples
	const numSamplesToCheck = Math.min(5, testData.length);
	if (numSamplesToCheck > 0) {
		console.log(
			`\nChecking predictions for the first ${numSamplesToCheck} test samples:`,
		);
		const predictions = model.predict(testData.slice(0, numSamplesToCheck));
		for (let i = 0; i < numSamplesToCheck; i++) {
			const predictedProbs = predictions[i];
			const actualLabelOneHot = testLabels[i];

			const predictedClassIndex = predictedProbs.indexOf(
				Math.max(...predictedProbs),
			);
			const actualClassIndex = actualLabelOneHot.indexOf(1);

			console.log(
				`Sample ${i + 1}: Predicted: ${labelNames[predictedClassIndex] || `Class ${predictedClassIndex}`}, Actual: ${labelNames[actualClassIndex] || `Class ${actualClassIndex}`}`, // Use renamed variable
			);
			displayImageInTerminal(
				testData[i],
				CIFAR100_IMAGE_WIDTH,
				CIFAR100_IMAGE_HEIGHT,
			);
		}
	}
} else {
	console.log("No test data to evaluate.");
}

console.log("\nCIFAR-100 MLP example finished.");
