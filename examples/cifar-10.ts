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
const CIFAR_RECORD_SIZE = 1 + CIFAR_IMAGE_SIZE; // 1 byte for label, 3072 for image

// Function to load a single CIFAR-10 batch file
function loadCifarBatch(
	filePath: string,
	numClasses: number,
): { images: number[][]; labels: number[][] } {
	const buffer = readFileSync(filePath);
	const numRecords = buffer.length / CIFAR_RECORD_SIZE;

	if (buffer.length % CIFAR_RECORD_SIZE !== 0) {
		console.warn(
			`[loadCifarBatch] Warning: File "${filePath}" size (${buffer.length}) is not a multiple of record size (${CIFAR_RECORD_SIZE}). File might be corrupted or incomplete. Expected records: ${buffer.length / CIFAR_RECORD_SIZE}`,
		);
	}

	const images: number[][] = [];
	const labels: number[][] = [];

	for (let i = 0; i < numRecords; i++) {
		const offset = i * CIFAR_RECORD_SIZE;

		// Extract label
		const label = buffer.readUInt8(offset);
		const oneHotLabel = Array(numClasses).fill(0);
		if (label < numClasses) {
			oneHotLabel[label] = 1;
		} else {
			console.warn(
				`[loadCifarBatch] Warning: Label ${label} out of bounds for numClasses ${numClasses} in file ${filePath}`,
			);
		}
		labels.push(oneHotLabel);

		// Extract image data
		const imageData = new Array(CIFAR_IMAGE_SIZE);
		for (let j = 0; j < CIFAR_IMAGE_SIZE; j++) {
			// Normalize pixel values to [0, 1]
			imageData[j] = buffer.readUInt8(offset + 1 + j) / 255.0;
		}
		images.push(imageData);
	}
	return { images, labels };
}

// Function to load multiple CIFAR-10 batch files (for training data)
function loadCifarDataset(
	batchFilePaths: string[],
	numClasses: number,
): { images: number[][]; labels: number[][] } {
	let allImages: number[][] = [];
	let allLabels: number[][] = [];

	for (const filePath of batchFilePaths) {
		console.log(`Loading batch file: ${filePath}...`);
		try {
			const batch = loadCifarBatch(filePath, numClasses);
			allImages = allImages.concat(batch.images);
			allLabels = allLabels.concat(batch.labels);
			console.log(
				`Loaded ${batch.images.length} images and ${batch.labels.length} labels from ${filePath}.`,
			);
		} catch (e) {
			console.error(`Error loading batch file ${filePath}:`, e);
			throw e;
		}
	}
	return { images: allImages, labels: allLabels };
}

// Function to load label names from batches.meta.txt
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
		// Fallback label names
		return Array.from({ length: CIFAR_NUM_CLASSES }, (_, i) => `Class ${i}`);
	}
}

function displayImageInTerminal(
    data: number[],
    width: number,
    height: number
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

console.log("CIFAR-10 Example using a Multi-Layer Perceptron (MLP)");
console.log(
	"Note: MLPs have limitations for image data. CNNs are preferred for better accuracy.",
);
console.log(
	"This example might be slow and memory-intensive due to the dataset size and MLP structure.",
);

// 1. Load CIFAR-10 Data
const BATCHES_META_PATH = "./examples/cifar-10-batches-bin/batches.meta.txt";
const DATA_BATCH_PATHS = [
	"./examples/cifar-10-batches-bin/data_batch_1.bin",
	"./examples/cifar-10-batches-bin/data_batch_2.bin",
	"./examples/cifar-10-batches-bin/data_batch_3.bin",
	"./examples/cifar-10-batches-bin/data_batch_4.bin",
	"./examples/cifar-10-batches-bin/data_batch_5.bin",
];
const TEST_BATCH_PATH = "./examples/cifar-10-batches-bin/test_batch.bin";

let trainingData: number[][];
let trainingLabels: number[][];
let testData: number[][];
let testLabels: number[][];
let labelNames: string[];

try {
	console.log("Loading CIFAR-10 label names...");
	labelNames = loadLabelNames(BATCHES_META_PATH);
	if (labelNames.length !== CIFAR_NUM_CLASSES) {
		console.warn(
			`Expected ${CIFAR_NUM_CLASSES} label names, but found ${labelNames.length}. Using provided names.`,
		);
	}
	console.log(`Loaded label names: ${labelNames.join(", ")}`);

	console.log("Loading CIFAR-10 training data...");
	const trainDataset = loadCifarDataset(DATA_BATCH_PATHS, CIFAR_NUM_CLASSES);
	trainingData = trainDataset.images;
	trainingLabels = trainDataset.labels;
	console.log(
		`Loaded ${trainingData.length} total training images and ${trainingLabels.length} total training labels.`,
	);

	console.log("Loading CIFAR-10 test data...");
	// Test data is a single batch file, so pass it as an array to loadCifarDataset
	const testDataset = loadCifarDataset([TEST_BATCH_PATH], CIFAR_NUM_CLASSES);
	testData = testDataset.images;
	testLabels = testDataset.labels;
	console.log(
		`Loaded ${testData.length} total test images and ${testLabels.length} total test labels.`,
	);
} catch (error) {
	console.error(
		"Error loading CIFAR-10 data. Make sure the .bin files and batches.meta.txt are in the 'example/cifar-10/' directory and are not corrupted.",
		error,
	);
	process.exit(1);
}

// For demonstration, let's use a small subset of the data to speed up training.
// Remove or adjust these lines to use the full dataset.
const subsetSizeTrain = 500; // e.g., 1000 training samples
const subsetSizeTest = 100; // e.g., 200 test samples

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
const epochs = 5; // Adjust epochs as needed
const batchSize = 32; // Adjust batch size based on memory and performance
await model.fit(trainingData, trainingLabels, epochs, batchSize, true); // Enable debugEpochEnabled

// 5. Evaluate the Model
if (testData.length > 0 && testLabels.length > 0) {
	console.log("\nEvaluating model...");
	const evaluation = model.evaluate(testData, testLabels);
	console.log(
		"Model Evaluation (conceptual, built-in accuracy is for binary):",
		evaluation,
	);
}

// 6. Use the Model for Predictions
for (let i = 0; i < 10; i++) {
	const prediction = model.predict([testData[i]]);
	const predictedLabel = prediction[0].indexOf(Math.max(...prediction[0]));
	const expectedLabel = testLabels[i].indexOf(Math.max(...testLabels[i]));

	console.log(`Displaying test image ${i + 1}:`);
	console.log(
		`Predicted label: ${predictedLabel} (${labelNames[predictedLabel]}), Expected label: ${expectedLabel} (${labelNames[expectedLabel]})`,
	);
	displayImageInTerminal(testData[i], CIFAR_IMAGE_WIDTH, CIFAR_IMAGE_HEIGHT);
}
