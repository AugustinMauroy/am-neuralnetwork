import type { MeanSquaredError } from "../losses/mse.ts";
import type { Optimizer } from "../optimizes/optimizer.ts";

/**
 * Represents a single layer in a neural network.
 */
export interface Layer {
	/**
	 * Performs the forward pass of the layer.
	 * @param input The input data to the layer.
	 * @returns The output data from the layer.
	 */
	forward(input: number[][]): number[][];

	/**
	 * Performs the backward pass of the layer (backpropagation).
	 * @param outputGradient The gradient of the loss with respect to the output of this layer.
	 * @returns The gradient of the loss with respect to the input of this layer.
	 * Signature might need adjustment for specific layer needs.
	 */
	backward(outputGradient: number[][]): number[][];
}

/**
 * Represents a layer that has trainable weights and biases.
 * Extends the base {@link Layer} interface.
 */
export interface TrainableLayer extends Layer {
	/**
	 * Retrieves the current weights and biases of the layer.
	 * @returns A map where keys are 'weights' or 'biases' and values are the corresponding matrices or vectors.
	 * Example: weights and biases
	 */
	getWeights(): Map<string, number[] | number[][]>;

	/**
	 * Calculates the gradients of the loss with respect to the layer's weights and biases.
	 * @param outputGradient The gradient of the loss with respect to the output of this layer.
	 * @param layerInput The input data that was fed into this layer during the forward pass.
	 * @returns A map containing the gradients for weights and biases.
	 * Gradients for weights and biases
	 */
	getWeightGradients(
		outputGradient: number[][],
		layerInput: number[][],
	): Map<string, number[] | number[][]>;

	/**
	 * Updates the layer's weights and biases.
	 * @param updatedWeights A map containing the new weights and biases to apply.
	 */
	updateWeights(updatedWeights: Map<string, number[] | number[][]>): void;
}

/**
 * Represents a neural network model composed of a sequence of layers.
 */
export class Model {
	/** @hidden The sequence of layers in the model. */
	private layers: Layer[] = [];
	/** @hidden The optimizer used for training the model. */
	private optimizer!: Optimizer;
	/** @hidden The loss function used to evaluate the model's performance. */
	private lossFunction!: MeanSquaredError; // Assuming MeanSquaredError for now
	/** @hidden A list of metrics to evaluate during training and testing. */
	private metrics: string[] = [];

	/**
	 * Adds a layer to the model.
	 * @param layer The layer to add to the model.
	 */
	public addLayer(layer: Layer): void {
		this.layers.push(layer);
	}

	/**
	 * Configures the model for training.
	 * @param optimizer The optimizer to use for training.
	 * @param lossFunction The loss function to use.
	 * @param metrics A list of metrics to evaluate.
	 */
	public compile(
		optimizer: Optimizer,
		lossFunction: MeanSquaredError,
		metrics: string[],
	): void {
		this.optimizer = optimizer;
		this.lossFunction = lossFunction;
		this.metrics = metrics;
	}

	/**
	 * Trains the model for a fixed number of epochs (iterations on a dataset).
	 *
	 * @param trainingData The input data for training.
	 * @param trainingLabels The target labels for training.
	 * @param epochs The number of epochs to train the model.
	 * @param batchSize The number of samples per gradient update.
	 * @param debugEpochEnabled Whether to log loss information after each epoch. Defaults to false.
	 * @returns A promise that resolves when training is complete.
	 *
	 * @example
	 * ```typescript
	 * const model = new Model();
	 * // ... add layers ...
	 * model.compile(new AdamOptimizer(), new MeanSquaredError(), ['accuracy']);
	 * await model.fit(trainingData, trainingLabels, 10, 32);
	 * ```
	 */
	public async fit(
		trainingData: number[][],
		trainingLabels: number[][],
		epochs: number,
		batchSize: number,
		debugEpochEnabled = false,
	): Promise<void> {
		if (!this.optimizer || !this.lossFunction) {
			throw new Error("Model must be compiled before training.");
		}
		if (trainingData.length !== trainingLabels.length) {
			throw new Error(
				"Training data and labels must have the same number of samples.",
			);
		}

		const numSamples = trainingData.length;

		for (let epoch = 0; epoch < epochs; epoch++) {
			const epochLoss = 0;
			// Shuffle training data and labels at the beginning of each epoch
			for (let k = numSamples - 1; k > 0; k--) {
				const j = Math.floor(Math.random() * (k + 1));
				[trainingData[k], trainingData[j]] = [trainingData[j], trainingData[k]];
				[trainingLabels[k], trainingLabels[j]] = [
					trainingLabels[j],
					trainingLabels[k],
				];
			}

			for (let i = 0; i < numSamples; i += batchSize) {
				const batchInputs = trainingData.slice(
					i,
					Math.min(i + batchSize, numSamples),
				);
				const batchTargets = trainingLabels.slice(
					i,
					Math.min(i + batchSize, numSamples),
				);

				if (batchInputs.length === 0) continue;

				// 1. Forward pass - store intermediate outputs for backpropagation
				const layerOutputs: number[][][] = [];
				let currentBatchOutput = batchInputs;
				layerOutputs.push(currentBatchOutput); // Store initial input

				for (const layer of this.layers) {
					currentBatchOutput = layer.forward(currentBatchOutput);
					layerOutputs.push(currentBatchOutput);
				}
				const finalPredictions = currentBatchOutput;

				// 2. Calculate loss
				const dLdOutput: number[][] = finalPredictions.map((predRow, rIdx) =>
					predRow.map(
						(predVal, cIdx) =>
							(2 * (predVal - batchTargets[rIdx][cIdx])) / predRow.length,
					),
				);

				// 3. Backward pass
				let currentGradient = dLdOutput;
				for (let j = this.layers.length - 1; j >= 0; j--) {
					const layer = this.layers[j];
					const layerInput = layerOutputs[j]; // Input that went into this layer
					currentGradient = layer.backward(currentGradient);
				}

				// Conceptual optimizer step - this requires layers to expose weights and gradients
				// TODO: Implement weight update logic for trainable layers:
				// 1. Iterate through layers to identify trainable ones
				// 2. Get current weights and gradients
				// 3. Use optimizer to calculate weight updates
				// 4. Apply updated weights back to the layers
			}
			if (debugEpochEnabled) {
				console.log(
					`Epoch ${epoch + 1}/${epochs}, Loss: ${
						epochLoss / (numSamples / batchSize)
					}`,
				);
			}
			// TODO: Calculate and log metrics if any
		}
	}

	/**
	 * Generates output predictions for the input samples.
	 * @param inputData The input data for which to generate predictions.
	 * @returns The model's predictions.
	 */
	public predict(inputData: number[][]): number[][] {
		let currentOutput = inputData;
		for (const layer of this.layers) {
			currentOutput = layer.forward(currentOutput);
		}
		return currentOutput;
	}

	/**
	 * Evaluates the model on a validation dataset.
	 * @param validationData The input data for validation.
	 * @param validationLabels The target labels for validation.
	 * @returns An object containing the loss and any configured metrics (e.g., accuracy).
	 */
	public evaluate(
		validationData: number[][],
		validationLabels: number[][],
	): { loss: number; accuracy?: number } {
		if (!this.lossFunction) {
			throw new Error("Model must be compiled before evaluation.");
		}
		if (validationData.length !== validationLabels.length) {
			throw new Error(
				"Validation data and labels must have the same number of samples.",
			);
		}

		const predictions = this.predict(validationData);
		let totalLoss = 0;
		for (let i = 0; i < predictions.length; i++) {
			totalLoss += this.lossFunction.calculate(
				predictions[i],
				validationLabels[i],
			);
		}
		const avgLoss = totalLoss / predictions.length;

		const results: { loss: number; accuracy?: number } = { loss: avgLoss };

		if (this.metrics.includes("accuracy")) {
			let correctPredictions = 0;
			for (let i = 0; i < predictions.length; i++) {
				const predictedValue = predictions[i][0] > 0.5 ? 1 : 0; // Assuming single binary output
				const targetValue = validationLabels[i][0];
				if (predictedValue === targetValue) {
					correctPredictions++;
				}
			}
			results.accuracy = correctPredictions / predictions.length;
		}

		return results;
	}

	/**
	 * Saves the model's architecture, weights, and optimizer state.
	 * Note: This method is not fully implemented.
	 * @param filePath The path where the model will be saved.
	 */
	public save(_filePath: string): void {
		console.warn("Model.save() is not fully implemented.");
	}

	/**
	 * Loads a model from a file.
	 * Note: This method is not fully implemented and returns a new empty model.
	 * @param _filePath The path from which to load the model.
	 * @returns A new {@link Model} instance (currently empty).
	 */
	public static load(_filePath: string): Model {
		console.warn(
			"Model.load() is not fully implemented and returns a new empty model.",
		);

		return new Model();
	}
}

export default Model;
