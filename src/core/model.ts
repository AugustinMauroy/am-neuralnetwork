import type { Loss } from "../losses/loss.ts";
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

	/**
	 * Retrieves the name of the layer.
	 * @returns The name of the layer.
	 */
	getName(): string;

	/**
	 * Gets the configuration of the layer.
	 * @returns A configuration object containing layer-specific parameters.
	 */
	getConfig(): Record<string, unknown>;
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
	private lossFunction!: Loss; 
	/** @hidden A list of metrics to evaluate during training and testing. */
	private metrics: string[] = [];

	/** @hidden Helper to check if a layer is trainable */
	private isTrainableLayer(layer: Layer): layer is TrainableLayer {
		return (
			typeof (layer as TrainableLayer).getWeights === "function" &&
			typeof (layer as TrainableLayer).getWeightGradients === "function" &&
			typeof (layer as TrainableLayer).updateWeights === "function"
		);
	}

	/**
	 * Adds a layer to the model.
	 * @param layer The layer to add to the model.
	 */
	public addLayer(layer: Layer): void {
		this.layers.push(layer);
	}

	/**
	 * Retrieves the layers of the model.
	 * @returns An array of layers in the model.
	 */
	public getLayers(): Layer[] {
		return this.layers;
	}

	/**
	 * Configures the model for training.
	 * @param optimizer The optimizer to use for training.
	 * @param lossFunction The loss function to use.
	 * @param metrics A list of metrics to evaluate.
	 */
	public compile(
		optimizer: Optimizer,
		lossFunction: Loss,
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
				let dLdOutput_currentLayer = dLdOutput; // Gradient of loss w.r.t. output of the final layer
				const batchWeightGradients: {
					layerIndex: number;
					gradients: Map<string, number[] | number[][]>;
				}[] = [];

				for (let j = this.layers.length - 1; j >= 0; j--) {
					const layer = this.layers[j];
					const layerInput = layerOutputs[j]; // Input that was fed to layers[j]

					if (this.isTrainableLayer(layer)) {
						// Calculate gradients of loss w.r.t. weights and biases of this layer
						const weightGradients = layer.getWeightGradients(
							dLdOutput_currentLayer,
							layerInput,
						);
						batchWeightGradients.unshift({
							layerIndex: j,
							gradients: weightGradients,
						});
					}

					// Propagate gradient to the previous layer
					dLdOutput_currentLayer = layer.backward(dLdOutput_currentLayer);
				}

				// 4. Update weights for trainable layers using the optimizer
				for (const {
					layerIndex,
					gradients: weightGradientsMap,
				} of batchWeightGradients) {
					const layer = this.layers[layerIndex] as TrainableLayer; // Known to be trainable
					const currentWeightsMap = layer.getWeights();
					const newWeightsMap = new Map<string, number[] | number[][]>();

					for (const [paramName, paramGradientsUntyped] of weightGradientsMap) {
						const paramCurrentWeightsUntyped = currentWeightsMap.get(paramName);

						if (!paramCurrentWeightsUntyped) {
							console.warn(
								`No current weights found for param ${paramName} in layer ${layerIndex}`,
							);
							continue;
						}

						if (
							Array.isArray(paramCurrentWeightsUntyped) &&
							Array.isArray(paramGradientsUntyped)
						) {
							if (
								paramCurrentWeightsUntyped.length > 0 &&
								typeof paramCurrentWeightsUntyped[0] === "number"
							) {
								// It's a 1D array (e.g., biases)
								const paramCurrentWeights =
									paramCurrentWeightsUntyped as number[];
								const paramGradients = paramGradientsUntyped as number[];
								const updatedParamValues: number[] = [];
								for (let k = 0; k < paramCurrentWeights.length; k++) {
									const weightKey = `layer${layerIndex}_${paramName}_${k}`;
									const tempWeightMap = new Map<string, number>([
										[weightKey, paramCurrentWeights[k]],
									]);
									const tempGradientMap = new Map<string, number>([
										[weightKey, paramGradients[k]],
									]);

									// Assuming optimizer.update handles state per unique key
									// Note: The Optimizer base class should define the 'update' method.
									// Using 'as any' if it's not yet defined on the base type.
									const updatedValMap = this.optimizer.update(
										tempWeightMap,
										tempGradientMap,
									);
									// @ts-ignore
									// biome-ignore lint/style/noNonNullAssertion : I know what I'm doing
									updatedParamValues.push(updatedValMap.get(weightKey)!);
								}
								newWeightsMap.set(paramName, updatedParamValues);
							} else if (
								paramCurrentWeightsUntyped.length > 0 &&
								Array.isArray(paramCurrentWeightsUntyped[0])
							) {
								// It's a 2D array (e.g., weights)
								const paramCurrentWeights =
									paramCurrentWeightsUntyped as number[][];
								const paramGradients = paramGradientsUntyped as number[][];
								const updatedParamValues: number[][] = [];
								for (let r = 0; r < paramCurrentWeights.length; r++) {
									const newRow: number[] = [];
									for (let c = 0; c < paramCurrentWeights[r].length; c++) {
										const weightKey = `layer${layerIndex}_${paramName}_${r}_${c}`;
										const tempWeightMap = new Map<string, number>([
											[weightKey, paramCurrentWeights[r][c]],
										]);
										const tempGradientMap = new Map<string, number>([
											[weightKey, paramGradients[r][c]],
										]);

										const updatedValMap = this.optimizer.update(
											tempWeightMap,
											tempGradientMap,
										);
										// @ts-ignore
										// biome-ignore lint/style/noNonNullAssertion : I know what I'm doing
										newRow.push(updatedValMap.get(weightKey)!);
									}
									updatedParamValues.push(newRow);
								}
								newWeightsMap.set(paramName, updatedParamValues);
							}
						}
					}
					layer.updateWeights(newWeightsMap);
				}
			}
			if (debugEpochEnabled) {
				console.log(
					`Epoch ${epoch + 1}/${epochs}, Loss: ${
						epochLoss / (numSamples / batchSize)
					}`,
				);
			}
			if (debugEpochEnabled && this.metrics.includes("accuracy")) {
				const predictions = this.predict(trainingData);
				let correctPredictions = 0;
				for (let i = 0; i < predictions.length; i++) {
					// Assuming binary classification and output is a single value
					const predictedValue = predictions[i][0] > 0.5 ? 1 : 0;
					const targetValue = trainingLabels[i][0];
					if (predictedValue === targetValue) {
						correctPredictions++;
					}
				}
				const accuracy = correctPredictions / predictions.length;
				console.log(
					`Epoch ${epoch + 1}/${epochs}, Accuracy: ${accuracy.toFixed(4)}`,
				);
			}
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
}

export default Model;
