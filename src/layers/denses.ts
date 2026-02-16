import type { Layer, TrainableLayer } from "../core/model.ts";

/**
 * Represents a fully connected (dense) layer in a neural network.
 * Implements both {@link Layer} and {@link TrainableLayer} interfaces.
 *
 * @example
 * ```typescript
 * const denseLayer = new Dense(784, 128); // A dense layer with 784 inputs and 128 outputs
 * const input = [[...Array(784).keys()].map(x => x / 784)]; // Example batch of 1 sample
 * const output = denseLayer.forward(input);
 * ```
 */
export class Dense implements TrainableLayer {
	/** @hidden The number of input units to this layer. */
	private inputUnits: number;
	/** @hidden The number of output units (neurons) in this layer. */
	private outputUnits: number;
	/** @hidden The weights matrix of the layer. Shape: [inputUnits, outputUnits] */
	private weights: number[][];
	/** @hidden The biases vector of the layer. Shape: [outputUnits] */
	private biases: number[];
	/** @hidden Stores the input from the last forward pass, needed for backpropagation. */
	//private lastInput!: number[][];

	/**
	 * Creates an instance of a Dense layer.
	 * @param inputUnits The number of input features.
	 * @param outputUnits The number of neurons in this layer.
	 */
	constructor(inputUnits: number, outputUnits: number) {
		this.inputUnits = inputUnits;
		this.outputUnits = outputUnits;
		this.weights = this.initializeWeights();
		this.biases = this.initializeBiases();
	}

	/**
	 * Returns the number of input units.
	 * @returns The number of input units.
	 */
	public getInputUnits(): number {
		return this.inputUnits;
	}

	/**
	 * Returns the number of output units.
	 * @returns The number of output units.
	 */
	public getOutputUnits(): number {
		return this.outputUnits;
	}

	/**
	 * Initializes the weights for the layer.
	 * Weights are initialized with random values between 0 and 1.
	 * @hidden
	 * @returns A 2D array representing the weights.
	 */
	private initializeWeights(): number[][] {
		const weights = [];
		// W_ij is the weight from input unit i to output unit j
		for (let i = 0; i < this.inputUnits; i++) {
			weights[i] = Array(this.outputUnits)
				.fill(0)
				.map(() => Math.random() * 0.1 - 0.05); // Small random numbers
		}
		return weights;
	}

	/**
	 * Initializes the biases for the layer.
	 * Biases are initialized with random values between 0 and 1.
	 * @hidden
	 * @returns A 1D array representing the biases.
	 */
	private initializeBiases(): number[] {
		return Array(this.outputUnits)
			.fill(0)
			.map(() => Math.random() * 0.1 - 0.05); // Small random numbers
	}

	/**
	 * Performs the forward pass through the dense layer.
	 * Output = Input * Weights + Biases
	 * @param input The input data to the layer. Shape: [batchSize, inputUnits]
	 * @returns The output data from the layer. Shape: [batchSize, outputUnits]
	 */
	public forward(input: number[][]): number[][] {
		this.lastInput = input; // Save input for backpropagation
		const output: number[][] = [];
		const batchSize = input.length;

		for (let b = 0; b < batchSize; b++) {
			// For each sample in the batch
			const currentSampleOutput: number[] = [];
			for (let j = 0; j < this.outputUnits; j++) {
				// For each output neuron
				let weightedSum = this.biases[j];
				for (let i = 0; i < this.inputUnits; i++) {
					// For each input feature
					weightedSum += input[b][i] * this.weights[i][j];
				}
				currentSampleOutput.push(weightedSum);
			}
			output.push(currentSampleOutput);
		}
		return output;
	}

	/**
	 * Performs the backward pass (backpropagation) through the dense layer.
	 * Calculates the gradient of the loss with respect to the layer's input.
	 * @param outputGradient The gradient of the loss with respect to the output of this layer (dL/dY).
	 *                       Shape: [batchSize, outputUnits]
	 * @returns The gradient of the loss with respect to the input of this layer (dL/dX).
	 *          Shape: [batchSize, inputUnits]
	 */
	public backward(outputGradient: number[][]): number[][] {
		const batchSize = outputGradient.length;
		if (batchSize === 0) {
			return [];
		}
		if (outputGradient[0].length !== this.outputUnits) {
			throw new Error(
				`Output gradient dimension mismatch. Expected ${this.outputUnits}, got ${outputGradient[0].length}`,
			);
		}

		const inputGradient: number[][] = Array(batchSize)
			.fill(null)
			.map(() => Array(this.inputUnits).fill(0));

		// dL/dX_i = sum_j (dL/dY_j * W_ij)
		// For each sample in the batch
		for (let b = 0; b < batchSize; b++) {
			// For each input unit of this layer (which will be an output to the previous layer)
			for (let i = 0; i < this.inputUnits; i++) {
				let gradSum = 0;
				// Sum over all output units of this layer
				for (let j = 0; j < this.outputUnits; j++) {
					gradSum += outputGradient[b][j] * this.weights[i][j];
				}
				inputGradient[b][i] = gradSum;
			}
		}
		return inputGradient;
	}

	/**
	 * Retrieves the current weights and biases of the layer.
	 * @returns A map containing 'weights' and 'biases'.
	 */
	public getWeights(): Map<string, number[] | number[][]> {
		return new Map<string, number[] | number[][]>([
			["weights", this.weights.map((row) => [...row])], // Deep copy
			["biases", [...this.biases]], // Deep copy
		]);
	}

	/**
	 * Calculates the gradients of the loss with respect to the layer's weights and biases.
	 * dL/dW_ij = dL/dY_j * X_i
	 * dL/dB_j = dL/dY_j
	 * @param outputGradient The gradient of the loss with respect to the output of this layer (dL/dY).
	 *                       Shape: [batchSize, outputUnits]
	 * @param layerInput The input data that was fed into this layer during the forward pass (this.lastInput).
	 *                   Shape: [batchSize, inputUnits]
	 * @returns A map containing the gradients for 'weights' and 'biases'.
	 */
	public getWeightGradients(
		outputGradient: number[][],
		layerInput: number[][], // Typically this.lastInput
	): Map<string, number[] | number[][]> {
		const batchSize = outputGradient.length;
		if (batchSize === 0) {
			return new Map([
				[
					"weights",
					Array(this.inputUnits)
						.fill(null)
						.map(() => Array(this.outputUnits).fill(0)),
				],
				["biases", Array(this.outputUnits).fill(0)],
			]);
		}
		if (
			outputGradient[0].length !== this.outputUnits ||
			layerInput[0].length !== this.inputUnits ||
			layerInput.length !== batchSize
		) {
			throw new Error("Dimension mismatch in getWeightGradients.");
		}

		const dLdW: number[][] = Array(this.inputUnits)
			.fill(null)
			.map(() => Array(this.outputUnits).fill(0));
		const dLdB: number[] = Array(this.outputUnits).fill(0);

		// Accumulate gradients over the batch
		for (let b = 0; b < batchSize; b++) {
			for (let j = 0; j < this.outputUnits; j++) {
				// For each output unit
				dLdB[j] += outputGradient[b][j]; // Gradient for bias_j
				for (let i = 0; i < this.inputUnits; i++) {
					// For each input unit
					dLdW[i][j] += layerInput[b][i] * outputGradient[b][j]; // Gradient for weight_ij
				}
			}
		}

		// Average gradients over the batch
		for (let j = 0; j < this.outputUnits; j++) {
			dLdB[j] /= batchSize;
			for (let i = 0; i < this.inputUnits; i++) {
				dLdW[i][j] /= batchSize;
			}
		}

		return new Map<string, number[] | number[][]>([
			["weights", dLdW],
			["biases", dLdB],
		]);
	}

	/**
	 * Updates the layer's weights and biases.
	 * @param updatedWeights A map containing the new 'weights' and 'biases' to apply.
	 */
	public updateWeights(
		updatedWeights: Map<string, number[] | number[][]>,
	): void {
		const newWeights = updatedWeights.get("weights") as number[][];
		const newBiases = updatedWeights.get("biases") as number[];

		if (
			newWeights &&
			newWeights.length === this.inputUnits &&
			newWeights[0].length === this.outputUnits
		) {
			this.weights = newWeights;
		} else {
			console.warn(
				"Dense layer: Mismatch in new weights dimensions or new weights not provided.",
			);
		}

		if (newBiases && newBiases.length === this.outputUnits) {
			this.biases = newBiases;
		} else {
			console.warn(
				"Dense layer: Mismatch in new biases dimensions or new biases not provided.",
			);
		}
	}

	/**
	 * Returns the name of the layer.
	 * @returns The name of the layer type.
	 */
	getName(): string {
		return "Dense";
	}

	/**
	 * Returns the configuration of the layer.
	 * @returns An object containing the layer type and its parameters.
	 */
	getConfig(): Record<string, unknown> {
		return {
			name: this.getName(),
			inputUnits: this.inputUnits,
			outputUnits: this.outputUnits,
		};
	}
}
