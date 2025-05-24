import type { Layer } from "../core/model.ts";

/**
 * Abstract base class for activation function layers.
 * All activation layers must implement the `forward` and `backward` methods.
 */
export abstract class Activation implements Layer {
	/**
	 * Performs the forward pass of the activation function.
	 * @param input The input data, a 2D array of numbers (batchSize x inputSize).
	 * @returns The output data after applying the activation function, a 2D array of numbers (batchSize x outputSize).
	 */
	abstract forward(input: number[][]): number[][];

	/**
	 * Performs the backward pass of the activation function.
	 * Calculates the gradient of the loss with respect to the input of this layer.
	 * @param outputGradient The gradient of the loss with respect to the output of this layer, a 2D array of numbers (batchSize x outputSize).
	 * @returns The gradient of the loss with respect to the input of this layer, a 2D array of numbers (batchSize x inputSize).
	 */
	abstract backward(outputGradient: number[][]): number[][];

	/**
	 * Returns the type of the layer.
	 * @returns A string representing the type of the layer.
	 */
	getName(): string {
		return this.constructor.name;
	}

	/**
	 * Returns the type of this layer.
	 * @returns The type of the layer, which is "activation".
	 */
	getConfig(): Record<string, unknown> {
		return {
			name: this.getName(),
		};
	}
}

/**
 * Sigmoid activation function.
 * Outputs values between 0 and 1.
 * Formula: `S(x) = 1 / (1 + e^(-x))`
 */
export class Sigmoid extends Activation {
	/** Stores the input from the forward pass, used during the backward pass. */
	private lastInput!: number[][];

	/**
	 * Applies the sigmoid function element-wise to the input.
	 * `S(x) = 1 / (1 + e^(-x))`
	 * @param input The input data (batchSize x inputSize).
	 * @returns The output data after applying sigmoid (batchSize x outputSize).
	 */
	forward(input: number[][]): number[][] {
		this.lastInput = input;
		return input.map((row) => row.map((x) => 1 / (1 + Math.exp(-x))));
	}

	/**
	 * Computes the gradient of the loss with respect to the input of the sigmoid layer.
	 * The derivative of sigmoid `S(x)` is `S(x) * (1 - S(x))`.
	 * @param outputGradient The gradient of the loss with respect to the output of this layer.
	 * @returns The gradient of the loss with respect to the input of this layer.
	 * @throws Error if the forward pass has not been called before the backward pass.
	 */
	backward(outputGradient: number[][]): number[][] {
		if (!this.lastInput) {
			throw new Error(
				"Forward pass must be called before backward pass for Sigmoid.",
			);
		}
		const sigmoidOutput = this.lastInput.map((row) =>
			row.map((x) => 1 / (1 + Math.exp(-x))),
		);
		return outputGradient.map((gradRow, i) =>
			gradRow.map((grad, j) => {
				const s = sigmoidOutput[i][j];
				return grad * s * (1 - s);
			}),
		);
	}
}

/**
 * Rectified Linear Unit (ReLU) activation function.
 * Outputs the input directly if it is positive, otherwise, it outputs zero.
 * Formula: `ReLU(x) = max(0, x)`
 */
export class ReLU extends Activation {
	/** Stores the input from the forward pass, used during the backward pass. */
	private lastInput!: number[][];

	/**
	 * Applies the ReLU function element-wise to the input.
	 * `ReLU(x) = max(0, x)`
	 * @param input The input data (batchSize x inputSize).
	 * @returns The output data after applying ReLU (batchSize x outputSize).
	 */
	forward(input: number[][]): number[][] {
		this.lastInput = input;
		return input.map((row) => row.map((x) => Math.max(0, x)));
	}

	/**
	 * Computes the gradient of the loss with respect to the input of the ReLU layer.
	 * The derivative of ReLU `ReLU(x)` is `1` if `x > 0`, and `0` otherwise.
	 * @param outputGradient The gradient of the loss with respect to the output of this layer.
	 * @returns The gradient of the loss with respect to the input of this layer.
	 * @throws Error if the forward pass has not been called before the backward pass.
	 */
	backward(outputGradient: number[][]): number[][] {
		if (!this.lastInput) {
			throw new Error(
				"Forward pass must be called before backward pass for ReLU.",
			);
		}
		return outputGradient.map((gradRow, i) =>
			gradRow.map((grad, j) => (this.lastInput[i][j] > 0 ? grad : 0)),
		);
	}
}

/**
 * Hyperbolic Tangent (Tanh) activation function.
 * Outputs values between -1 and 1.
 * Formula: `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
 */
export class Tanh extends Activation {
	/** Stores the input from the forward pass, used during the backward pass. */
	private lastInput!: number[][];

	/**
	 * Applies the Tanh function element-wise to the input.
	 * `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
	 * @param input The input data (batchSize x inputSize).
	 * @returns The output data after applying Tanh (batchSize x outputSize).
	 */
	forward(input: number[][]): number[][] {
		this.lastInput = input;
		return input.map((row) => row.map((x) => Math.tanh(x)));
	}

	/**
	 * Computes the gradient of the loss with respect to the input of the Tanh layer.
	 * The derivative of `tanh(x)` is `1 - tanh^2(x)`.
	 * @param outputGradient The gradient of the loss with respect to the output of this layer.
	 * @returns The gradient of the loss with respect to the input of this layer.
	 * @throws Error if the forward pass has not been called before the backward pass.
	 */
	backward(outputGradient: number[][]): number[][] {
		if (!this.lastInput) {
			throw new Error(
				"Forward pass must be called before backward pass for Tanh.",
			);
		}
		return outputGradient.map((gradRow, i) =>
			gradRow.map((grad, j) => {
				const tanhX = Math.tanh(this.lastInput[i][j]);
				return grad * (1 - tanhX * tanhX);
			}),
		);
	}
}

/**
 * Softmax activation function.
 * Typically used in the output layer of a multi-class classification network.
 * Converts a vector of K real numbers into a probability distribution of K possible outcomes.
 * Formula: `Softmax(x_i) = e^(x_i) / sum(e^(x_j))` for j = 1 to K.
 */
export class Softmax extends Activation {
	/** Stores the output of the forward pass, used during the backward pass. */
	private lastOutput!: number[][];

	/**
	 * Applies the Softmax function to each row of the input.
	 * Numerical stability is improved by subtracting the maximum value from each input row before exponentiation.
	 * @param input The input data (batchSize x numClasses).
	 * @returns The output data after applying Softmax, representing probabilities (batchSize x numClasses).
	 */
	forward(input: number[][]): number[][] {
		this.lastOutput = input.map((row) => {
			// Subtract max for numerical stability
			const maxVal = Math.max(...row);
			const exps = row.map((x) => Math.exp(x - maxVal));
			const sumExps = exps.reduce((sum, val) => sum + val, 0);
			return exps.map((exp) => exp / sumExps);
		});
		return this.lastOutput;
	}

	/**
	 * Computes the gradient of the loss with respect to the input of the Softmax layer.
	 * The Jacobian of the Softmax function is a bit more complex.
	 * If `i == j`, `dS_i / dZ_j = S_i * (1 - S_j)`.
	 * If `i != j`, `dS_i / dZ_j = -S_i * S_j`.
	 * This method computes `dL/dZ_j = sum_i (dL/dS_i * dS_i/dZ_j)`.
	 * For a single sample, `dL/dZ_j = S_j * (dL/dS_j - sum_k(dL/dS_k * S_k))`.
	 * @param outputGradient The gradient of the loss with respect to the output of this layer (dL/dS).
	 * @returns The gradient of the loss with respect to the input of this layer (dL/dZ).
	 * @throws Error if the forward pass has not been called before the backward pass.
	 * @throws Error if outputGradient and lastOutput batch sizes do not match.
	 * @throws Error if gradient row and softmax output row lengths do not match.
	 */
	backward(outputGradient: number[][]): number[][] {
		if (!this.lastOutput) {
			throw new Error(
				"Forward pass must be called before backward pass for Softmax.",
			);
		}
		if (outputGradient.length !== this.lastOutput.length) {
			throw new Error(
				"Output gradient and last output must have the same batch size.",
			);
		}

		return outputGradient.map((gradRow, i) => {
			const softmaxOutputRow = this.lastOutput[i];
			if (gradRow.length !== softmaxOutputRow.length) {
				throw new Error(
					"Gradient row and softmax output row must have the same number of units.",
				);
			}

			// Calculate the dot product of the output gradient and the softmax output for this sample
			// This is sum_k(dL/dS_k * S_k)
			const dotProduct = gradRow.reduce(
				(sum, gradVal, j) => sum + gradVal * softmaxOutputRow[j],
				0,
			);

			// Calculate the input gradient for this sample
			// dL/dZ_j = S_j * (dL/dS_j - sum_k(dL/dS_k * S_k))
			return softmaxOutputRow.map(
				(sVal, j) => sVal * (gradRow[j] - dotProduct),
			);
		});
	}
}
