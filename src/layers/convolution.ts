/**
 * A basic 2D convolution layer that performs a convolution operation on input data.
 * @remarks This class is intended as a demonstration and does not include optimizations.
 * @param kernelSize - The height and width of the filter.
 * @param filters - The number of output channels (filters).
 * @param stride - The stride for the convolution.
 * @param padding - Zero-padding around the input.
 */
export class Convolution {
	private kernelSize: number;
	private filters: number;
	private stride: number;
	private padding: number;
	private lastInput: number[][][][] = [];
	private weights: number[][][][] = [];
	private biases: number[] = [];

	constructor(kernelSize = 3, filters = 8, stride = 1, padding = 0) {
		this.kernelSize = kernelSize;
		this.filters = filters;
		this.stride = stride;
		this.padding = padding;
		// Initialize weights as [filters][inChannels][kernelSize][kernelSize], biases as [filters]
		// Actual shape is set once we know inChannels from first forward.
	}

	/**
	 * Forward pass for this convolution layer.
	 * @param input - A 4D array [batchSize][inChannels][height][width].
	 * @returns The convolved output as a 4D array [batchSize][filters][newHeight][newWidth].
	 */
	forward(input: number[][][][]): number[][][][] {
		this.lastInput = input;
		const batchSize = input.length;
		const inChannels = input[0].length;
		const inHeight = input[0][0].length;
		const inWidth = input[0][0][0].length;

		// Lazy weight initialization if needed
		if (this.weights.length === 0) {
			this.weights = Array.from({ length: this.filters }, () =>
				Array.from({ length: inChannels }, () =>
					Array.from({ length: this.kernelSize }, () =>
						Array(this.kernelSize).fill(0.01),
					),
				),
			);
			this.biases = Array(this.filters).fill(0);
		}

		// Compute output dimensions
		const outHeight = Math.floor(
			(inHeight + 2 * this.padding - this.kernelSize) / this.stride + 1,
		);
		const outWidth = Math.floor(
			(inWidth + 2 * this.padding - this.kernelSize) / this.stride + 1,
		);

		// Allocate output
		const output: number[][][][] = Array.from({ length: batchSize }, () =>
			Array.from({ length: this.filters }, () =>
				Array.from({ length: outHeight }, () => Array(outWidth).fill(0)),
			),
		);

		// Naive convolution
		for (let b = 0; b < batchSize; b++) {
			for (let f = 0; f < this.filters; f++) {
				for (let oy = 0; oy < outHeight; oy++) {
					for (let ox = 0; ox < outWidth; ox++) {
						let sum = this.biases[f];
						for (let c = 0; c < inChannels; c++) {
							for (let ky = 0; ky < this.kernelSize; ky++) {
								for (let kx = 0; kx < this.kernelSize; kx++) {
									const iy = oy * this.stride + ky - this.padding;
									const ix = ox * this.stride + kx - this.padding;
									if (iy >= 0 && iy < inHeight && ix >= 0 && ix < inWidth) {
										sum += input[b][c][iy][ix] * this.weights[f][c][ky][kx];
									}
								}
							}
						}
						output[b][f][oy][ox] = sum;
					}
				}
			}
		}

		return output;
	}

	/**
	 * Backward pass for this convolution layer.
	 * @param gradient - The gradient flowing back into this layer.
	 * @returns The gradient of the loss with respect to the input.
	 */
	backward(gradient: number[][][][]): number[][][][] {
		if (!this.lastInput.length) {
			return gradient;
		}

		const batchSize = this.lastInput.length;
		const inChannels = this.lastInput[0].length;
		const inHeight = this.lastInput[0][0].length;
		const inWidth = this.lastInput[0][0][0].length;

		const inputGrad: number[][][][] = Array.from({ length: batchSize }, () =>
			Array.from({ length: inChannels }, () =>
				Array.from({ length: inHeight }, () => Array(inWidth).fill(0)),
			),
		);

		// Compute grads for weights, biases, and input
		for (let b = 0; b < batchSize; b++) {
			for (let f = 0; f < this.filters; f++) {
				for (let oy = 0; oy < gradient[b][f].length; oy++) {
					for (let ox = 0; ox < gradient[b][f][oy].length; ox++) {
						const gradVal = gradient[b][f][oy][ox];
						this.biases[f] += gradVal; // dBias
						for (let c = 0; c < inChannels; c++) {
							for (let ky = 0; ky < this.kernelSize; ky++) {
								for (let kx = 0; kx < this.kernelSize; kx++) {
									const iy = oy * this.stride + ky - this.padding;
									const ix = ox * this.stride + kx - this.padding;
									if (iy >= 0 && iy < inHeight && ix >= 0 && ix < inWidth) {
										// dWeight
										this.weights[f][c][ky][kx] +=
											this.lastInput[b][c][iy][ix] * gradVal;
										// dInput
										inputGrad[b][c][iy][ix] +=
											this.weights[f][c][ky][kx] * gradVal;
									}
								}
							}
						}
					}
				}
			}
		}

		return inputGrad;
	}

	getOutputShape(inputShape: number[]): number[] {
		// Compute output shape based on kernelSize, stride, padding, and filters
		const [batchSize, inChannels, inHeight, inWidth] = inputShape;
		const outHeight = Math.floor(
			(inHeight + 2 * this.padding - this.kernelSize) / this.stride + 1,
		);
		const outWidth = Math.floor(
			(inWidth + 2 * this.padding - this.kernelSize) / this.stride + 1,
		);
		return [batchSize, this.filters, outHeight, outWidth];
	}

	getInputShape(): number[] {
		// Returns the input shape from the last forward pass, if available
		if (this.lastInput.length > 0) {
			const batchSize = this.lastInput.length;
			const inChannels = this.lastInput[0].length;
			const inHeight = this.lastInput[0][0].length;
			const inWidth = this.lastInput[0][0][0].length;

			return [batchSize, inChannels, inHeight, inWidth];
		}

		return [0, 0, 0, 0];
	}

	getName(): string {
		return "Convolution";
	}

	getConfig(): Record<string, unknown> {
		return {
			kernelSize: this.kernelSize,
			filters: this.filters,
			stride: this.stride,
			padding: this.padding,
		};
	}
}
