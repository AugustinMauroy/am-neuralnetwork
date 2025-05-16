/**
 * This module provides a basic Tensor class for numerical operations,
 * primarily supporting 2D tensors (matrices).
 */
export class Tensor {
	/** @hidden The underlying 2D array storing the tensor's data. */
	private data: number[][];
	/** @hidden The dimensions of the tensor. For a 2D tensor, it's [rows, columns]. */
	private shape: number[];

	/**
	 * Creates a new Tensor instance.
	 * Currently supports 2D tensors (matrices).
	 * @param data A 2D array of numbers representing the tensor data.
	 * @example
	 * ```typescript
	 * const tensorA = new Tensor([[1, 2], [3, 4]]);
	 * ```
	 */
	constructor(data: number[][]) {
		if (!data || data.length === 0 || !Array.isArray(data[0])) {
			throw new Error("Input data must be a non-empty 2D array of numbers.");
		}
		this.data = data;
		this.shape = [data.length, data[0].length];
	}

	/**
	 * Performs element-wise addition with another tensor.
	 * The shapes of the two tensors must be identical.
	 * @param other The tensor to add to the current tensor.
	 * @returns A new Tensor representing the result of the addition.
	 * @throws Error if the tensor shapes do not match.
	 * @example
	 * ```typescript
	 * const a = new Tensor([[1, 2], [3, 4]]);
	 * const b = new Tensor([[5, 6], [7, 8]]);
	 * const sum = a.add(b); // Result: [[6, 8], [10, 12]]
	 * ```
	 */
	public add(other: Tensor): Tensor {
		if (this.shape[0] !== other.shape[0] || this.shape[1] !== other.shape[1]) {
			throw new Error("Shapes of tensors must match for addition.");
		}

		const resultData = this.data.map((row, i) =>
			row.map((value, j) => value + other.data[i][j]),
		);

		return new Tensor(resultData);
	}

	/**
	 * Performs matrix multiplication with another tensor.
	 * The number of columns in the first tensor must match the number of rows in the second tensor.
	 * @param other The tensor to multiply with the current tensor.
	 * @returns A new Tensor representing the result of the multiplication.
	 * @throws Error if the inner dimensions do not match for multiplication.
	 * @example
	 * ```typescript
	 * const a = new Tensor([[1, 2], [3, 4]]); // 2x2
	 * const b = new Tensor([[5, 6], [7, 8]]); // 2x2
	 * const product = a.multiply(b); // Result: [[19, 22], [43, 50]]
	 *
	 * const c = new Tensor([[1, 2, 3], [4, 5, 6]]); // 2x3
	 * const d = new Tensor([[7, 8], [9, 10], [11, 12]]); // 3x2
	 * const productCD = c.multiply(d); // Result: [[58, 64], [139, 154]]
	 * ```
	 */
	public multiply(other: Tensor): Tensor {
		if (this.shape[1] !== other.shape[0]) {
			throw new Error(
				`Inner dimensions must match for multiplication. Got ${this.shape[1]} and ${other.shape[0]}`,
			);
		}

		const resultData = Array.from({ length: this.shape[0] }, () =>
			Array(other.shape[1]).fill(0),
		);

		for (let i = 0; i < this.shape[0]; i++) {
			for (let j = 0; j < other.shape[1]; j++) {
				for (let k = 0; k < this.shape[1]; k++) {
					resultData[i][j] += this.data[i][k] * other.data[k][j];
				}
			}
		}

		return new Tensor(resultData);
	}

	/**
	 * Reshapes the tensor to a new shape.
	 * The total number of elements must remain the same.
	 * Currently only supports reshaping to another 2D shape.
	 * @param newShape An array representing the new dimensions (e.g., [rows, columns]).
	 * @returns A new Tensor with the specified shape.
	 * @throws Error if the total number of elements changes or if newShape is not for a 2D tensor.
	 * @example
	 * ```typescript
	 * const tensor = new Tensor([[1, 2, 3, 4], [5, 6, 7, 8]]); // 2x4
	 * const reshaped = tensor.reshape([4, 2]); // Reshapes to 4x2
	 * // reshaped.data will be [[1, 2], [3, 4], [5, 6], [7, 8]]
	 * ```
	 */
	public reshape(newShape: number[]): Tensor {
		const totalElements = this.shape.reduce((a, b) => a * b, 1);
		const newTotalElements = newShape.reduce((a, b) => a * b, 1);

		if (totalElements !== newTotalElements) {
			throw new Error("Total number of elements must remain the same.");
		}
		if (newShape.length !== 2) {
			// Assuming we are still working with 2D tensors primarily
			throw new Error("New shape must be a 2D shape [rows, columns].");
		}

		const flatData = this.data.flat();
		const resultData: number[][] = [];
		let index = 0;

		for (let i = 0; i < newShape[0]; i++) {
			const row: number[] = [];
			for (let j = 0; j < newShape[1]; j++) {
				row.push(flatData[index++]);
			}
			resultData.push(row);
		}

		return new Tensor(resultData);
	}

	/**
	 * Gets the shape of the tensor.
	 * @returns An array of numbers representing the dimensions of the tensor.
	 * For a 2D tensor, this will be [number of rows, number of columns].
	 * @example
	 * ```typescript
	 * const tensor = new Tensor([[1, 2, 3], [4, 5, 6]]);
	 * console.log(tensor.getShape()); // Output: [2, 3]
	 * ```
	 */
	public getShape(): number[] {
		return [...this.shape]; // Return a copy to prevent external modification
	}

	/**
	 * Gets the data of the tensor.
	 * @returns A 2D array of numbers representing the tensor data.
	 * Returns a copy of the internal data to prevent direct modification.
	 * @example
	 * ```typescript
	 * const tensor = new Tensor([[1, 2], [3, 4]]);
	 * console.log(tensor.getData()); // Output: [[1, 2], [3, 4]]
	 * ```
	 */
	public getData(): number[][] {
		// Return a deep copy to prevent external modification of internal data
		return this.data.map((row) => [...row]);
	}
}
