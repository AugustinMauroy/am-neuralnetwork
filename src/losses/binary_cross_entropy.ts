/**
 * BinaryCrossEntropyLoss calculates the binary cross-entropy loss between predictions and target values.
 * This loss is commonly used for binary classification tasks.
 *
 * Formula:
 * `L = -Σ(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))`
 *
 * @example
 * ```typescript
 * const binaryCrossEntropy = new BinaryCrossEntropyLoss();
 * const predictions = [0.9, 0.2, 0.8];
 * const targets = [1, 0, 1];
 * const loss = binaryCrossEntropy.calculate(predictions, targets);
 * console.log("Binary CrossEntropy Loss:", loss); // Output: ~0.1839
 * ```
 */
export class BinaryCrossEntropyLoss {
	/**
	 * Calculates the binary cross-entropy loss.
	 * @param predictions An array of predicted probabilities (values between 0 and 1).
	 * @param targets An array of binary target values (0 or 1).
	 * @returns The calculated binary cross-entropy loss.
	 * @throws Error if the predictions and targets arrays do not have the same length.
	 */
	calculate(predictions: number[], targets: number[]): number {
		if (predictions.length !== targets.length) {
			throw new Error("Predictions and targets must have the same length.");
		}

		if (predictions.length === 0 || targets.length === 0) {
			return 0; // Return 0 for empty arrays
		}

		const epsilon = 1e-12; // To avoid log(0)
		let loss = 0;

		for (let i = 0; i < predictions.length; i++) {
			const yTrue = targets[i];
			const yPred = Math.min(Math.max(predictions[i], epsilon), 1 - epsilon); // Clamp predictions to avoid log(0)
			loss -= yTrue * Math.log(yPred) + (1 - yTrue) * Math.log(1 - yPred);
		}

		return loss / predictions.length; // Average loss
	}
}
