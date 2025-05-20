/**
 * CrossEntropyLoss calculates the cross-entropy loss between predictions and target values.
 * This loss is commonly used for classification tasks.
 *
 * Formula:
 * `L = -Σ(y_true * log(y_pred))`
 *
 * @example
 * ```typescript
 * const crossEntropy = new CrossEntropyLoss();
 * const predictions = [0.7, 0.2, 0.1];
 * const targets = [1, 0, 0];
 * const loss = crossEntropy.calculate(predictions, targets);
 * console.log("CrossEntropy Loss:", loss); // Output: ~0.3567
 * ```
 */
export class CrossEntropyLoss {
	/**
	 * Calculates the cross-entropy loss.
	 * @param predictions An array of predicted probabilities (must sum to 1).
	 * @param targets An array of one-hot encoded target values.
	 * @returns The calculated cross-entropy loss, summed across all samples.
	 * @throws Error if the predictions and targets arrays do not have the same length.
	 */
	calculate(predictions: number[], targets: number[]): number {
		if (predictions.length !== targets.length) {
			throw new Error("Predictions and targets must have the same length.");
		}

		let loss = 0;
		for (let i = 0; i < predictions.length; i++) {
			if (targets[i] === 1) {
				// Avoid log(0) by adding a small epsilon
				const epsilon = 1e-12;
				loss -= Math.log(predictions[i] + epsilon);
			}
		}

		// Normalize the loss by the number of samples
		return loss / predictions.length;
	}
}
