/**
 * Calculates the Hinge Loss between predictions and targets.
 * Commonly used for "maximum-margin" classification, most notably for support vector machines (SVMs).
 * For a predicted score `y` and a true label `t` (either -1 or 1):
 * L = max(0, 1 - t * y)
 *
 * @example
 * ```typescript
 * const hinge = new HingeLoss();
 * const predictions = [0.5, -0.3, 1.2]; // Raw scores from a classifier
 * const targets = [1, -1, 1];          // True labels (-1 or 1)
 * const loss = hinge.calculate(predictions, targets);
 * console.log("Hinge Loss:", loss); // Hinge Loss: (max(0, 1-1*0.5) + max(0, 1-(-1)*(-0.3)) + max(0, 1-1*1.2)) / 3
 *                                  //             = (0.5 + max(0, 1-0.3) + max(0, 1-1.2)) / 3
 *                                  //             = (0.5 + 0.7 + 0) / 3 = 1.2 / 3 = 0.4
 * ```
 */
export class HingeLoss {
	/**
	 * Calculates the Hinge loss.
	 * @param predictions An array of predicted numerical scores.
	 * @param targets An array of actual numerical labels (expected to be -1 or 1).
	 * @returns The calculated Hinge Loss.
	 * @throws Error if the predictions and targets arrays do not have the same length.
	 * @throws Error if targets array contains values other than -1 or 1.
	 */
	calculate(predictions: number[], targets: number[]): number {
		if (predictions.length !== targets.length) {
			throw new Error("Predictions and targets must have the same length.");
		}
		if (predictions.length === 0) {
			return 0;
		}

		let sum = 0;
		for (let i = 0; i < predictions.length; i++) {
			if (targets[i] !== 1 && targets[i] !== -1) {
				throw new Error("Target labels for Hinge Loss must be 1 or -1.");
			}
			sum += Math.max(0, 1 - targets[i] * predictions[i]);
		}

		return sum / predictions.length;
	}
}
