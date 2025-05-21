/**
 * Calculates the Huber Loss between predictions and targets.
 * Huber Loss is less sensitive to outliers than MSE and behaves like MAE for large errors.
 *
 * @example
 * ```ts
 * const huber = new HuberLoss(1.0); // delta = 1.0
 * const predictions = [1, 2, 3];
 * const targets = [1.5, 1.8, 2.5];
 * const loss = huber.calculate(predictions, targets);
 * console.log("Huber Loss:", loss); // Huber Loss: ...
 * ```
 */
export class HuberLoss {
	private readonly delta: number;

	constructor(delta = 1.0) {
		this.delta = delta;
	}

	/**
	 * Calculates the Huber Loss.
	 * @param predictions Array of predicted numerical values.
	 * @param targets Array of actual numerical values.
	 * @returns The calculated Huber Loss.
	 * @throws Error if input arrays are not the same length.
	 */
	calculate(predictions: number[], targets: number[]): number {
		if (predictions.length !== targets.length) {
			throw new Error("Predictions and targets must have the same length.");
		}
		if (predictions.length === 0) {
			return 0;
		}

		let totalLoss = 0;

		for (let i = 0; i < predictions.length; i++) {
			const error = predictions[i] - targets[i];
			const absError = Math.abs(error);

			if (absError <= this.delta) {
				totalLoss += 0.5 * error * error;
			} else {
				totalLoss += this.delta * (absError - 0.5 * this.delta);
			}
		}

		return totalLoss / predictions.length;
	}
}
