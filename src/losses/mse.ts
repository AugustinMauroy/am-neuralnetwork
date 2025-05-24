import { Loss } from "./loss.ts";

/**
 * Calculates the Mean Squared Error (MSE) between predictions and target values.
 * MSE is defined as the average of the squared differences between predicted and actual values.
 * Formula: `MSE = (1/n) * Σ(prediction_i - target_i)^2`
 *
 * @example
 * ```typescript
 * const mse = new MeanSquaredError();
 * const predictions = [1, 2, 3];
 * const targets = [1.1, 1.9, 3.2];
 * const loss = mse.calculate(predictions, targets);
 * console.log("MSE Loss:", loss); // MSE Loss: 0.03
 * ```
 */
export class MeanSquaredError extends Loss { 
	public readonly name = "MeanSquaredError";

	/**
	 * Calculates the MSE loss.
	 * @param predictions An array of predicted numerical values.
	 * @param targets An array of actual numerical values (ground truth).
	 * @returns The calculated Mean Squared Error.
	 * @throws Error if the predictions and targets arrays do not have the same length.
	 */
	calculate(predictions: number[], targets: number[]): number {
		if (predictions.length !== targets.length) {
			throw new Error("Predictions and targets must have the same length.");
		}
		if (predictions.length === 0) {
			return 0; // Or throw an error, depending on desired behavior for empty inputs
		}

		let sum = 0;
		for (let i = 0; i < predictions.length; i++) {
			const error = predictions[i] - targets[i];
			sum += error * error;
		}

		return sum / predictions.length;
	}
}
