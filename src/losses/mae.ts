/**
 * Calculates the Mean Absolute Error (MAE) between predictions and target values.
 * MAE is defined as the average of the absolute differences between predicted and actual values.
 * Formula: `MAE = (1/n) * Σ|prediction_i - target_i|`
 *
 * @example
 * ```typescript
 * const mae = new MeanAbsoluteError();
 * const predictions = [1, 2, 3];
 * const targets = [1.1, 1.9, 3.2];
 * const loss = mae.calculate(predictions, targets);
 * console.log("MAE Loss:", loss); // MAE Loss: 0.13333333333333344
 * ```
 */
export class MeanAbsoluteError {
    /**
     * Calculates the MAE loss.
     * @param predictions An array of predicted numerical values.
     * @param targets An array of actual numerical values (ground truth).
     * @returns The calculated Mean Absolute Error.
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
            const error = Math.abs(predictions[i] - targets[i]);
            sum += error;
        }

        return sum / predictions.length;
    }
}
