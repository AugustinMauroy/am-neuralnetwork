/**
 * Base class for loss functions in machine learning.
 */
export abstract class Loss {
	/** The name of the loss function. */
	public abstract readonly name: string;

	/**
	 * Calculates the loss value based on predictions and targets.
	 * @param predictions An array of predicted numerical values.
	 * @param targets An array of actual numerical values (ground truth).
	 * @returns The calculated loss value.
	 * @throws Error if the predictions and targets arrays do not have the same length.
	 */
	public abstract calculate(predictions: number[], targets: number[]): number;

	/**
	 * Returns the configuration of the loss function, if available.
	 * @returns An object containing the configuration of the loss function, or null if not implemented.
	 */
	public getConfig?(): Record<string, unknown>;
}
