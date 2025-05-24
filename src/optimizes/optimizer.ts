/**
 * Base class for all optimizers.
 * It primarily handles the learning rate.
 */
export abstract class Optimizer {
	/** The learning rate for the optimizer. */
	learningRate: number;

	/**
	 * Creates an instance of the Optimizer.
	 * @param learningRate The learning rate to be used for updating weights. Defaults to `0.01`.
	 */
	constructor(learningRate = 0.01) {
		this.learningRate = learningRate;
	}

	/**
	 * Updates the weights of the model.
	 * This method must be implemented by subclasses.
	 * @param params The parameters (weights and biases) of the model to be updated.
	 * @param grads The gradients of the parameters.
	 */
	abstract update(
		params: Map<string, number>,
		grads: Map<string, number>,
	): unknown;

	/**
	 * Returns the name of the optimizer.
	 * This method must be implemented by subclasses.
	 * @returns The name of the optimizer.
	 */
	abstract getName(): string;

	/**
	 * Returns the configuration of the optimizer.
	 * This method provides the name and learning rate of the optimizer.
	 * @returns An object containing the optimizer's name and learning rate.
	 * */
	abstract getConfig(): Record<string, unknown>;
}
