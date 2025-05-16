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
}
