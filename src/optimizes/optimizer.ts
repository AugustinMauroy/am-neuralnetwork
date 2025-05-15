/**
 * Base class for all optimizers.
 * It primarily handles the learning rate.
 */
export class Optimizer {
	/** The learning rate for the optimizer. */
	learningRate: number;

	/**
	 * Creates an instance of the Optimizer.
	 * @param learningRate The learning rate to be used for updating weights. Defaults to `0.01`.
	 */
	constructor(learningRate = 0.01) {
		this.learningRate = learningRate;
	}
}
