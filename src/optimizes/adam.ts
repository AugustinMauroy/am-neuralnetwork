import { Optimizer } from "./optimizer.ts";

/**
 * Adam (Adaptive Moment Estimation) optimizer.
 * Adam is an optimization algorithm that can be used instead of the classical
 * stochastic gradient descent procedure to update network weights iteratively
 * based on training data.
 *
 * It combines the advantages of two other extensions of stochastic gradient
 * descent: Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp).
 */
export class Adam extends Optimizer {
	/** Exponential decay rate for the first moment estimates. */
	private beta1: number;
	/** Exponential decay rate for the second moment estimates. */
	private beta2: number;
	/** A small constant for numerical stability. */
	private epsilon: number;
	/** Stores the first moment vector (moving average of the gradients). */
	private m: Map<string, number>;
	/** Stores the second moment vector (moving average of the squared gradients). */
	private v: Map<string, number>;
	/** Timestep, used for bias correction. */
	private t: number;

	/**
	 * Creates an instance of the Adam optimizer.
	 * @param learningRate The learning rate. Defaults to `0.001`.
	 * @param beta1 The exponential decay rate for the first moment estimates. Defaults to `0.9`.
	 * @param beta2 The exponential decay rate for the second moment estimates. Defaults to `0.999`.
	 * @param epsilon A small constant for numerical stability. Defaults to `1e-8`.
	 */
	constructor(
		learningRate = 0.001,
		beta1 = 0.9,
		beta2 = 0.999,
		epsilon = 1e-8,
	) {
		super(learningRate);
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.epsilon = epsilon;
		this.m = new Map();
		this.v = new Map();
		this.t = 0;
	}

	/**
	 * Updates the weights based on the gradients using the Adam optimization algorithm.
	 * @param weights A map representing the current weights of the model, where keys are parameter names and values are their current values.
	 * @param gradients A map representing the gradients of the loss with respect to the weights, with the same structure as `weights`.
	 * @returns A new map with the updated weights.
	 */
	public update(
		weights: Map<string, number>,
		gradients: Map<string, number>,
	): Map<string, number> {
		this.t += 1;

		const updatedWeights = new Map(weights);

		gradients.forEach((gradient, key) => {
			const mKey = this.m.get(key) || 0;
			const vKey = this.v.get(key) || 0;

			// Update biased first moment estimate
			const newM = this.beta1 * mKey + (1 - this.beta1) * gradient;
			// Update biased second raw moment estimate
			const newV = this.beta2 * vKey + (1 - this.beta2) * gradient * gradient;

			this.m.set(key, newM);
			this.v.set(key, newV);

			// Compute bias-corrected first moment estimate
			const mHat = newM / (1 - this.beta1 ** this.t);
			// Compute bias-corrected second raw moment estimate
			const vHat = newV / (1 - this.beta2 ** this.t);

			// Update parameters
			const weightUpdate =
				(this.learningRate * mHat) / (Math.sqrt(vHat) + this.epsilon);
			updatedWeights.set(key, (updatedWeights.get(key) || 0) - weightUpdate);
		});

		return updatedWeights;
	}

	/**
	 * Returns the name of the optimizer.
	 * @returns The name of the optimizer, which is "Adam".
	 */
	public getName(): string {
		return "Adam";
	}

	/**
	 * Returns the configuration of the Adam optimizer.
	 * @returns An object containing the name, learning rate, beta1, beta2, and epsilon.
	 */
	public getConfig(): Record<string, unknown> {
		return {
			name: this.getName(),
			learningRate: this.learningRate,
			beta1: this.beta1,
			beta2: this.beta2,
			epsilon: this.epsilon,
		};
	}
}
