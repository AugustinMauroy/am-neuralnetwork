import { Optimizer } from "./optimizer.ts"; // Assuming Optimizer base class exists

/**
 * Adagrad (Adaptive Gradient Algorithm) optimizer.
 * Adagrad is an optimizer with parameter-specific learning rates,
 * which are adapted relative to how frequently a parameter gets updated during training.
 * The more a parameter receives updates, the smaller the updates will be.
 */
export class Adagrad extends Optimizer {
  /** A small constant for numerical stability. */
  private epsilon: number;
  /** Stores the sum of the squares of past gradients for each parameter. */
  private accumulatedSquaredGradients: Map<string, number>;

  /**
   * Creates an instance of the Adagrad optimizer.
   * @param learningRate The learning rate. Defaults to `0.01`.
   * @param epsilon A small constant for numerical stability. Defaults to `1e-8`.
   */
  constructor(learningRate: number = 0.01, epsilon: number = 1e-8) {
    super(learningRate);
    this.epsilon = epsilon;
    this.accumulatedSquaredGradients = new Map();
  }

  /**
   * Updates the weights based on the gradients using the Adagrad optimization algorithm.
   * @param weights A map representing the current weights of the model, where keys are parameter names and values are their current values.
   * @param gradients A map representing the gradients of the loss with respect to the weights, with the same structure as `weights`.
   * @returns A new map with the updated weights.
   */
  public update(
    weights: Map<string, number>,
    gradients: Map<string, number>,
  ): Map<string, number> {
    const updatedWeights = new Map(weights);

    gradients.forEach((gradient, key) => {
      const accumulated = (this.accumulatedSquaredGradients.get(key) || 0) +
        gradient * gradient;
      this.accumulatedSquaredGradients.set(key, accumulated);

      const currentWeight = updatedWeights.get(key) || 0;
      // Adagrad update rule
      const weightUpdate = this.learningRate * gradient /
        (Math.sqrt(accumulated) + this.epsilon);
      updatedWeights.set(key, currentWeight - weightUpdate);
    });

    return updatedWeights;
  }
}
