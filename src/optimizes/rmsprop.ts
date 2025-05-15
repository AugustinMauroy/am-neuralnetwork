import { Optimizer } from "./optimizer.ts";

/**
 * Implements the RMSprop (Root Mean Square Propagation) optimization algorithm.
 *
 * RMSprop is an adaptive learning rate method that maintains a moving average
 * of the square of gradients, and divides the gradient by the root of this average.
 * This helps to dampen oscillations in directions with high gradients and accelerate
 * learning in directions with low gradients.
 */
export class RMSprop extends Optimizer {
  private decayRate: number;
  private epsilon: number;
  private cache: Map<string, number>;

  /**
   * Creates an instance of the RMSprop optimizer.
   * @param learningRate The learning rate for the optimizer. Defaults to 0.001.
   * @param decayRate The decay rate for the moving average of squared gradients. Defaults to 0.9.
   * @param epsilon A small constant to prevent division by zero. Defaults to 1e-8.
   */
  constructor(
    learningRate: number = 0.001,
    decayRate: number = 0.9,
    epsilon: number = 1e-8,
  ) {
    super(learningRate);
    this.decayRate = decayRate;
    this.epsilon = epsilon;
    this.cache = new Map();
  }

  /**
   * Updates the weights based on the gradients using the RMSprop algorithm.
   *
   * @param weights A map representing the current weights of the model.
   * @param gradients A map representing the gradients of the loss function with respect to the weights.
   * @returns A new map with the updated weights.
   *
   * @example
   * ```ts
   * const optimizer = new RMSprop();
   * const weights = new Map([['w1', 0.5], ['w2', -0.2]]);
   * const gradients = new Map([['w1', 0.1], ['w2', 0.05]]);
   * const updatedWeights = optimizer.update(weights, gradients);
   * console.log(updatedWeights);
   * ```
   */
  public update(
    weights: Map<string, number>,
    gradients: Map<string, number>,
  ): Map<string, number> {
    const updatedWeights = new Map(weights);

    gradients.forEach((gradient, key) => {
      const cachedValue = this.cache.get(key) || 0;
      const newCache = this.decayRate * cachedValue +
        (1 - this.decayRate) * gradient * gradient;
      this.cache.set(key, newCache);

      const currentWeight = updatedWeights.get(key) || 0;
      const weightUpdate = this.learningRate * gradient /
        (Math.sqrt(newCache) + this.epsilon);
      updatedWeights.set(key, currentWeight - weightUpdate);
    });

    return updatedWeights;
  }
}
