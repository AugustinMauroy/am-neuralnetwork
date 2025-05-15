import { Optimizer } from "./optimizer.ts";

/**
 * Implements the Stochastic Gradient Descent (SGD) optimization algorithm.
 *
 * SGD updates parameters by taking steps proportional to the negative of the gradient.
 * It's a foundational optimization algorithm used in training machine learning models.
 */
export class SGD extends Optimizer {
  /**
   * Creates an instance of the SGD optimizer.
   * @param learningRate The learning rate for the optimizer. Defaults to 0.01.
   */
  constructor(learningRate: number = 0.01) {
    super(learningRate);
  }

  /**
   * Updates the weights based on the gradients using the SGD algorithm.
   *
   * @param weights A map representing the current weights of the model.
   * @param gradients A map representing the gradients of the loss function with respect to the weights.
   * @returns A new map with the updated weights.
   *
   * @example
   * ```ts
   * const optimizer = new SGD(0.1);
   * const weights = new Map([['w1', 0.5], ['w2', -0.2]]);
   * const gradients = new Map([['w1', 0.1], ['w2', 0.05]]);
   * const updatedWeights = optimizer.update(weights, gradients);
   * console.log(updatedWeights);
   * // Expected output: Map(2) { 'w1' => 0.4, 'w2' => -0.25 }
   * ```
   */
  public update(
    weights: Map<string, number>,
    gradients: Map<string, number>,
  ): Map<string, number> {
    const updatedWeights = new Map(weights);

    gradients.forEach((gradient, key) => {
      const currentWeight = updatedWeights.get(key) || 0;
      updatedWeights.set(key, currentWeight - this.learningRate * gradient);
    });

    return updatedWeights;
  }
}
