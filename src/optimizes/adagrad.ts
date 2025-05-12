export class Adagrad {
  private learningRate: number;
  private epsilon: number;
  private accumulatedSquaredGradients: Map<string, number>;

  constructor(learningRate: number = 0.01, epsilon: number = 1e-8) {
    this.learningRate = learningRate;
    this.epsilon = epsilon;
    this.accumulatedSquaredGradients = new Map();
  }

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
      const weightUpdate = this.learningRate * gradient /
        (Math.sqrt(accumulated) + this.epsilon);
      updatedWeights.set(key, currentWeight - weightUpdate);
    });

    return updatedWeights;
  }
}
