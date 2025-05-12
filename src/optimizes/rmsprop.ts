export class RMSprop {
  private learningRate: number;
  private decayRate: number;
  private epsilon: number;
  private cache: Map<string, number>;

  constructor(
    learningRate: number = 0.001,
    decayRate: number = 0.9,
    epsilon: number = 1e-8,
  ) {
    this.learningRate = learningRate;
    this.decayRate = decayRate;
    this.epsilon = epsilon;
    this.cache = new Map();
  }

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
