export class SGD {
  private learningRate: number;

  constructor(learningRate: number = 0.01) {
    this.learningRate = learningRate;
  }

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
