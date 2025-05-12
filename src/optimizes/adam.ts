export class Adam {
  private learningRate: number;
  private beta1: number;
  private beta2: number;
  private epsilon: number;
  private m: Map<string, number>;
  private v: Map<string, number>;
  private t: number;

  constructor(
    learningRate: number = 0.001,
    beta1: number = 0.9,
    beta2: number = 0.999,
    epsilon: number = 1e-8,
  ) {
    this.learningRate = learningRate;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    this.m = new Map();
    this.v = new Map();
    this.t = 0;
  }

  public update(
    weights: Map<string, number>,
    gradients: Map<string, number>,
  ): Map<string, number> {
    this.t += 1;

    const updatedWeights = new Map(weights);

    gradients.forEach((gradient, key) => {
      const mKey = this.m.get(key) || 0;
      const vKey = this.v.get(key) || 0;

      const newM = this.beta1 * mKey + (1 - this.beta1) * gradient;
      const newV = this.beta2 * vKey + (1 - this.beta2) * gradient * gradient;

      this.m.set(key, newM);
      this.v.set(key, newV);

      const mHat = newM / (1 - Math.pow(this.beta1, this.t));
      const vHat = newV / (1 - Math.pow(this.beta2, this.t));

      const weightUpdate = this.learningRate * mHat /
        (Math.sqrt(vHat) + this.epsilon);
      updatedWeights.set(key, updatedWeights.get(key)! - weightUpdate);
    });

    return updatedWeights;
  }
}
