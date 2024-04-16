export class MathUtils {
  public static sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  public static tanh(x: number): number {
    return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
  }
}
