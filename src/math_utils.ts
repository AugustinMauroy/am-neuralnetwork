/**
 * MathUtils class
 *
 * This class contains some useful math functions.
 *
 * @class MathUtils
 * @example
 * MathUtils.sigmoid(0); // 0.5
 * MathUtils.tanh(0); // 0
 */
export class MathUtils {
	/**
	 * JS function that calculates the sigmoid of a number.
	 */
	public static sigmoid(x: number): number {
		return 1 / (1 + Math.exp(-x));
	}

	/**
	 * JS function that calculates the hyperbolic tangent of a number.
	 */
	public static tanh(x: number): number {
		return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
	}
}
