import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { HuberLoss } from "./huber_loss.ts";

describe("HuberLoss", () => {
	it("should return 0 when predictions are perfect", () => {
		const yTrue = [1, 2, 3];
		const yPred = [1, 2, 3];
		const huber = new HuberLoss(1.0);
		const result = huber.calculate(yPred, yTrue);
		assert.strictEqual(result, 0);
	});

	it("should behave like MSE for small errors (|error| <= delta)", () => {
		const yTrue = [1, 2, 3];
		const yPred = [1.1, 1.2, 2.9];
		const huber = new HuberLoss(1.0);
		const result = huber.calculate(yPred, yTrue);
		// Approximate MSE since all differences < delta
		const expected = (0.5 * 0.1 ** 2 + 0.5 * 0.8 ** 2 + 0.5 * 0.1 ** 2) / 3;
		assert.ok(Math.abs(result - expected) < 1e-10);
	});

	it("should behave like MAE for large errors (|error| > delta)", () => {
		const yTrue = [0];
		const yPred = [3];
		const delta = 1.0;
		const huber = new HuberLoss(delta);
		const result = huber.calculate(yPred, yTrue);
		const expected = delta * (Math.abs(3 - 0) - 0.5 * delta); // 1 * (3 - 0.5)
		assert.strictEqual(result, expected);
	});

	it("should throw an error for arrays of different lengths", () => {
		const yTrue = [1, 2];
		const yPred = [1, 2, 3];
		const huber = new HuberLoss(1.0);
		assert.throws(
			() => {
				huber.calculate(yPred, yTrue);
			},
			{
				message: "Predictions and targets must have the same length.",
			},
		);
	});

	it("should return 0 for empty arrays", () => {
		const huber = new HuberLoss(1.0);
		const result = huber.calculate([], []);
		assert.strictEqual(result, 0);
	});
});
