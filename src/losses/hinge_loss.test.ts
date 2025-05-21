import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { HingeLoss } from "./hinge_loss.ts";

describe("HingeLoss", () => {
	const hingeLoss = new HingeLoss();

	it("should return 0 when predictions are correctly classified with sufficient margin", () => {
		const predictions = [1.5, -1.2, 2.0];
		const targets = [1, -1, 1];
		// 1 - 1*1.5 = -0.5 -> 0
		// 1 - (-1)*(-1.2) = 1 - 1.2 = -0.2 -> 0
		// 1 - 1*2.0 = -1.0 -> 0
		assert.strictEqual(hingeLoss.calculate(predictions, targets), 0);
	});

	it("should calculate hinge loss correctly", () => {
		const predictions = [0.5, -0.3, 0.8];
		const targets = [1, -1, 1];
		// 1 - 1*0.5 = 0.5
		// 1 - (-1)*(-0.3) = 1 - 0.3 = 0.7
		// 1 - 1*0.8 = 0.2
		const expectedLoss = (0.5 + 0.7 + 0.2) / 3;
		assert.strictEqual(hingeLoss.calculate(predictions, targets), expectedLoss);
	});

	it("should calculate hinge loss correctly when some predictions are on the margin", () => {
		const predictions = [1.0, -1.0, 0.5];
		const targets = [1, -1, -1];
		// 1 - 1*1.0 = 0
		// 1 - (-1)*(-1.0) = 0
		// 1 - (-1)*0.5 = 1 + 0.5 = 1.5
		const expectedLoss = (0 + 0 + 1.5) / 3;
		assert.strictEqual(hingeLoss.calculate(predictions, targets), expectedLoss);
	});

	it("should throw an error for different length arrays", () => {
		const predictions = [0.5, 0.2];
		const targets = [1, -1, 1];
		assert.throws(
			() => {
				hingeLoss.calculate(predictions, targets);
			},
			{
				message: "Predictions and targets must have the same length.",
			},
		);
	});

	it("should return 0 for empty arrays", () => {
		assert.strictEqual(hingeLoss.calculate([], []), 0);
	});

	it("should throw an error if targets are not -1 or 1", () => {
		const predictions = [0.5, 0.2];
		const targets = [1, 0];
		assert.throws(
			() => {
				hingeLoss.calculate(predictions, targets);
			},
			{
				message: "Target labels for Hinge Loss must be 1 or -1.",
			},
		);
	});
});
