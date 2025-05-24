import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { BinaryCrossEntropyLoss } from "./binary_cross_entropy.ts";

describe("BinaryCrossEntropyLoss", () => {
	it("should calculate the binary cross-entropy loss correctly", () => {
		const predictions = [0.9, 0.2, 0.8];
		const targets = [1, 0, 1];
		const binaryCrossEntropy = new BinaryCrossEntropyLoss();
		const loss = binaryCrossEntropy.calculate(predictions, targets);

		assert.strictEqual(loss, 0.18388253942874858);
	});

	it("should throw an error for different length arrays", () => {
		const predictions = [0.9, 0.2];
		const targets = [1, 0, 1];
		const binaryCrossEntropy = new BinaryCrossEntropyLoss();

		assert.throws(
			() => {
				binaryCrossEntropy.calculate(predictions, targets);
			},
			{
				message: "Predictions and targets must have the same length.",
			},
		);
	});

	it("should return 0 for empty arrays", () => {
		const predictions: number[] = [];
		const targets: number[] = [];
		const binaryCrossEntropy = new BinaryCrossEntropyLoss();
		const loss = binaryCrossEntropy.calculate(predictions, targets);

		assert.strictEqual(loss, 0);
	});

	it("name should be accessible", () => {
		const binaryCrossEntropy = new BinaryCrossEntropyLoss();

		assert.strictEqual(binaryCrossEntropy.name, "BinaryCrossEntropyLoss");
	});
});
