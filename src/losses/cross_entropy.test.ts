import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { CrossEntropyLoss } from "./cross_entropy.ts";

describe("CrossEntropyLoss", () => {
	it("should calculate the correct loss for given predictions and targets", () => {
		const crossEntropy = new CrossEntropyLoss();
		const predictions = [0.7, 0.2, 0.1];
		const targets = [1, 0, 0];
		const loss = crossEntropy.calculate(predictions, targets);

		assert.strictEqual(loss, 0.1188916479791013);
	});

	it("should throw an error if predictions and targets have different lengths", () => {
		const crossEntropy = new CrossEntropyLoss();
		const predictions = [0.7, 0.2];
		const targets = [1, 0, 0];

		assert.throws(
			() => {
				crossEntropy.calculate(predictions, targets);
			},
			{
				message: "Predictions and targets must have the same length.",
			},
		);
	});

	it("name should be accessible", () => {
		const crossEntropy = new CrossEntropyLoss();

		assert.strictEqual(crossEntropy.name, "CrossEntropyLoss");
	});
});
