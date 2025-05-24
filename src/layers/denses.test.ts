import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { Dense } from "./denses.ts";

describe("Dense Layer", () => {
	it("should initialize with correct dimensions and properties", () => {
		const inputUnits = 3;
		const outputUnits = 2;
		const layer = new Dense(inputUnits, outputUnits);

		assert.strictEqual(
			layer.getInputUnits(),
			inputUnits,
			"Input units should be set correctly",
		);
		assert.strictEqual(
			layer.getOutputUnits(),
			outputUnits,
			"Output units should be set correctly",
		);

		const weights = layer.getWeights().get("weights") as number[][];
		const biases = layer.getWeights().get("biases") as number[];

		assert.ok(weights, "Weights should be initialized");
		assert.strictEqual(
			weights.length,
			inputUnits,
			"Weights should have rows equal to inputUnits",
		);
		assert.strictEqual(
			weights[0].length,
			outputUnits,
			"Weights should have columns equal to outputUnits",
		);

		assert.ok(biases, "Biases should be initialized");
		assert.strictEqual(
			biases.length,
			outputUnits,
			"Biases should have length equal to outputUnits",
		);
	});

	it("should perform a forward pass correctly", () => {
		const layer = new Dense(2, 2);
		// Manually set weights and biases for predictable output
		const mockWeights = [
			[0.1, 0.2],
			[0.3, 0.4],
		];
		const mockBiases = [0.05, 0.15];
		layer.updateWeights(
			new Map<string, number[][] | number[]>([
				["weights", mockWeights],
				["biases", mockBiases],
			]),
		);

		const input = [[1, 2]]; // Batch size 1, 2 input features
		const output = layer.forward(input);

		// Expected output:
		// output[0][0] = input[0][0]*weights[0][0] + input[0][1]*weights[1][0] + biases[0]
		//              = 1*0.1 + 2*0.3 + 0.05 = 0.1 + 0.6 + 0.05 = 0.75
		// output[0][1] = input[0][0]*weights[0][1] + input[0][1]*weights[1][1] + biases[1]
		//              = 1*0.2 + 2*0.4 + 0.15 = 0.2 + 0.8 + 0.15 = 1.15
		assert.deepStrictEqual(output.length, 1, "Output batch size should be 1");
		assert.deepStrictEqual(
			output[0].length,
			2,
			"Output feature size should be 2",
		);
		assert.ok(
			Math.abs(output[0][0] - 0.75) < 1e-9,
			"Output[0][0] calculation is incorrect",
		);
		assert.ok(
			Math.abs(output[0][1] - 1.15) < 1e-9,
			"Output[0][1] calculation is incorrect",
		);

		assert.deepStrictEqual(
			// @ts-expect-error access private member for testing
			layer.lastInput,
			input,
			"lastInput should be stored",
		);
	});

	it("should perform a forward pass with batch input correctly", () => {
		const layer = new Dense(2, 1);
		const mockWeights = [[0.5], [0.5]];
		const mockBiases = [0.1];
		layer.updateWeights(
			new Map<string, number[][] | number[]>([
				["weights", mockWeights],
				["biases", mockBiases],
			]),
		);

		const input = [
			[1, 1],
			[2, 2],
		]; // Batch size 2
		const output = layer.forward(input);

		// Sample 1: 1*0.5 + 1*0.5 + 0.1 = 1.1
		// Sample 2: 2*0.5 + 2*0.5 + 0.1 = 2.1
		assert.strictEqual(output.length, 2);
		assert.ok(Math.abs(output[0][0] - 1.1) < 1e-9);
		assert.ok(Math.abs(output[1][0] - 2.1) < 1e-9);
	});

	it("getWeights should return deep copies of weights and biases", () => {
		const layer = new Dense(2, 2);
		const initialWeights = layer.getWeights();
		const weightsArray = initialWeights.get("weights") as number[][];
		const biasesArray = initialWeights.get("biases") as number[];

		weightsArray[0][0] = 999;
		biasesArray[0] = 888;

		const currentWeights = layer.getWeights();
		assert.notStrictEqual(
			(currentWeights.get("weights") as number[][])[0][0],
			999,
			"Modifying returned weights should not affect layer weights",
		);
		assert.notStrictEqual(
			(currentWeights.get("biases") as number[])[0],
			888,
			"Modifying returned biases should not affect layer biases",
		);
	});

	it("updateWeights should update layer weights and biases", () => {
		const layer = new Dense(1, 1);
		const newWeights = [[0.5]];
		const newBiases = [0.1];
		const updates = new Map<string, number[][] | number[]>([
			["weights", newWeights],
			["biases", newBiases],
		]);

		layer.updateWeights(updates);

		const currentParams = layer.getWeights();
		assert.deepStrictEqual(currentParams.get("weights"), newWeights);
		assert.deepStrictEqual(currentParams.get("biases"), newBiases);
	});

	it("updateWeights should handle dimension mismatches gracefully", () => {
		const layer = new Dense(2, 2);
		const originalWeights = JSON.parse(
			JSON.stringify(layer.getWeights().get("weights")),
		); // Deep copy
		const originalBiases = JSON.parse(
			JSON.stringify(layer.getWeights().get("biases")),
		); // Deep copy

		// Mismatched weights
		const wrongDimWeights = [[0.1]];
		layer.updateWeights(
			new Map<string, number[][] | number[]>([["weights", wrongDimWeights]]),
		);
		assert.deepStrictEqual(
			layer.getWeights().get("weights"),
			originalWeights,
			"Weights should not update with wrong dimensions",
		);

		// Mismatched biases
		const wrongDimBiases = [0.1, 0.2, 0.3];
		layer.updateWeights(
			new Map<string, number[][] | number[]>([["biases", wrongDimBiases]]),
		);
		assert.deepStrictEqual(
			layer.getWeights().get("biases"),
			originalBiases,
			"Biases should not update with wrong dimensions",
		);
	});

	it("backward pass should compute input gradients correctly", () => {
		const layer = new Dense(2, 2);
		// W = [[w11, w12], [w21, w22]]
		// B = [b1, b2]
		// Y = XW + B
		// dL/dX = dL/dY * W^T
		const mockWeights = [
			[0.1, 0.2], // w11, w12
			[0.3, 0.4], // w21, w22
		];
		const mockBiases = [0.05, 0.15];
		layer.updateWeights(
			new Map<string, number[][] | number[]>([
				["weights", mockWeights],
				["biases", mockBiases],
			]),
		);

		const outputGradient = [[0.5, 1.0]]; // dL/dY1, dL/dY2 for one sample
		const inputGradient = layer.backward(outputGradient);

		// W^T = [[0.1, 0.3], [0.2, 0.4]]
		// dL/dX1 = dL/dY1 * w11 + dL/dY2 * w12 = 0.5*0.1 + 1.0*0.2 = 0.05 + 0.2 = 0.25
		// dL/dX2 = dL/dY1 * w21 + dL/dY2 * w22 = 0.5*0.3 + 1.0*0.4 = 0.15 + 0.4 = 0.55
		// Expected: [[0.25, 0.55]]

		assert.strictEqual(inputGradient.length, 1);
		assert.strictEqual(inputGradient[0].length, 2);
		assert.ok(Math.abs(inputGradient[0][0] - 0.25) < 1e-9);
		assert.ok(Math.abs(inputGradient[0][1] - 0.55) < 1e-9);
	});

	it("backward pass should compute input gradients correctly", () => {
		const layer = new Dense(2, 2);
		// W = [[w11, w12], [w21, w22]]
		// B = [b1, b2]
		// Y = XW + B
		// dL/dX = dL/dY * W^T
		const mockWeights = [
			[0.1, 0.2], // w11, w12
			[0.3, 0.4], // w21, w22
		];
		const mockBiases = [0.05, 0.15];
		layer.updateWeights(
			new Map<string, number[][] | number[]>([
				["weights", mockWeights],
				["biases", mockBiases],
			]),
		);

		const outputGradient = [[0.5, 1.0]]; // dL/dY1, dL/dY2 for one sample
		const inputGradient = layer.backward(outputGradient);

		// W^T = [[0.1, 0.3], [0.2, 0.4]]
		// dL/dX1 = dL/dY1 * w11 + dL/dY2 * w12 = 0.5*0.1 + 1.0*0.2 = 0.05 + 0.2 = 0.25
		// dL/dX2 = dL/dY1 * w21 + dL/dY2 * w22 = 0.5*0.3 + 1.0*0.4 = 0.15 + 0.4 = 0.55
		// Expected: [[0.25, 0.55]]

		assert.strictEqual(inputGradient.length, 1);
		assert.strictEqual(inputGradient[0].length, 2);
		assert.ok(Math.abs(inputGradient[0][0] - 0.25) < 1e-9);
		assert.ok(Math.abs(inputGradient[0][1] - 0.55) < 1e-9);
	});

	it("backward pass should return empty array for empty outputGradient", () => {
		const layer = new Dense(2, 2);
		const inputGradient = layer.backward([]);
		assert.deepStrictEqual(
			inputGradient,
			[],
			"Should return empty array for empty input",
		);
	});

	it("backward pass should throw error for mismatched outputGradient dimensions", () => {
		const layer = new Dense(2, 2); // Expects outputGradient with 2 columns
		const wrongOutputGradient = [[0.5, 1.0, 1.5]]; // Has 3 columns
		assert.throws(
			() => layer.backward(wrongOutputGradient),
			/Output gradient dimension mismatch. Expected 2, got 3/,
			"Should throw error for wrong outputGradient dimensions",
		);
	});

	it("getWeightGradients should compute gradients for weights and biases", () => {
		const layer = new Dense(2, 1); // 2 inputs, 1 output
		const mockWeights = [[0.5], [0.2]];
		const mockBiases = [0.1];
		layer.updateWeights(
			new Map<string, number[][] | number[]>([
				["weights", mockWeights],
				["biases", mockBiases],
			]),
		);

		const layerInput = [
			[1, 2],
			[3, 4],
		]; // Batch of 2 samples
		const outputGradient = [[0.5], [1.0]]; // dL/dY for each sample

		// dL/dW_ij = (1/N) * sum(dL/dY_bj * X_bi)
		// dL/dB_j  = (1/N) * sum(dL/dY_bj)

		// For dL/dW[0][0]: (1/2) * (outputGradient[0][0]*layerInput[0][0] + outputGradient[1][0]*layerInput[1][0])
		//                = (1/2) * (0.5*1 + 1.0*3) = (1/2) * (0.5 + 3.0) = 3.5 / 2 = 1.75
		// For dL/dW[1][0]: (1/2) * (outputGradient[0][0]*layerInput[0][1] + outputGradient[1][0]*layerInput[1][1])
		//                = (1/2) * (0.5*2 + 1.0*4) = (1/2) * (1.0 + 4.0) = 5.0 / 2 = 2.5

		// For dL/dB[0]: (1/2) * (outputGradient[0][0] + outputGradient[1][0])
		//             = (1/2) * (0.5 + 1.0) = 1.5 / 2 = 0.75

		const gradients = layer.getWeightGradients(outputGradient, layerInput);
		const dLdW = gradients.get("weights") as number[][];
		const dLdB = gradients.get("biases") as number[];

		assert.ok(dLdW, "Weight gradients should exist");
		assert.strictEqual(dLdW.length, 2);
		assert.strictEqual(dLdW[0].length, 1);
		assert.ok(Math.abs(dLdW[0][0] - 1.75) < 1e-9, "dL/dW[0][0] incorrect");
		assert.ok(Math.abs(dLdW[1][0] - 2.5) < 1e-9, "dL/dW[1][0] incorrect");

		assert.ok(dLdB, "Bias gradients should exist");
		assert.strictEqual(dLdB.length, 1);
		assert.ok(Math.abs(dLdB[0] - 0.75) < 1e-9, "dL/dB[0] incorrect");
	});

	it("getWeightGradients should handle empty batch", () => {
		const layer = new Dense(2, 1);
		const gradients = layer.getWeightGradients([], []);
		const dLdW = gradients.get("weights") as number[][];
		const dLdB = gradients.get("biases") as number[];

		assert.deepStrictEqual(dLdW, [[0], [0]]);
		assert.deepStrictEqual(dLdB, [0]);
	});

	it("getWeightGradients should throw error on dimension mismatch", () => {
		const layer = new Dense(2, 1);
		const validOutputGradient = [[0.5]];
		const validLayerInput = [[1, 2]];

		// Mismatch outputGradient columns
		assert.throws(
			() => layer.getWeightGradients([[0.5, 0.5]], validLayerInput),
			/Dimension mismatch/,
		);
		// Mismatch layerInput columns
		assert.throws(
			() => layer.getWeightGradients(validOutputGradient, [[1]]),
			/Dimension mismatch/,
		);
		// Mismatch batch size
		assert.throws(
			() =>
				layer.getWeightGradients(validOutputGradient, [
					[1, 2],
					[3, 4],
				]),
			/Dimension mismatch/,
		);
	});
});
