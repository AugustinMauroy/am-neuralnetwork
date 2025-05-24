import assert from "node:assert/strict";
import { describe, it, mock } from "node:test";
import { Model } from "./model.ts";
import { Dense } from "../layers/mod.ts";
import { SGD } from "../optimizes/mod.ts";
import { MeanSquaredError } from "../losses/mod.ts";
import type { Layer, TrainableLayer } from "./model.ts";

// Mock Layer for testing predict and evaluate logic without full layer complexities
class MockPredictLayer implements Layer {
	forward(input: number[][]): number[][] {
		// Simple transformation: multiply by 2
		return input.map((row) => row.map((x) => x * 2));
	}
	// Dummy implementations for other Layer methods
	backward(gradient: number[][]): number[][] {
		return gradient;
	} // Not directly used by predict/evaluate
	getOutputShape(inputShape?: number[]): number[] {
		return inputShape || [Number.NaN, Number.NaN];
	}
	getInputShape(): number[] {
		return [Number.NaN, Number.NaN];
	}
	getName(): string {
		return "MockPredictLayer";
	}
	getConfig(): Record<string, unknown> {
		return {};
	}
}

class MockEvalLayer implements Layer {
	forward(input: number[][]): number[][] {
		// Mock output for binary classification: if input > 0.5, output 0.8, else 0.2
		return input.map((row) => [row[0] > 0.5 ? 0.8 : 0.2]);
	}
	backward(gradient: number[][]): number[][] {
		return gradient;
	}
	getOutputShape(inputShape?: number[]): number[] {
		return inputShape ? [inputShape[0], 1] : [Number.NaN, 1];
	}
	getInputShape(): number[] {
		return [Number.NaN, Number.NaN];
	}
	getName(): string {
		return "MockEvalLayer";
	}
	getConfig(): Record<string, unknown> {
		return {};
	}
}

// Mock TrainableLayer for testing specific fit conditions
class MockInconsistentTrainableLayer implements TrainableLayer {
	forward(input: number[][]): number[][] {
		return input; // Simple pass-through
	}
	backward(gradient: number[][]): number[][] {
		return gradient; // Simple pass-through
	}
	getWeights(): Map<string, number[] | number[][]> {
		// Only has 'actual_param'
		return new Map([["actual_param", [[0]]]]);
	}
	getWeightGradients(
		_outputGradient: number[][],
		_layerInput: number[][],
	): Map<string, number[] | number[][]> {
		// Returns gradient for 'ghost_param' (not in getWeights) and 'actual_param'
		return new Map([
			["ghost_param", [[0.1]]],
			["actual_param", [[0.2]]],
		]);
	}
	updateWeights(_updatedWeights: Map<string, number[] | number[][]>): void {
		// No-op
	}
	getName(): string {
		return "MockInconsistentTrainableLayer";
	}
	getConfig(): Record<string, unknown> {
		return {};
	}
}

describe("Model", () => {
	it("should create an empty model", () => {
		const model = new Model();

		assert.ok(model instanceof Model, "Model instance should be created");
		assert.deepStrictEqual(
			model.getLayers(),
			[],
			"New model should have no layers",
		);
	});

	it("should add a layer to the model", () => {
		const model = new Model();
		const layer = new Dense(2, 3); // Example Dense layer
		model.addLayer(layer);

		assert.strictEqual(
			model.getLayers().length,
			1,
			"Model should have one layer after adding",
		);
		assert.strictEqual(
			model.getLayers()[0],
			layer,
			"The added layer should be in the model",
		);
	});

	it("should compile the model and allow fit/evaluate to proceed past compilation check", async () => {
		const model = new Model();
		const optimizer = new SGD();
		const loss = new MeanSquaredError();
		const metrics = ["accuracy"];
		model.compile(optimizer, loss, metrics);

		// Test fit: should throw error related to data, not compilation
		const trainingDataFit = [[0, 0]];
		const trainingLabelsFit = [[0], [1]]; // Mismatched length

		await assert.rejects(
			async () => model.fit(trainingDataFit, trainingLabelsFit, 1, 1),
			/Training data and labels must have the same number of samples./,
			"Fit should throw data mismatch error after compilation, not 'not compiled' error.",
		);

		// Test evaluate: should throw error related to data, not compilation
		const validationDataEval = [[0, 0]];
		const validationLabelsEval = [[0], [1]]; // Mismatched length

		assert.throws(
			() => model.evaluate(validationDataEval, validationLabelsEval),
			/Validation data and labels must have the same number of samples./,
			"Evaluate should throw data mismatch error after compilation, not 'not compiled' error.",
		);
	});

	it("should throw an error if fit is called before compile", async () => {
		const model = new Model();
		const trainingData = [[0, 0]];
		const trainingLabels = [[0]];

		await assert.rejects(
			async () => model.fit(trainingData, trainingLabels, 1, 1),
			/Model must be compiled before training./,
			"Fit should require compilation.",
		);
	});

	it("should throw an error if evaluate is called before compile", () => {
		const model = new Model();
		const validationData = [[0, 0]];
		const validationLabels = [[0]];

		assert.throws(
			() => model.evaluate(validationData, validationLabels),
			/Model must be compiled before evaluation./,
			"Evaluate should require compilation.",
		);
	});

	it("should predict with a simple model using a mock layer", () => {
		const model = new Model();
		model.addLayer(new MockPredictLayer());
		const inputData = [[1], [2], [3]];
		const predictions = model.predict(inputData);

		assert.deepStrictEqual(
			predictions,
			[[2], [4], [6]],
			"Predictions should be double the input for MockPredictLayer",
		);
	});

	it("should throw error during fit if training data and labels have different lengths", async () => {
		const model = new Model();
		model.compile(new SGD(), new MeanSquaredError(), []);
		const trainingData = [
			[0, 0],
			[1, 1],
		];
		const trainingLabels = [[0]]; // Mismatched length

		await assert.rejects(
			async () => model.fit(trainingData, trainingLabels, 1, 1),
			/Training data and labels must have the same number of samples./,
		);
	});

	it("should throw error during evaluate if validation data and labels have different lengths", () => {
		const model = new Model();
		model.compile(new SGD(), new MeanSquaredError(), []);
		const validationData = [
			[0, 0],
			[1, 1],
		];
		const validationLabels = [[0]]; // Mismatched length

		assert.throws(
			() => model.evaluate(validationData, validationLabels),
			/Validation data and labels must have the same number of samples./,
		);
	});

	// This test uses a real Dense layer, acting as a mini-integration test for Model + Dense + SGD + MSE.
	it("should run fit and evaluate without throwing for a minimal setup with Dense layer", async () => {
		const model = new Model();
		const denseLayer = new Dense(2, 1); // Input 2 features, Output 1 feature
		model.addLayer(denseLayer);
		model.compile(new SGD(0.01), new MeanSquaredError(), ["accuracy"]);

		const trainingData = [
			[0, 0],
			[0, 1],
			[1, 0],
			[1, 1],
		]; // XOR inputs
		const trainingLabels = [[0], [1], [1], [0]]; // XOR labels

		await assert.doesNotThrow(
			async () => model.fit(trainingData, trainingLabels, 1, 1), // 1 epoch, batch size 1
			"Model.fit should not throw with valid setup and data.",
		);

		const validationData = [
			[0, 0],
			[1, 1],
		];
		const validationLabels = [[0], [0]]; // Example labels
		assert.doesNotThrow(() => {
			const evaluation = model.evaluate(validationData, validationLabels);
			assert.ok(
				typeof evaluation.loss === "number",
				"Evaluation loss should be a number.",
			);
			assert.ok(
				typeof evaluation.accuracy === "number",
				"Evaluation accuracy should be a number.",
			);
		}, "Model.evaluate should not throw with valid setup and data.");
	});

	it("evaluate should calculate loss and accuracy correctly with a mock layer", () => {
		const model = new Model();
		model.addLayer(new MockEvalLayer());
		model.compile(new SGD(), new MeanSquaredError(), ["accuracy"]);

		const validationData = [[1], [0], [1], [0]]; // Inputs for MockEvalLayer
		const validationLabels = [[1], [0], [1], [0]]; // Expected labels

		// MockEvalLayer output: [[0.8], [0.2], [0.8], [0.2]]
		// MSE: ((0.8-1)^2 + (0.2-0)^2 + (0.8-1)^2 + (0.2-0)^2) / 4
		// = (0.04 + 0.04 + 0.04 + 0.04) / 4 = 0.16 / 4 = 0.04
		// Accuracy: (0.8 > 0.5 -> 1) == 1 (correct); (0.2 > 0.5 -> 0) == 0 (correct). All 4 correct. Accuracy = 1.
		const evaluation = model.evaluate(validationData, validationLabels);
		assert.strictEqual(
			evaluation.loss.toFixed(2),
			"0.04",
			"Calculated loss is incorrect.",
		);
		assert.strictEqual(
			evaluation.accuracy,
			1,
			"Calculated accuracy is incorrect.",
		);
	});

	it("should handle fit with empty training data without errors", async () => {
		const model = new Model();
		model.compile(new SGD(), new MeanSquaredError(), []);
		const trainingData: number[][] = [];
		const trainingLabels: number[][] = [];
		await assert.doesNotThrow(
			async () => model.fit(trainingData, trainingLabels, 1, 1),
			"Fit should handle empty training data gracefully.",
		);
	});

	it("should handle evaluate with empty validation data, resulting in NaN for loss and accuracy", () => {
		const model = new Model();
		model.compile(new SGD(), new MeanSquaredError(), ["accuracy"]);
		const validationData: number[][] = [];
		const validationLabels: number[][] = [];
		const evaluation = model.evaluate(validationData, validationLabels);
		assert.ok(
			Number.isNaN(evaluation.loss),
			"Loss should be NaN for empty validation data.",
		);
		assert.ok(
			Number.isNaN(evaluation.accuracy),
			"Accuracy should be NaN for empty validation data.",
		);
	});

	it("should warn if a gradient is present for a non-existent weight during fit", async () => {
		const model = new Model();
		model.addLayer(new MockInconsistentTrainableLayer());
		model.compile(new SGD(), new MeanSquaredError(), []);

		const trainingData = [[1]];
		const trainingLabels = [[1]];

		const consoleWarnSpy = mock.method(console, "warn");

		await model.fit(trainingData, trainingLabels, 1, 1);

		assert.ok(
			consoleWarnSpy.mock.calls.some((call) =>
				String(call.arguments[0]).includes(
					"No current weights found for param ghost_param",
				),
			),
			"Should warn about missing weights for 'ghost_param'.",
		);
		consoleWarnSpy.mock.restore();
	});

	it("should log NaN for loss and accuracy if debugEpochEnabled and training data is empty", async () => {
		const model = new Model();
		model.compile(new SGD(), new MeanSquaredError(), ["accuracy"]);
		const trainingData: number[][] = [];
		const trainingLabels: number[][] = [];

		const consoleLogSpy = mock.method(console, "log");

		await model.fit(trainingData, trainingLabels, 1, 1, true); // debugEpochEnabled = true

		// Expect two log calls: one for loss, one for accuracy.
		assert.strictEqual(
			consoleLogSpy.mock.calls.length,
			2,
			"Should log twice for an empty dataset with debug enabled",
		);

		const lossLog = consoleLogSpy.mock.calls[0].arguments[0];
		const accuracyLog = consoleLogSpy.mock.calls[1].arguments[0];

		assert.match(
			lossLog,
			/Epoch 1\/1, Loss: NaN/,
			"Loss log for empty data is incorrect.",
		);
		assert.match(
			accuracyLog,
			/Epoch 1\/1, Accuracy: NaN/,
			"Accuracy log for empty data is incorrect.",
		);

		consoleLogSpy.mock.restore();
	});

	it("should log loss and accuracy if debugEpochEnabled and metrics include 'accuracy'", async () => {
		const model = new Model();
		model.addLayer(new MockEvalLayer()); // Using MockEvalLayer for predictable output
		model.compile(new SGD(), new MeanSquaredError(), ["accuracy"]);
		const trainingData: number[][] = [[1], [0]]; // Sample data
		const trainingLabels: number[][] = [[1], [0]]; // Corresponding labels

		const consoleLogSpy = mock.method(console, "log");

		await model.fit(trainingData, trainingLabels, 1, 1, true); // debugEpochEnabled = true

		assert.ok(
			consoleLogSpy.mock.calls.some((call) =>
				String(call.arguments[0]).includes("Epoch 1/1, Loss:"),
			),
			"Should log loss information.",
		);
		assert.ok(
			consoleLogSpy.mock.calls.some((call) =>
				String(call.arguments[0]).includes("Epoch 1/1, Accuracy:"),
			),
			"Should log accuracy information.",
		);

		// Check that accuracy is calculated and logged (e.g., not NaN)
		const accuracyLogCall = consoleLogSpy.mock.calls.find((call) =>
			String(call.arguments[0]).includes("Accuracy:"),
		);

		assert.ok(accuracyLogCall, "Accuracy log should exist");
		// MockEvalLayer: input [1] -> output [0.8] (>0.5 -> 1), input [0] -> output [0.2] (<=0.5 -> 0)
		// Labels: [1], [0]. Both correct. Accuracy should be 1.0000
		assert.match(
			String(accuracyLogCall.arguments[0]),
			/Accuracy: 1.0000/,
			"Logged accuracy value is incorrect.",
		);

		consoleLogSpy.mock.restore();
	});

	it("Should serialize and deserialize the model correctly", () => {
		const model = new Model();
		model.addLayer(new Dense(2, 3)); // Example Dense layer
		model.compile(new SGD(), new MeanSquaredError(), ["accuracy"]);

		const serialized: string = model.save();

		assert.ok(
			typeof serialized === "string",
			"Serialized model should be a string.",
		);
		assert.ok(
			serialized.length > 0,
			"Serialized model string should not be empty.",
		);

		const deserializedModel = Model.load(serialized);

		assert.ok(
			deserializedModel instanceof Model,
			"Deserialized model should be an instance of Model.",
		);
		assert.deepStrictEqual(
			deserializedModel.getLayers().length,
			1,
			"Deserialized model should have the same number of layers.",
		);
	});

	describe("Model.load error handling", () => {
		it("should throw error if layer class not found during load", () => {
			const modelJson = JSON.stringify({
				layers: [{ name: "NonExistentLayer", config: {} }],
				optimizer: { name: "SGD", config: { learningRate: 0.01 } },
				lossFunction: {
					name: "HuberLoss",
					config: {
						delta: 1.0,
					},
				},
				metrics: [],
			});

			assert.throws(
				() => Model.load(modelJson),
				"Error: Loss function class NonExistentLoss not found.",
			);
		});

		it("should work with a loss function that didn't have config", () => {
			const modelJson = JSON.stringify({
				layers: [{ name: "Dense", config: { inputUnits: 1, outputUnits: 1 } }],
				optimizer: { name: "SGD", config: { learningRate: 0.01 } },
				lossFunction: { name: "MeanSquaredError" },
				metrics: [],
			});

			const model = Model.load(modelJson);

			assert.ok(model instanceof Model, "Model should be loaded successfully.");
			assert.strictEqual(
				model.getLayers().length,
				1,
				"Model should have one layer after loading.",
			);
			assert.strictEqual(
				model.getLayers()[0].getName(),
				"Dense",
				"Loaded layer should be a Dense layer.",
			);
		});

		it("should throw error if optimizer class not found during load", () => {
			const modelJson = JSON.stringify({
				layers: [{ name: "Dense", config: { inputUnits: 1, outputUnits: 1 } }],
				optimizer: { name: "NonExistentOptimizer", config: {} },
				lossFunction: { name: "MeanSquaredError", config: {} },
				metrics: [],
			});

			assert.throws(
				() => Model.load(modelJson),
				/Optimizer class NonExistentOptimizer not found./,
			);
		});

		it("should throw error if loss function class not found during load", () => {
			const modelJson = JSON.stringify({
				layers: [{ name: "Dense", config: { inputUnits: 1, outputUnits: 1 } }],
				optimizer: { name: "SGD", config: { learningRate: 0.01 } },
				lossFunction: { name: "NonExistentLoss", config: {} },
				metrics: [],
			});

			assert.throws(() => Model.load(modelJson), {
				message: "Loss function class NonExistentLoss not found.",
			});
		});
	});
});
