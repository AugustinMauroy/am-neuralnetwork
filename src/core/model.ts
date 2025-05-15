import { MeanSquaredError } from "../losses/mse.ts";
import type { Optimizer } from "../optimizes/mod.ts";

/**
 * Represents a single layer in a neural network.
 */
export interface Layer {
  /**
   * Performs the forward pass of the layer.
   * @param input The input data to the layer.
   * @returns The output data from the layer.
   */
  forward(input: number[][]): number[][];

  /**
   * Performs the backward pass of the layer (backpropagation).
   * @param outputGradient The gradient of the loss with respect to the output of this layer.
   * @returns The gradient of the loss with respect to the input of this layer.
   * Signature might need adjustment for specific layer needs.
   */
  backward(outputGradient: number[][]): number[][];
  // For a full training loop, layers might also need methods like:
  // getWeights(): any; // To get current weights for optimizer/saving
  // getGradients(outputGradient: number[][], layerInput: number[][]): any; // To calculate gradients for its weights
  // updateWeights(optimizer: Adam, gradients: any): void; // To apply updates from optimizer
  // getConfig(): any; // For saving model architecture
}

/**
 * Represents a layer that has trainable weights and biases.
 * Extends the base {@link Layer} interface.
 */
export interface TrainableLayer extends Layer {
  /**
   * Retrieves the current weights and biases of the layer.
   * @returns A map where keys are 'weights' or 'biases' and values are the corresponding matrices or vectors.
   * Example: weights and biases
   */
  getWeights(): Map<string, number[] | number[][]>;

  /**
   * Calculates the gradients of the loss with respect to the layer's weights and biases.
   * @param outputGradient The gradient of the loss with respect to the output of this layer.
   * @param layerInput The input data that was fed into this layer during the forward pass.
   * @returns A map containing the gradients for weights and biases.
   * Gradients for weights and biases
   */
  getWeightGradients(
    outputGradient: number[][],
    layerInput: number[][],
  ): Map<string, number[] | number[][]>;

  /**
   * Updates the layer's weights and biases.
   * @param updatedWeights A map containing the new weights and biases to apply.
   */
  updateWeights(updatedWeights: Map<string, number[] | number[][]>): void;
}

/**
 * Represents a neural network model composed of a sequence of layers.
 */
export class Model {
  /** @hidden The sequence of layers in the model. */
  private layers: Layer[] = [];
  /** @hidden The optimizer used for training the model. */
  private optimizer!: Optimizer;
  /** @hidden The loss function used to evaluate the model's performance. */
  private lossFunction!: MeanSquaredError; // Assuming MeanSquaredError for now
  /** @hidden A list of metrics to evaluate during training and testing. */
  private metrics: string[] = [];

  /**
   * Creates an instance of the Model.
   */
  constructor() {}

  /**
   * Adds a layer to the model.
   * @param layer The layer to add to the model.
   */
  public addLayer(layer: Layer): void {
    this.layers.push(layer);
  }

  /**
   * Configures the model for training.
   * @param optimizer The optimizer to use for training.
   * @param lossFunction The loss function to use.
   * @param metrics A list of metrics to evaluate.
   */
  public compile(
    optimizer: any,
    lossFunction: MeanSquaredError,
    metrics: string[],
  ): void {
    this.optimizer = optimizer;
    this.lossFunction = lossFunction;
    this.metrics = metrics;
  }

  /**
   * Trains the model for a fixed number of epochs (iterations on a dataset).
   *
   * @param trainingData The input data for training.
   * @param trainingLabels The target labels for training.
   * @param epochs The number of epochs to train the model.
   * @param batchSize The number of samples per gradient update.
   * @param debugEpochEnabled Whether to log loss information after each epoch. Defaults to false.
   * @returns A promise that resolves when training is complete.
   *
   * @example
   * ```typescript
   * const model = new Model();
   * // ... add layers ...
   * model.compile(new AdamOptimizer(), new MeanSquaredError(), ['accuracy']);
   * await model.fit(trainingData, trainingLabels, 10, 32);
   * ```
   */
  public async fit(
    trainingData: number[][],
    trainingLabels: number[][],
    epochs: number,
    batchSize: number,
    debugEpochEnabled: boolean = false,
  ): Promise<void> {
    if (!this.optimizer || !this.lossFunction) {
      throw new Error("Model must be compiled before training.");
    }
    if (trainingData.length !== trainingLabels.length) {
      throw new Error(
        "Training data and labels must have the same number of samples.",
      );
    }

    const numSamples = trainingData.length;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      // TODO: Implement proper data shuffling here for each epoch

      for (let i = 0; i < numSamples; i += batchSize) {
        const batchInputs = trainingData.slice(
          i,
          Math.min(i + batchSize, numSamples),
        );
        const batchTargets = trainingLabels.slice(
          i,
          Math.min(i + batchSize, numSamples),
        );

        if (batchInputs.length === 0) continue;

        // 1. Forward pass - store intermediate outputs for backpropagation
        const layerOutputs: number[][][] = [];
        let currentBatchOutput = batchInputs;
        layerOutputs.push(currentBatchOutput); // Store initial input

        for (const layer of this.layers) {
          currentBatchOutput = layer.forward(currentBatchOutput);
          layerOutputs.push(currentBatchOutput);
        }
        const finalPredictions = currentBatchOutput;

        // 2. Calculate loss
        let dLdOutput: number[][] = finalPredictions.map((predRow, rIdx) =>
          predRow.map((predVal, cIdx) =>
            2 * (predVal - batchTargets[rIdx][cIdx]) / predRow.length
          )
        );

        // 3. Backward pass
        let currentGradient = dLdOutput;
        for (let j = this.layers.length - 1; j >= 0; j--) {
          const layer = this.layers[j];
          const layerInput = layerOutputs[j]; // Input that went into this layer
          currentGradient = layer.backward(currentGradient);
        }

        // Conceptual optimizer step - this requires layers to expose weights and gradients
        this.layers.forEach((layer) => {
          if (
            typeof (layer as any).getWeights === "function" &&
            typeof (layer as any).getWeightGradients === "function" &&
            typeof (layer as any).setWeights === "function"
          ) {
            // TODO: Implement weight update logic:
            // 1. Get current weights from the layer (trainable layer)
            // 2. Get weight gradients based on output gradient and layer input
            // 3. Use optimizer to calculate weight updates
            // 4. Apply updated weights back to the layer
          }
        });
      }
      if (debugEpochEnabled) {
        console.log(
          `Epoch ${epoch + 1}/${epochs}, Loss: ${
            epochLoss / (numSamples / batchSize)
          }`,
        );
      }
      // TODO: Calculate and log metrics if any
    }
  }

  /**
   * Generates output predictions for the input samples.
   * @param inputData The input data for which to generate predictions.
   * @returns The model's predictions.
   */
  public predict(inputData: number[][]): number[][] {
    let currentOutput = inputData;
    for (const layer of this.layers) {
      currentOutput = layer.forward(currentOutput);
    }
    return currentOutput;
  }

  /**
   * Evaluates the model on a validation dataset.
   * @param validationData The input data for validation.
   * @param validationLabels The target labels for validation.
   * @returns An object containing the loss and any configured metrics (e.g., accuracy).
   */
  public evaluate(
    validationData: number[][],
    validationLabels: number[][],
  ): any {
    if (!this.lossFunction) {
      throw new Error("Model must be compiled before evaluation.");
    }
    if (validationData.length !== validationLabels.length) {
      throw new Error(
        "Validation data and labels must have the same number of samples.",
      );
    }

    const predictions = this.predict(validationData);
    let totalLoss = 0;
    for (let i = 0; i < predictions.length; i++) {
      totalLoss += this.lossFunction.calculate(
        predictions[i],
        validationLabels[i],
      );
    }
    const avgLoss = totalLoss / predictions.length;

    const results: { loss: number; accuracy?: number } = { loss: avgLoss };

    if (this.metrics.includes("accuracy")) {
      let correctPredictions = 0;
      for (let i = 0; i < predictions.length; i++) {
        const predictedValue = predictions[i][0] > 0.5 ? 1 : 0; // Assuming single binary output
        const targetValue = validationLabels[i][0];
        if (predictedValue === targetValue) {
          correctPredictions++;
        }
      }
      results.accuracy = correctPredictions / predictions.length;
    }

    return results;
  }

  /**
   * Saves the model's architecture, weights, and optimizer state.
   * Note: This method is not fully implemented.
   * @param filePath The path where the model will be saved.
   */
  public save(filePath: string): void {
    console.warn("Model.save() is not fully implemented.");
    const modelState = {
      layersConfig: this.layers.map((layer) => {
        if (
          typeof (layer as any).getConfig === "function" &&
          typeof (layer as any).getWeights === "function"
        ) {
          return {
            type: layer.constructor.name,
            config: (layer as any).getConfig(),
            weights: (layer as any).getWeights(),
          };
        }
        return { type: layer.constructor.name }; // Fallback
      }),
      optimizerConfig: this.optimizer ? {/* optimizer state */} : null,
      lossFunctionConfig: this.lossFunction
        ? {/* loss function state */}
        : null,
      metrics: this.metrics,
    };
    try {
      console.log(
        `Model state (conceptual): ${JSON.stringify(modelState, null, 2)}`,
      );
      console.log(
        `Model would be saved to ${filePath} (actual file writing not implemented here)`,
      );
    } catch (e) {
      console.error("Failed to save model:", e);
    }
  }

  /**
   * Loads a model from a file.
   * Note: This method is not fully implemented and returns a new empty model.
   * @param _filePath The path from which to load the model.
   * @returns A new {@link Model} instance (currently empty).
   */
  public static load(_filePath: string): Model {
    console.warn(
      "Model.load() is not fully implemented and returns a new empty model.",
    );
    try {
      // Placeholder for loading model architecture and weights.
    } catch (e) {
      console.error("Failed to load model:", e);
    }
    return new Model();
  }
}

export default Model;
