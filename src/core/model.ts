import { MeanSquaredError } from "../losses/mse.ts";

export interface Layer {
  forward(input: number[][]): number[][];
  backward(outputGradient: number[][]): number[][]; // Signature might need adjustment for specific layer needs
  // For a full training loop, layers might also need methods like:
  // getWeights(): any; // To get current weights for optimizer/saving
  // getGradients(outputGradient: number[][], layerInput: number[][]): any; // To calculate gradients for its weights
  // updateWeights(optimizer: Adam, gradients: any): void; // To apply updates from optimizer
  // getConfig(): any; // For saving model architecture
}

// A more specific type for layers that have trainable weights.
export interface TrainableLayer extends Layer {
  getWeights(): Map<string, number[] | number[][]>; // Example: weights and biases
  getWeightGradients(
    outputGradient: number[][],
    layerInput: number[][],
  ): Map<string, number[] | number[][]>; // Gradients for weights and biases
  updateWeights(updatedWeights: Map<string, number[] | number[][]>): void;
}

class Model {
  private layers: Layer[] = [];
  private optimizer!: Adam; // Assuming Adam optimizer for now
  private lossFunction!: MeanSquaredError; // Assuming MeanSquaredError for now
  private metrics: string[] = [];

  constructor() {}

  public addLayer(layer: Layer): void {
    this.layers.push(layer);
  }

  public compile(
    optimizer: any,
    lossFunction: MeanSquaredError,
    metrics: string[],
  ): void {
    this.optimizer = optimizer;
    this.lossFunction = lossFunction;
    this.metrics = metrics;
  }

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

  public predict(inputData: number[][]): number[][] {
    let currentOutput = inputData;
    for (const layer of this.layers) {
      currentOutput = layer.forward(currentOutput);
    }
    return currentOutput;
  }

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

  public static load(_filePath: string): Promise<Model> {
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
