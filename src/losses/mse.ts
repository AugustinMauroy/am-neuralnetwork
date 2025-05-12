export class MeanSquaredError {
  calculate(predictions: number[], targets: number[]): number {
    if (predictions.length !== targets.length) {
      throw new Error("Predictions and targets must have the same length.");
    }

    let sum = 0;
    for (let i = 0; i < predictions.length; i++) {
      const error = predictions[i] - targets[i];
      sum += error * error;
    }

    return sum / predictions.length;
  }
}
