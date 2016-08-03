package rd.neuron.neuron.perceptron;

import java.util.Arrays;

import org.jblas.FloatMatrix;

import rd.data.DataStreamer;

/**
 * Single Layer Perceptron - as a sample implementation
 * 
 * @author azahar
 *
 */
public class Perceptron {

	private final int width;
	private float[] weights;

	/**
	 * 
	 * @param width - number of inputs
	 */
	public Perceptron(int width) {
		this.width = width + 1;// For Bias
		weights = new float[this.width];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = (float) Math.random();
		}

	}

	public int getNumberOfInputs() {
		return width;
	}

	public float[] getWeights() {
		return weights;
	}

	/**
	 * Predict
	 * @param input - array
	 * @return +1 or -1 (class)
	 */
	public int predict(float[] input) {
		float sum = 0;
		for (int i = 0; i < input.length; i++) {
			sum += input[i] * weights[i + 1];
		}
		sum += weights[0];

		return sum > 0 ? 1 : -1;

	}

	/**
	 * Train
	 * @param ds - data streamer with training data
	 * @param learningRate - small number (0.1)
	 */
	public void train(DataStreamer ds, float learningRate) {
		float[] newWeights = weights.clone();

		for (FloatMatrix item : ds) {
			float[] input = item.toArray();
			float output = ds.getOutput(item).get(0);
			float sum = 0;
			for (int i = 0; i < input.length; i++) {

				sum += input[i] * weights[i + 1];
			}
			sum += weights[0] * 1;

			float delta = (output > 0 ? 1 : -1);

			if (!(sum > 0 && delta > 0) && !(sum < 0 && delta < 0)) {

				for (int i = 0; i < newWeights.length; i++) {
					if (i == 0) {
						newWeights[i] -= learningRate;
					} else {
						newWeights[i] -= learningRate * input[i - 1];
					}
				}
			}
		}
		weights = newWeights;
	}
}
