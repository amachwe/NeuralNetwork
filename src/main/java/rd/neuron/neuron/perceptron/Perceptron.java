package rd.neuron.neuron.perceptron;

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
	 * @param width
	 *            - number of inputs
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
	 * 
	 * @param input
	 *            - array
	 * @return +1 or -1 (class)
	 */
	public int predict(float[] input) {
		float sum = 0;
		for (int i = 0; i < input.length; i++) {
			sum += input[i] * weights[i + 1];
		}
		// Bias
		sum += weights[0];

		return sum > 0 ? 1 : -1;

	}

	/**
	 * Train
	 * 
	 * @param ds
	 *            - data streamer with training data
	 * @param learningRate
	 *            - small number ( between 0.5 and 0.05)
	 */
	public void train(DataStreamer ds, float learningRate) {
		float[] newWeights = weights.clone();

		for (FloatMatrix item : ds) {
			train(item.toArray(), ds.getOutput(item).get(0), weights, newWeights, learningRate);
		}
		weights = newWeights;
	}

	public void train(float[] input, float output, float[] weights, float[] newWeights, float learningRate) {

		float sum = 0;
		for (int i = 0; i < input.length; i++) {

			sum += input[i] * weights[i + 1];
		}
		// Bias
		sum += weights[0] * 1;

		float delta = (output > 0 ? 1 : -1);

		if (!(sum > 0 && delta > 0) && !(sum < 0 && delta < 0)) {

			for (int i = 0; i < newWeights.length; i++) {
				if (i == 0) {
					// Bias
					newWeights[i] -= learningRate;
				} else {
					newWeights[i] -= learningRate * input[i - 1];
				}
			}
		}
	}

	/**
	 * Train
	 * 
	 * @param ds
	 *            - data streamer with training data
	 * @param miniBatch
	 *            - batch size
	 * @param learningRate
	 *            - small number ( between 0.5 and 0.05)
	 */
	public void trainSGD(DataStreamer ds, int miniBatch, float learningRate) {
		float[] newWeights = weights.clone();
		DataStreamer batch = new DataStreamer(ds.getRandom().length, ds.getOutput(ds.getRandom()).length);
		for (int i = 0; i < miniBatch; i++) {
			FloatMatrix item = ds.getRandom();

			batch.add(item.toArray(), ds.getOutput(item).toArray());
		}
		for (FloatMatrix item : batch) {
			train(item.toArray(), ds.getOutput(item).get(0), weights, newWeights, learningRate);
		}
		weights = newWeights;
	}
}
