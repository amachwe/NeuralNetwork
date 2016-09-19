package rd.neuron.deep;

import java.util.Arrays;
/**
 * Layer implementation using simple Arrays
 * @author azahar
 *
 */
public class Layer implements NeuralElement {

	private final int numInputN, numOutputN;
	private final double[][] weights;
	private final double[] inputBias, outputBias;

	public Layer(double[] inputBias, double[] outputBias) {
		this.numInputN = inputBias.length;
		this.numOutputN = outputBias.length;

		this.inputBias = inputBias;
		this.outputBias = outputBias;

		this.weights = new double[this.numInputN][this.numOutputN];
	}

	public Layer(double[] inputBias, double[] outputBias, double[][] weights) {
		this.numInputN = inputBias.length;
		this.numOutputN = outputBias.length;

		this.inputBias = inputBias;
		this.outputBias = outputBias;

		this.weights = weights;
	}

	private double[] empty = new double[1];

	@Override
	public double[] process(int[] input, Direction direction) {

		if (direction == Direction.Forward) {
			double[] output = new double[numOutputN];
			for (int j = 0; j < this.numOutputN; j++) {
				output[j] = outputBias[j];
				for (int i = 0; i < this.numInputN; i++) {
					output[j] += input[i] * this.weights[i][j];
				}
			}
			return output;
		} else if (direction == Direction.Backward) {
			double[] output = new double[numInputN];
			for (int i = 0; i < this.numInputN; i++) {
				output[i] = inputBias[i];
				for (int j = 0; j < this.numOutputN; j++) {
					output[i] += input[j] * this.weights[i][j];
				}
			}
			return output;
		}

		return empty;
	}

	@Override
	public void updateWeight(int indexIn, int indexOut, double delta) {
		this.weights[indexIn][indexOut] += delta;

	}

	@Override
	public int[] postProcess(double[] in) {

		int[] val = new int[in.length];
		for (int i = 0; i < in.length; i++) {
			val[i] = (int) in[i];
		}

		return val;
	}

	@Override
	public void updateHiddenBias(int indexNeuron, double delta) {
		outputBias[indexNeuron] += delta;
	}

	@Override
	public void updateVisibleBias(int indexNeuron, double delta) {
		inputBias[indexNeuron] += delta;

	}

	@Override
	public int getVisibleNeuronCount() {
		return inputBias.length;
	}

	@Override
	public int getHiddenNeuronCount() {
		return outputBias.length;
	}

	@Override
	public double[] reconstruct(int[] in) {
		double[] output = new double[numOutputN];
		for (int j = 0; j < this.numOutputN; j++) {
			output[j] = outputBias[j];
			for (int i = 0; i < this.numInputN; i++) {
				output[j] += in[i] * this.weights[i][j];
			}
		}
		return output;
	}

	@Override
	public double[] getInputBias() {
		return Arrays.copyOf(this.inputBias, this.inputBias.length);
	}

	@Override
	public double[] getOutputBias() {
		return Arrays.copyOf(this.outputBias, this.outputBias.length);
	}

	@Override
	public double[][] getWeights() {
		return (this.weights.clone());
	}

}
