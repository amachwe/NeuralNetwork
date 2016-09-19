package rd.neuron.deep;

import java.util.Random;

/**
 * As name suggest - process method uses sigmoid and post process uses Binomial Trial 
 * 
 * @author azahar
 *
 */
public class SigmoidBinomialSamplingLayer extends LayerDecorator {

	private final Random rnd;

	public SigmoidBinomialSamplingLayer(NeuralElement ne, Random rnd) {
		super(ne);
		this.rnd = rnd;

	}

	@Override
	public double[] process(int[] in, Direction d) {

		double[] out = this.l.process(in, d);

		for (int i = 0; i < out.length; i++) {
			out[i] = sigmoid(out[i]);
		}

		return out;

	}

	private final int binomial(double prob) {
		if (prob < 0 || prob > 1) {
			return 0;
		}
		if (rnd.nextFloat() < prob) {
			return 1;
		} else {
			return 0;
		}
	}

	private final double sigmoid(double in) {
		return 1 / (1 + Math.exp(-in));

	}

	@Override
	public int[] postProcess(double[] in) {
		int[] output = new int[in.length];
		for (int i = 0; i < in.length; i++) {
			output[i] = binomial(in[i]);
		}
		return output;
	}

	@Override
	public void updateWeight(int indexIn, int indexOut, double delta) {
		this.l.updateWeight(indexIn, indexOut, delta);

	}

	private double[] inputBias = null;
	private double[][] weights = null;

	private void prepareReconstruction() {
		inputBias = getInputBias();
		weights = getWeights();
	}

	@Override
	public double[] reconstruct(int[] in) {
		if (inputBias == null || weights == null) {
			prepareReconstruction();
		}
		double[] output = process(in, Direction.Forward);
		double[] recon = new double[this.getVisibleNeuronCount()];
		for (int i = 0; i < this.getVisibleNeuronCount(); i++) {
			recon[i] = inputBias[i];
			for (int j = 0; j < this.getHiddenNeuronCount(); j++) {
				recon[i] += output[j] * weights[i][j];
			}
			recon[i] = sigmoid(recon[i]);
		}
		return recon;
	}

}
