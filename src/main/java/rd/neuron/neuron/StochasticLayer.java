package rd.neuron.neuron;

import java.util.Random;

import org.jblas.FloatMatrix;

import rd.data.TimedDistributionStructure;

/**
 * Stochastic Layer that add support for Binomial Trial based output Also
 * provided the per layer contrastive divergence training method
 * 
 * Based on JBLAS Float Matrix
 * 
 * @author azahar
 *
 */
public class StochasticLayer extends Layer {

	private final Random rnd;

	/**
	 * 
	 * @param weights
	 *            - weights
	 * @param bias
	 *            - at the hidden units
	 * @param inputBias
	 *            - at the visible units
	 * @param rnd
	 *            - random number generator
	 */
	public StochasticLayer(FloatMatrix weights, FloatMatrix bias, FloatMatrix inputBias, Random rnd) {
		super(weights, bias, inputBias, Function.LOGISTIC);

		this.rnd = rnd;
	}

	@Override
	public FloatMatrix io(FloatMatrix input) {
		return super.io(input);
	}

	@Override
	public FloatMatrix oi(FloatMatrix input) {
		return super.oi(input);
	}

	/**
	 * Generate stochastic output - input is a vector of activaiton values
	 * between 0 and 1 (e.g. output of sigmoid)
	 * 
	 * @param input
	 *            vector of values between 0 and 1 - such as the kind generated
	 *            by a sigmoid layer
	 * @return binary output vector of same length as input
	 */
	public FloatMatrix stochasticLayer(FloatMatrix input) {
		FloatMatrix output = new FloatMatrix(input.rows, input.columns);

		for (int i = 0; i < input.columns; i++) {
			for (int j = 0; j < input.rows; j++) {
				output.put(j, i, input.get(j, i) > rnd.nextFloat() ? 1f : 0f);
			}
		}
		return output;
	}
	
	/**
	 * Generate stochastic output - input is a vector of activaiton values
	 * between 0 and 1 (e.g. output of sigmoid)
	 * 
	 * @param input
	 *            vector of values between 0 and 1 - such as the kind generated
	 *            by a sigmoid layer
	 * @return binary output vector of same length as input
	 */
	public static FloatMatrix stochasticLayer(FloatMatrix input,Random rnd) {
		FloatMatrix output = new FloatMatrix(input.rows, input.columns);

		for (int i = 0; i < input.columns; i++) {
			for (int j = 0; j < input.rows; j++) {
				output.put(j, i, input.get(j, i) > rnd.nextFloat() ? 1f : 0f);
			}
		}
		return output;
	}

	private TimedDistributionStructure<String, String> distHV = null;

	/**
	 * set the Distribution Structure
	 * 
	 * @param tds
	 */
	public void setDistHV(TimedDistributionStructure<String, String> tds) {
		distHV = tds;
	}

	/**
	 * Contrastive Divergence Learning Method
	 * 
	 * @param input
	 *            - input vector
	 * @param iter
	 *            - number of CD iterations (between 1 and 10)
	 * @param learningRate
	 *            - learning Rate
	 * 
	 */
	@Override
	public void train(FloatMatrix input, int iter, float learningRate) {

		if (iter <= 0) {
			iter = 1;
		}

		FloatMatrix vk = null, hk = null, hk0 = null, hm = null, hm0 = null;

		hk0 = stochasticLayer(hm0 = io(input));
		addDist(hk0, input);
		for (int i = 0; i < iter; i++) {

			if (i == 0) {

				vk = stochasticLayer(oi(hk0));

				hk = stochasticLayer(hm = io(vk));

			} else {
				vk = stochasticLayer(oi(hk));

				hk = stochasticLayer(hm = io(vk));

			}

		}
		addDist(hk, vk);

		updateWeights(learningRate, hm0, hm, input, vk);
		updateHiddenBias(learningRate, hk0, hm);
		updateVisibleBias(learningRate, input, vk);

	}

	/**
	 * Update Hidden unit bias
	 * 
	 * @param learningRate
	 * @param initialHiddenSample
	 * @param currentHiddenMean
	 */
	private void updateHiddenBias(float learningRate, FloatMatrix initialHiddenSample, FloatMatrix currentHiddenMean) {

		bias = bias.add((initialHiddenSample.sub(currentHiddenMean)).mul(learningRate));
	}

	/**
	 * Update Visible unit bias
	 * 
	 * @param learningRate
	 * @param input
	 * @param currentVisibleSample
	 */
	private void updateVisibleBias(float learningRate, FloatMatrix input, FloatMatrix currentVisibleSample) {

		inputBias = inputBias.add((input.sub(currentVisibleSample)).mul(learningRate));
	}

	/**
	 * Update Weights
	 * 
	 * @param learningRate
	 * @param initialHiddenMean
	 * @param currentHiddenMean
	 * @param input
	 * @param visibleSample
	 */
	private void updateWeights(float learningRate, FloatMatrix initialHiddenMean, FloatMatrix currentHiddenMean,
			FloatMatrix input, FloatMatrix visibleSample) {

		FloatMatrix init = input.mmul(initialHiddenMean.transpose());
		FloatMatrix fin = visibleSample.mmul(currentHiddenMean.transpose());

		FloatMatrix delta = init.sub(fin);

		weights = weights.add(delta.mul(learningRate));

	}

	/**
	 * Friendly matrix printing method
	 * 
	 * @param s
	 *            - user friendly label for matrix
	 * @param v
	 */
	public static void p(String s, FloatMatrix v) {
		System.out.println(s + " : " + v.rows + "x" + v.columns);
		System.out.println(v);
	}

	/**
	 * Add to distribution structure if present
	 * 
	 * @param hiddenKey
	 *            - hidden layer output
	 * @param visibleKey
	 *            - visible layer output
	 */
	private final void addDist(FloatMatrix hiddenKey, FloatMatrix visibleKey) {
		if (distHV != null) {
			try {
				distHV.add(hiddenKey.elementsAsList().toString().replace(",", ""),
						visibleKey.elementsAsList().toString().replace(",", ""));
			} catch (Exception e) {
				e.printStackTrace(System.err);
			}
		}
	}
}
