package rd.neuron.deep.training;

import rd.neuron.deep.NeuralElement;
import rd.neuron.deep.NeuralElement.Direction;
/**
 * Implementation of Contrastive Divergence using simple arrays
 * @author azahar
 *
 */
public class ContrastiveDivergence {

	public static NeuralElement train(int[][] inputs, int iter, float learningRate, NeuralElement l) {

		int batchSize = inputs.length;
		int[] hk = new int[l.getHiddenNeuronCount()],hk0=hk;
		int[] vk = new int[l.getVisibleNeuronCount()];

		double[] pHM = null;
		double[] hm = null;

		if (iter <= 0) {
			iter = 1;
		}

		for (int ex = 0; ex < inputs.length; ex++) {
			int[] x0 = inputs[ex];
			hk0 = l.postProcess(pHM = l.process(x0, Direction.Forward));
			for (int i = 0; i < iter; i++) {

				if (i == 0) {
					vk = l.postProcess(l.process(hk0, Direction.Backward));
					hk = l.postProcess(hm = l.process(vk, Direction.Forward));
				} else {
					vk = l.postProcess(l.process(hk, Direction.Backward));
					hk = l.postProcess(hm = l.process(vk, Direction.Forward));
				}

			}
			updateWeights(learningRate, batchSize, l, pHM, hm, x0, vk);
			updateHiddenBias(learningRate, batchSize, l, hk0, hm);
			updateVisibleBias(learningRate, batchSize, l, x0, vk);
		}

		return l;

	}

	private static void updateHiddenBias(float learningRate, int batchSize, NeuralElement l, int initialHiddenSample[],
			double currentHiddenMean[]) {
		for (int i = 0; i < initialHiddenSample.length; i++) {
			l.updateHiddenBias(i, (initialHiddenSample[i] - currentHiddenMean[i]) * learningRate / batchSize);
		}
	}

	private static void updateVisibleBias(float learningRate, int batchSize, NeuralElement l, int input[],
			int currentVisibleSample[]) {
		for (int i = 0; i < input.length; i++) {
			l.updateVisibleBias(i, (input[i] - currentVisibleSample[i]) * learningRate / batchSize);
		}
	}

	private static void updateWeights(float learningRate, int batchSize, NeuralElement l, double[] initialHiddenMean,
			double[] currentHiddenMean, int[] input, int[] visibleSample) {
		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < initialHiddenMean.length; j++) {
				double delta = (initialHiddenMean[j] * input[i]) - (currentHiddenMean[j] * visibleSample[i]);
				l.updateWeight(i, j, delta * learningRate / batchSize);

			}
		}

	}
}
