package rd.neuron.deep.training;

import rd.neuron.deep.NeuralElement;
import rd.neuron.deep.NeuralElement.Direction;

public class ContrastiveDivergence {

	public static NeuralElement train(int[][] inputs, int iter, float learningRate, NeuralElement l) {

		int batchSize = inputs.length;
		int[] resOut = new int[l.getHiddenNeuronCount()];
		int[] resIn = new int[l.getVisibleNeuronCount()];

		double[] pHM = null;
		double[] cHM = null;

		if (iter <= 0) {
			iter = 1;
		}
		for (int ex = 0; ex < inputs.length; ex++) {
			int[] x0 = inputs[ex];
			resOut = l.postProcess(pHM = l.process(x0, Direction.Forward));
			for (int i = 0; i < iter; i++) {

				resIn = l.postProcess(l.process(resOut, Direction.Backward));
				resOut = l.postProcess(cHM = l.process(resIn, Direction.Forward));
				updateWeights(learningRate, batchSize, l, pHM, cHM, x0, resIn);
				updateHiddenBias(learningRate, batchSize, l, pHM, cHM);
				updateVisibleBias(learningRate, batchSize, l, x0, resIn);
			}
		}

		return l;

	}

	private static void updateHiddenBias(float learningRate, int batchSize, NeuralElement l, double initialHiddenMean[],
			double currentHiddenMean[]) {
		for (int i = 0; i < initialHiddenMean.length; i++) {
			l.updateHiddenBias(i, (initialHiddenMean[i] - currentHiddenMean[i]) * learningRate / batchSize);
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
