package rd.neuron.deep.test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import rd.deep.learning.RBM;
import rd.neuron.deep.Layer;
import rd.neuron.deep.NeuralElement;
import rd.neuron.deep.SigmoidBinomialSamplingLayer;
import rd.neuron.deep.training.ContrastiveDivergence;

public class TestRBM {

	private final Random rand = new Random(123);

	// Training and Testing instances per pattern
	private final int trainNEach = 200, testNEach = 2;

	// Number of visible units per pattern
	private final int nVisibleEach = 4;

	private final float pNoiseTrain = 0.05f, pNoiseTest = 0.025f;

	// Number of patterns in data (1 pattern per class - thus 3 classes)
	private final int patterns = 3;

	// Total training and test instances
	private final int trainN = trainNEach * patterns, testN = testNEach * patterns;

	// Number of visible units
	private final int nVisible = nVisibleEach * patterns;

	// Number of hidden units
	private final int nHidden = 6;

	// Data for training, testing and output (for comparing)
	private final int[][] trainX = new int[trainN][nVisible], testX = new int[testN][nVisible];
	private final double[][] reconstrX = new double[testN][nVisible];

	private final int epochs = 10000;
	private float learningRate = 0.2f;

	// Mini-batch size and number of instances of mini-batches
	private final int miniBatchSize = 10, miniBatchN = trainN / miniBatchSize;

	@Test
	public void doTest() {

		// Prepare Training Mini Batch
		int[][][] trainMiniBatch = new int[miniBatchN][miniBatchSize][nVisible];
		List<Integer> miniBatchIndex = new ArrayList<>();
		for (int i = 0; i < trainN; i++) {
			miniBatchIndex.add(i);
		}
		Collections.shuffle(miniBatchIndex, rand);

		//
		// Create training data and test data for demo.
		// Data without noise would be:
		// class 1 : [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
		// class 2 : [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
		// class 3 : [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
		// and to each data, we add some noise.
		// For example, one of the data in class 1 could be:
		// [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1]
		//
		for (int pattern = 0; pattern < patterns; pattern++) {
			for (int n = 0; n < trainNEach; n++) {

				int n_ = pattern * trainNEach + n;

				for (int i = 0; i < nVisible; i++) {
					if ((n_ >= trainNEach * pattern && n_ < trainNEach * (pattern + 1))
							&& (i >= nVisibleEach * pattern && i < nVisibleEach * (pattern + 1))) {
						trainX[n_][i] = RBM.binomial(1, 1 - pNoiseTrain, rand);
					} else {
						trainX[n_][i] = RBM.binomial(1, pNoiseTrain, rand);
					}
				}
			}

			for (int n = 0; n < testNEach; n++) {

				int n_ = pattern * testNEach + n;

				for (int i = 0; i < nVisible; i++) {
					if ((n_ >= testNEach * pattern && n_ < testNEach * (pattern + 1))
							&& (i >= nVisibleEach * pattern && i < nVisibleEach * (pattern + 1))) {
						testX[n_][i] = RBM.binomial(1, 1 - pNoiseTest, rand);
					} else {
						testX[n_][i] = RBM.binomial(1, pNoiseTest, rand);
					}
				}
			}
		}

		// create minibatches
		for (int i = 0; i < miniBatchN; i++) {
			for (int j = 0; j < miniBatchSize; j++) {
				trainMiniBatch[i][j] = trainX[miniBatchIndex.get(i * miniBatchSize + j)];
			}
		}

		//
		// Build Restricted Boltzmann Machine Model
		//

		// construct RBM
		NeuralElement model = new SigmoidBinomialSamplingLayer(new Layer(new double[nVisible], new double[nHidden]),
				rand);

		for (int epoch = 0; epoch < epochs; epoch++) {
			if (epoch % 100 == 0) {
				System.out.println(epoch);
			}
			for (int batch = 0; batch < miniBatchN; batch++) {
				// train with contrastive divergence
				model = ContrastiveDivergence.train(trainMiniBatch[batch], 10, learningRate, model);
			}
			learningRate *= 0.995;
		}
		// test (reconstruct noised data)
		for (int i = 0; i < testN; i++) {
			reconstrX[i] = model.reconstruct(testX[i]);
		}

		// evaluation
		System.out.println("-----------------------------------");
		System.out.println("RBM model reconstruction evaluation");
		System.out.println("-----------------------------------");

		for (int pattern = 0; pattern < patterns; pattern++) {

			System.out.printf("Class%d\n", pattern + 1);

			for (int n = 0; n < testNEach; n++) {

				int n_ = pattern * testNEach + n;

				System.out.print(Arrays.toString(testX[n_]) + " -> ");
				System.out.print("[");
				int delta = 0;
				for (int i = 0; i < nVisible - 1; i++) {
					int val = reconstrX[n_][i] >= 0.5f ? 1 : 0;
					delta += Math.abs(val - testX[n_][i]);
					// System.out.printf("%.5f, ", reconstrX[n_][i]);
					System.out.print((val) + ", ");
				}
				int val = reconstrX[n_][nVisible - 1] >= 0.5 ? 1 : 0;
				// System.out.printf("%.5f]\n", reconstrX[n_][nVisible - 1]);
				delta += Math.abs(val - testX[n_][nVisible - 1]);
				System.out.print((val) + "]");
				System.out.println();
				System.out.println(">>>" + ((nVisible - delta) * 100f / (float) nVisible));
			}

			System.out.println();
		}

	}
}
