package rd.deep.learning;

import java.util.Arrays;
import java.util.Random;

/**
 * Basic implementation of Restricted Boltzmann Machine as given in Java Deep
 * Learning Essentials (2016 - Yusuke Sugomori)
 * 
 * @author azahar
 *
 */
public class RBM {
	private final float[][] weights;
	private final float[] hbias, vbias;
	private final int nVisible, nHidden;
	private final Random rand;

	public RBM(int nVisible, int nHidden, float[][] exWeights, float exHbias[], float exVbias[], Random exRand) {

		this.nVisible = nVisible;
		this.nHidden = nHidden;

		if (exWeights == null) {
			this.weights = new float[nHidden][nVisible];
		} else {
			this.weights = exWeights;
		}

		if (exVbias == null) {
			this.vbias = new float[nVisible];
		} else {
			this.vbias = exVbias;
		}

		if (exHbias == null) {
			this.hbias = new float[nHidden];
		} else {
			this.hbias = exHbias;
		}

		if (exRand == null) {
			this.rand = new Random();
		} else {
			this.rand = exRand;
		}
	}

	public void contrastiveDivergence(int[][] X, int miniBatchSize, float learningRate, int cdIterations) {
		// Gradients
		float[][] gradWeights = new float[nHidden][nVisible];
		float[] gradHiddenBias = new float[nHidden];
		float[] gradVisibleBias = new float[nVisible];

		for (int n = 0; n < miniBatchSize; n++) {
			float[] pHiddenMean = new float[nHidden];
			int[] pHiddenSample = new int[nHidden];

			float[] nVisibleMean = new float[nVisible];
			int[] nVisibleSample = new int[nVisible];

			float[] nHiddenMean = new float[nHidden];
			int[] nHiddenSample = new int[nHidden];

			for (int step = 0; step < cdIterations; step++) {
				sampleHGivenV(X[n], pHiddenMean, pHiddenSample);
			}

			// Gibbs Sampling
			for (int step = 0; step < cdIterations; step++) {
				if (step == 0) {
					gibbsHVH(pHiddenSample, nVisibleMean, nVisibleSample, pHiddenMean, nHiddenSample);
				} else {
					gibbsHVH(nHiddenSample, nVisibleMean, nVisibleSample, nHiddenMean, nHiddenSample);
				}
			}

			// Calculate Gradient
			for (int j = 0; j < nHidden; j++) {
				for (int i = 0; i < nVisible; i++) {
					gradWeights[j][i] += (pHiddenMean[j] * X[n][i]) - (nHiddenMean[j] * nVisibleSample[i]);
				}

				gradHiddenBias[j] += pHiddenMean[j] - nHiddenMean[j];
			}

			for (int i = 0; i < nVisible; i++) {
				gradVisibleBias[i] += X[n][i] - nVisibleSample[i];
			}

			// Update Parameters
			for (int j = 0; j < nHidden; j++) {
				for (int i = 0; i < nVisible; i++) {
					weights[j][i] += learningRate * (gradWeights[j][i] / miniBatchSize);
				}

				hbias[j] += learningRate * (gradHiddenBias[j] / miniBatchSize);
			}

			for (int i = 0; i < nVisible; i++) {
				vbias[i] += learningRate * (gradVisibleBias[i] / miniBatchSize);
			}
		}

	}

	public float[] reconstruct(int[] visibleIn) {
		float[] output = new float[nVisible];
		float[] hiddenIn = new float[nHidden];

		for (int j = 0; j < nHidden; j++) {
			hiddenIn[j] = propUp(visibleIn, j);
		}

		for (int i = 0; i < nVisible; i++) {
			float preActivation = vbias[i];

			for (int j = 0; j < nHidden; j++) {
				preActivation += weights[j][i] * hiddenIn[j];
			}

			output[i] = sigmoid(preActivation);
		}

		return output;
	}

	public void gibbsHVH(int[] h0Sample, float[] nvMeans, int[] nvSample, float[] nhMeans, int[] nhSample) {
		sampleVGivenH(h0Sample, nvMeans, nvSample);
		sampleHGivenV(nvSample, nhMeans, nhSample);

	}

	public void sampleHGivenV(int[] v0Sample, float[] mean, int[] sample) {
		System.out.println("\nV Sample: "+Arrays.toString(v0Sample));
		System.out.println("Means: "+Arrays.toString(mean));
		System.out.println("Sample: "+Arrays.toString(sample));
		for (int j = 0; j < nHidden; j++) {
			mean[j] = propUp(v0Sample, j);
			sample[j] = binomial(1, mean[j]);
		}
	}

	public void sampleVGivenH(int[] h0Sample, float[] mean, int[] sample) {
		System.out.println("\nH Sample: "+Arrays.toString(h0Sample));
		System.out.println("Means: "+Arrays.toString(mean));
		System.out.println("Sample: "+Arrays.toString(sample));
		for (int i = 0; i < nVisible; i++) {
			mean[i] = propDown(h0Sample, i);
			sample[i] = binomial(1, mean[i]);
		}
	}

	public float propUp(int[] visible, int indexHid) {
		float preActivation = hbias[indexHid];

		for (int i = 0; i < nVisible; i++) {
			preActivation += weights[indexHid][i] * visible[i];
		}

		return sigmoid(preActivation);
	}

	public float propDown(int[] hidden, int indexVis) {
		float preActivation = vbias[indexVis];

		for (int j = 0; j < nHidden; j++) {
			preActivation += weights[j][indexVis] * hidden[j];
		}

		return sigmoid(preActivation);
	}

	private float sigmoid(float in) {
		return 1f / (float) (1 + Math.exp(-in));
	}

	private int binomial(int numberOfSamples, float probability) {
		if (probability < 0 || probability > 1) {
			return 0;
		}

		int countP = 0;
		for (int i = 0; i < numberOfSamples; i++) {
			if (rand.nextDouble() < probability) {
				countP++;
			}
		}

		return countP;
	}

	public static int binomial(int numberOfSamples, float probability, Random random) {
		if (probability < 0 || probability > 1) {
			return 0;
		}

		int countP = 0;
		for (int i = 0; i < numberOfSamples; i++) {
			if (random.nextDouble() < probability) {
				countP++;
			}
		}

		return countP;
	}
}
