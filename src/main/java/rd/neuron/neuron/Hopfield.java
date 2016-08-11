package rd.neuron.neuron;

/**
 * The classic Hopfield Network
 * 
 * @author azahar
 *
 */
public class Hopfield {
	private final float[][] weights;
	private final float[] states;
	private final float[] thresholds;
	private final int numberOfNeurons;

	/**
	 * 
	 * @param numberOfNeurons - for n x n matrix
	 */
	public Hopfield(int numberOfNeurons) {
		this.numberOfNeurons = numberOfNeurons;
		weights = new float[numberOfNeurons][numberOfNeurons];
		states = new float[numberOfNeurons];
		thresholds = new float[numberOfNeurons];
	}

	/**
	 * Calculate Energy
	 * @return
	 */
	public float calculateEnergy() {
		getStates(states);
		float energy = 0;
		for (int i = 0; i < numberOfNeurons; i++) {
			for (int j = 0; j < numberOfNeurons; j++) {
				energy += weights[i][j] * states[i] * states[j];
			}

			energy += thresholds[i] * states[i];
		}

		return -0.5f*energy;
	}

	public float[] getStates(float[] input) {
		for (int i = 0; i < numberOfNeurons; i++) {
			float sum = 0f;
			for (int j = 0; j < numberOfNeurons; j++) {
				sum += weights[i][j] * input[j];
			}
			if (sum > thresholds[i]) {
				states[i] = 1;
			} else {
				states[i] = 0;
			}
		}

		return states;
	}

	/**
	 * Hebbian Learning - eights are updated straight away
	 * 
	 * @param pattern
	 * @param numberOfPatterns
	 * @return current weights
	 */
	public float[][] addPatternHebbian(float[] pattern, int numberOfPatterns) {
		if (numberOfPatterns <= 0) {
			throw new IllegalArgumentException("Number of patterns must be greater than zero");
		}
		for (int i = 0; i < numberOfNeurons; i++) {
			for (int j = 0; j < numberOfNeurons; j++) {
				if (i == j) {
					weights[i][j] = 0;
				} else {
					weights[i][j] += (pattern[i] * pattern[j] / numberOfPatterns);
				}
			}

		}

		return weights;
	}

	/**
	 * Get weights
	 * @return
	 */
	public float[][] getWeights() {
		return weights;
	}

	/**
	 * Get weights as string
	 * @return
	 */
	public String getWeightsAsString() {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < numberOfNeurons; i++) {
			for (int j = 0; j < numberOfNeurons; j++) {
				sb.append(i).append(",").append(j).append("->").append(weights[i][j]).append("\n");
			}
		}

		return sb.toString();
	}
}
