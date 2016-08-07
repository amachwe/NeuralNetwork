package rd.learning.statistics;

import org.jblas.FloatMatrix;

import rd.data.DataStreamer;

/**
 * Multi Class Logistic Regression
 * 
 * @author azahar
 *
 */
public class MulticlassLogReg {

	private FloatMatrix weights;
	private FloatMatrix bias;
	private final int classCount, inputCount;

	/**
	 * 
	 * @param input
	 *            - number of inputs
	 * @param classes
	 *            - number of classes
	 */
	public MulticlassLogReg(int input, int classes) {
		this.classCount = classes;
		this.inputCount = input;

		weights = FloatMatrix.rand(inputCount, classCount);
		bias = FloatMatrix.rand(classCount, 1);

	}

	/**
	 * Predict probability profile across all classes, given the input
	 * 
	 * @param input
	 * @return probability profile across all classes
	 */
	public FloatMatrix predict(FloatMatrix input) {
		FloatMatrix score = new FloatMatrix(1, this.classCount);

		float max = 0;
		for (int classN = 0; classN < this.classCount; classN++) {
			float tempValue = weights.getColumn(classN).mul(input).sum() + bias.get(classN);
			score.put(classN, tempValue);

			if (max < tempValue) {
				max = tempValue;
			}
		}
		for (int classN = 0; classN < this.classCount; classN++) {
			if (score.get(classN) == max) {
				score.put(classN, 1);
			} else {
				score.put(classN, 0);
			}

		}

		return score;
	}

	/**
	 * Get the Weights
	 * 
	 * @return - a COPY of the weight
	 */
	public FloatMatrix getWeights() {
		return (new FloatMatrix()).copy(weights);
	}

	/**
	 * Get the bias
	 * 
	 * @return - a COPY of the bias
	 */
	public FloatMatrix getBias() {
		return (new FloatMatrix()).copy(bias);
	}

	/**
	 * Train the Multi-Class Log Reg model
	 * 
	 * @param ds
	 *            - data stream
	 * @param learningRate
	 *            - learning rate
	 */
	public void train(DataStreamer ds, float learningRate) {

		FloatMatrix update_w = FloatMatrix.zeros(inputCount, classCount);
		FloatMatrix update_b = FloatMatrix.zeros(classCount, 1);

		for (FloatMatrix item : ds) {
			FloatMatrix actualOutputs = predict(item);

			for (int classN = 0; classN < this.classCount; classN++) {
				FloatMatrix output = ds.getOutput(item);

				float actualOutput = actualOutputs.get(classN);
				float expOutput = output.get(classN);
				float delta = actualOutput - expOutput;

				update_w.putColumn(classN, update_w.getColumn(classN).add(item.mul(delta)));
				update_b.put(classN, update_b.get(classN) + delta);

			}

		}

		weights = weights.sub(update_w.mul(learningRate));
		bias = bias.sub(update_b.mul(learningRate));
	}
}
