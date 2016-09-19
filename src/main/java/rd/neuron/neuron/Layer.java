
package rd.neuron.neuron;

import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Layer Class represents a layer of neurons and weights to input of that layer.
 * 
 * @author azahar
 */
public class Layer {

	private static final Logger logger = LoggerFactory.getLogger(Layer.class);
	protected FloatMatrix weights;
	protected FloatMatrix bias;
	protected FloatMatrix inputBias;
	protected final Function function;

	private FloatMatrix outputNet;
	private FloatMatrix outputActual;
	private FloatMatrix revOutputNet, revOutputActual;

	/**
	 * Activation function type
	 * 
	 * @author azahar
	 *
	 */
	public static enum Function {

		LOGISTIC, ReLU
	};

	/**
	 * 
	 * @param weights
	 *            - weights for the layer
	 * @param bias
	 *            - bias for the neurons
	 * @param function
	 *            - activation function for the layer
	 */
	public Layer(FloatMatrix weights, FloatMatrix bias, Function function) {
		this.function = function;
		this.inputBias = null;
		this.weights = weights;
		this.bias = bias;
	}

	/**
	 * 
	 * @param weights
	 * @param bias
	 * @param inputBias
	 *            - for reversable prop
	 * @param function
	 */
	public Layer(FloatMatrix weights, FloatMatrix bias, FloatMatrix inputBias, Function function) {
		this.function = function;
		this.inputBias = inputBias;
		this.weights = weights;
		this.bias = bias;
	}

	public FloatMatrix getWeights() {
		return this.weights;
	}

	/**
	 * 
	 * @param inNeuron
	 *            - 0 indexed
	 * @param outNeuron
	 *            - 0 indexed
	 * @param newWt
	 *            - new weight
	 */
	public void setWeight(int inNeuron, int outNeuron, float newWt) {
		weights.put(inNeuron, outNeuron, newWt);
	}

	public void setAllWeights(FloatMatrix newWeights) {
		this.weights = newWeights;
	}

	public float getWeight(int inNeuron, int outNeuron) {
		return weights.get(inNeuron, outNeuron);
	}

	public void setBias(int inNeuron, float newBias) {
		bias.put(inNeuron, 0, newBias);
	}

	public void setAllBias(FloatMatrix bias) {
		this.bias = bias;
	}

	public float getBias(int inNeuron) {
		return bias.get(inNeuron, 0);
	}

	public FloatMatrix getAllBias() {
		return bias;
	}

	/**
	 * Propagate input from previous layer
	 * 
	 * @param input
	 *            - input vector to this layer
	 * @return output for next layer
	 */
	public FloatMatrix io(FloatMatrix input) {
		FloatMatrix output = weights.transpose().mmul(input);
		output = output.add(bias);
		this.outputNet = output;
		for (int i = 0; i < output.getRows(); i++) {
			if (function == Function.LOGISTIC) {
				output.put(i, 0, 1f / (float) (1 + Math.exp(-output.get(i, 0))));
			} else {
				output.put(i, 0, output.get(i, 0) > 0 ? output.get(i, 0) : 0f);
			}
		}
		this.outputActual = output;
		return output;
	}

	public FloatMatrix getNetOutput() {
		return this.outputNet;
	}

	public FloatMatrix getActualOutput() {
		return this.outputActual;
	}

	/**
	 * Show string representation
	 */
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder("Layer Neuron Count: ");
		sb.append(weights.getRows());
		sb.append("    next Layer Count: ");
		sb.append(weights.getColumns());
		sb.append("\n" + "Weights: [" + weights.length + "] ");
		sb.append(weights);
		sb.append("\nBias: [" + bias.length + "] ");
		sb.append(bias);
		sb.append("\n\n");

		return sb.toString();

	}

	public FloatMatrix oi(FloatMatrix input) {
		if (inputBias == null) {
			logger.warn("Error: This layer is not enabled for reverse propagation.");
			return FloatMatrix.EMPTY;
		}
		return revIO(input);
	}

	/**
	 * Propagate input from previous layer
	 * 
	 * @param input
	 *            - input vector to this layer
	 * @return output for next layer
	 */
	public FloatMatrix revIO(FloatMatrix input) {

		FloatMatrix output = weights.mmul(input);

		output = output.add(inputBias);
		this.revOutputNet = output;
		for (int i = 0; i < output.getRows(); i++) {
			if (function == Function.LOGISTIC) {
				output.put(i, 0, 1f / (float) (1 + Math.exp(-output.get(i, 0))));
			} else {
				output.put(i, 0, output.get(i, 0) > 0 ? output.get(i, 0) : 0f);
			}
		}
		this.revOutputActual = output;
		return output;
	}
}
