
package rd.neuron.neuron;

import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import rd.data.TimedDistributionStructure;

/**
 * Layer Class represents a layer of neurons and weights to input of that layer.
 * 
 * @author azahar
 */
public class Layer implements LayerIf {

	private static final Logger logger = LoggerFactory.getLogger(Layer.class);
	protected FloatMatrix weights;
	protected FloatMatrix bias;
	protected FloatMatrix inputBias;
	protected final Function function;
	private LayerType type;
	private int index;

	private FloatMatrix outputNet;
	private FloatMatrix outputActual;

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

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.neuron.neuron.LayerIf#getWeights()
	 */
	@Override
	public FloatMatrix getWeights() {
		return this.weights;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.neuron.neuron.LayerIf#setWeight(int, int, float)
	 */
	@Override
	public void setWeight(int inNeuron, int outNeuron, float newWt) {
		weights.put(inNeuron, outNeuron, newWt);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.neuron.neuron.LayerIf#setAllWeights(org.jblas.FloatMatrix)
	 */
	@Override
	public void setAllWeights(FloatMatrix newWeights) {
		this.weights = newWeights;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.neuron.neuron.LayerIf#getWeight(int, int)
	 */
	@Override
	public float getWeight(int inNeuron, int outNeuron) {
		return weights.get(inNeuron, outNeuron);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.neuron.neuron.LayerIf#setBias(int, float)
	 */
	@Override
	public void setBias(int inNeuron, float newBias) {
		bias.put(inNeuron, 0, newBias);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.neuron.neuron.LayerIf#setAllBias(org.jblas.FloatMatrix)
	 */
	@Override
	public void setAllBias(FloatMatrix bias) {
		this.bias = bias;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.neuron.neuron.LayerIf#getBias(int)
	 */
	@Override
	public float getBias(int inNeuron) {
		return bias.get(inNeuron, 0);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.neuron.neuron.LayerIf#getAllBias()
	 */
	@Override
	public FloatMatrix getAllBias() {
		return bias;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.neuron.neuron.LayerIf#io(org.jblas.FloatMatrix)
	 */
	@Override
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

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.neuron.neuron.LayerIf#getNetOutput()
	 */
	@Override
	public FloatMatrix getNetOutput() {
		return this.outputNet;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.neuron.neuron.LayerIf#getActualOutput()
	 */
	@Override
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

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.neuron.neuron.LayerIf#oi(org.jblas.FloatMatrix)
	 */
	@Override
	public FloatMatrix oi(FloatMatrix input) {
		if (inputBias == null) {
			logger.warn("Error: This layer is not enabled for reverse propagation.");
			return FloatMatrix.EMPTY;
		}
		return revIO(input);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.neuron.neuron.LayerIf#revIO(org.jblas.FloatMatrix)
	 */
	@Override
	public FloatMatrix revIO(FloatMatrix input) {

		FloatMatrix output = weights.mmul(input);

		output = output.add(inputBias);

		for (int i = 0; i < output.getRows(); i++) {
			if (function == Function.LOGISTIC) {
				output.put(i, 0, 1f / (float) (1 + Math.exp(-output.get(i, 0))));
			} else {
				output.put(i, 0, output.get(i, 0) > 0 ? output.get(i, 0) : 0f);
			}
		}

		return output;
	}

	@Override
	public void train(FloatMatrix input, int iter, float learningRate) {
		return;

	}
	
	

	@Override
	public void setDistHV(TimedDistributionStructure<String, String> tds) {
		throw new UnsupportedOperationException();

	}

	@Override
	public void setLayerIdentity(int index, LayerType type) {
		this.index = index;
		this.type = type;

	}

	@Override
	public int getLayerIndex() {
		return this.index;
	}

	@Override
	public LayerType getLayerType() {
		return this.type;
	}

}
