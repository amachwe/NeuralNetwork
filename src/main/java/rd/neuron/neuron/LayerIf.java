package rd.neuron.neuron;

import org.jblas.FloatMatrix;

import rd.data.TimedDistributionStructure;

public interface LayerIf {

	public enum LayerType {
		FIRST_HIDDEN, HIDDEN, LAST_HIDDEN, OUTPUT
	};

	FloatMatrix getWeights();

	/**
	 * 
	 * @param inNeuron
	 *            - 0 indexed
	 * @param outNeuron
	 *            - 0 indexed
	 * @param newWt
	 *            - new weight
	 */
	void setWeight(int inNeuron, int outNeuron, float newWt);

	void setAllWeights(FloatMatrix newWeights);

	float getWeight(int inNeuron, int outNeuron);

	void setBias(int inNeuron, float newBias);

	void setAllBias(FloatMatrix bias);

	float getBias(int inNeuron);

	FloatMatrix getAllBias();

	/**
	 * Propagate input from previous layer
	 * 
	 * @param input
	 *            - input vector to this layer
	 * @return output for next layer
	 */
	FloatMatrix io(FloatMatrix input);

	FloatMatrix getNetOutput();

	FloatMatrix getActualOutput();

	FloatMatrix oi(FloatMatrix input);

	/**
	 * Propagate input from previous layer
	 * 
	 * @param input
	 *            - input vector to this layer
	 * @return output for next layer
	 */
	FloatMatrix revIO(FloatMatrix input);

	/**
	 * Train the layer
	 * 
	 * @param input
	 *            - input for the training
	 * @param iter
	 *            - iterations
	 * @param learningRate
	 *            - learning rate
	 */
	void train(FloatMatrix input, int iter, float learningRate);

	/**
	 * set the Distribution Structure
	 * 
	 * @param tds
	 */
	public void setDistHV(TimedDistributionStructure<String, String> tds);

	public void setLayerIdentity(int index, LayerType type);
	public int getLayerIndex();
	public LayerType getLayerType();

}