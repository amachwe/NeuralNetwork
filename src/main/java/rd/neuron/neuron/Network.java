package rd.neuron.neuron;

import org.jblas.FloatMatrix;

public interface Network {

	/**
	 * Train output layer
	 * 
	 * @param learningRate
	 *            - learning rate value (around 0.01 - 0.05)
	 * @param expected
	 *            - expected output
	 * @param actuals
	 *            - actual output
	 * @return new weights (index 0) and biases (index 1)
	 */
	FloatMatrix[] trainOutputLayer(float learningRate, FloatMatrix expected, FloatMatrix actuals);

	/**
	 * Train output layer with L2 Normalisation
	 * 
	 * @param learningRate
	 *            - learning rate value (around 0.01 - 0.05)
	 * @param expected
	 *            - expected output
	 * @param actuals
	 *            - actual output
	 * @return new weights (index 0) and biases (index 1)
	 */
	FloatMatrix[] trainOutputLayerWithL2(float learningRate, float beta, FloatMatrix expected, FloatMatrix actuals);

	/**
	 * Train Hidden Layer with L2 Normalisation
	 * @param layer
	 *            - train layer
	 * @param learningRate
	 *            - learning rate value (around 0.01 - 0.05)
	 * @param expected
	 *            - expected output (at output layer)
	 * @param actuals
	 *            - actual output (at output layer)
	 * @param input
	 *            - input
	 * @return new weights (index 0) and biases (index 1)
	 */
	FloatMatrix[] trainHiddenLayerWithL2(int layer, float learningRate, float beta, FloatMatrix expected,
			FloatMatrix actuals, FloatMatrix input);

	/**
	 * 
	 * @param layer
	 *            - train layer
	 * @param learningRate
	 *            - learning rate value (around 0.01 - 0.05)
	 * @param expected
	 *            - expected output (at output layer)
	 * @param actuals
	 *            - actual output (at output layer)
	 * @param input
	 *            - input
	 * @return new weights (index 0) and biases (index 1)
	 */
	FloatMatrix[] trainHiddenLayer(int layer, float learningRate, FloatMatrix expected, FloatMatrix actuals,
			FloatMatrix input);

}