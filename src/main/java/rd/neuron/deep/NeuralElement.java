package rd.neuron.deep;
/**
 * Neural Element interface for decorator
 * @author azahar
 *
 */
public interface NeuralElement {
/**
 * Direction of propagation
 * @author azahar
 *
 */
	public static enum Direction {
		Forward, Backward
	};

	/**
	 * Get visible neuron count
	 * @return
	 */
	int getVisibleNeuronCount();

	/**
	 * Get hidden neuron count
	 * @return
	 */
	int getHiddenNeuronCount();

	/**
	 * Process neural element
	 * @param in - in vector
	 * @param d - direction of processing
	 * @return
	 */
	double[] process(int[] in, Direction d);

	/**
	 * Post - processing
	 * @param in - in vector
	 * @return
	 */
	int[] postProcess(double[] in);

	void updateWeight(int indexIn, int indexOut, double delta);

	void updateHiddenBias(int indexNeuron, double delta);

	void updateVisibleBias(int indexNeuron, double delta);

	/**
	 * Reconstruct output (forward propagation)
	 * @param in - input vectors
	 * @return output
	 */
	double[] reconstruct(int[] in);
	
	double[] getInputBias();
	double[] getOutputBias();
	double[][] getWeights();
}
