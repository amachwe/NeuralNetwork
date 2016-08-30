package rd.neuron.deep;

public interface NeuralElement {

	public static enum Direction {
		Forward, Backward
	};

	int getVisibleNeuronCount();

	int getHiddenNeuronCount();

	double[] process(int[] in, Direction d);

	int[] postProcess(double[] in);

	void updateWeight(int indexIn, int indexOut, double delta);

	void updateHiddenBias(int indexNeuron, double delta);

	void updateVisibleBias(int indexNeuron, double delta);

	double[] reconstruct(int[] in);
	
	double[] getInputBias();
	double[] getOutputBias();
	double[][] getWeights();
}
