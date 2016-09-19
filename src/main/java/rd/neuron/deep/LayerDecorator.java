package rd.neuron.deep;

public abstract class LayerDecorator implements NeuralElement {

	protected final NeuralElement l;

	public LayerDecorator(NeuralElement l) {
		this.l = l;
	}

	public abstract double[] process(int[] in, Direction d);

	public abstract int[] postProcess(double[] in);

	@Override
	public void updateHiddenBias(int indexNeuron, double delta) {
		this.l.updateHiddenBias(indexNeuron, delta);

	}

	@Override
	public void updateVisibleBias(int indexNeuron, double delta) {
		this.l.updateVisibleBias(indexNeuron, delta);

	}

	@Override
	public void updateWeight(int indexIn, int indexOut, double delta) {
		this.l.updateWeight(indexIn, indexOut, delta);
	}

	@Override
	public int getVisibleNeuronCount() {
		return this.l.getVisibleNeuronCount();
	}

	@Override
	public int getHiddenNeuronCount() {
		return this.l.getHiddenNeuronCount();
	}

	@Override
	public double[] getInputBias() {
		return this.l.getInputBias();
	}

	@Override
	public double[] getOutputBias() {
		return this.l.getOutputBias();
	}

	@Override
	public double[][] getWeights() {
		return this.l.getWeights();
	}
	
	
	public abstract double[] reconstruct(int[] in);
}
