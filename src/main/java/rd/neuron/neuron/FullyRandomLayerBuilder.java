package rd.neuron.neuron;

import org.jblas.FloatMatrix;

import rd.neuron.neuron.Layer.Function;
/**
 * Build a fully connected random layer with a middle and max value
 * @author azahar
 *
 */
public class FullyRandomLayerBuilder implements LayerBuilder {

	private final float mid, max;

	/**
	 * 
	 * @param mid - middle value
	 * @param max - max value
	 */
	public FullyRandomLayerBuilder(float mid, float max) {
		this.mid = mid;
		this.max = max;
	}

	/**
	 * Default with random values between 0 and 1
	 */
	public FullyRandomLayerBuilder() {
		this.mid = 0;
		this.max = 1;
	}

	@Override
	public Layer build(int numberOfInputNeurons, int numberOfOutputNeurons, Function f) {
		if (numberOfInputNeurons <= 0 || numberOfOutputNeurons <= 0) {
			throw new IllegalArgumentException("Number of neurons cannont be <=0");
		}

		FloatMatrix weights = FloatMatrix.rand(numberOfInputNeurons, numberOfOutputNeurons); // Out
																								// x
																								// In

		FloatMatrix bias = FloatMatrix.rand(numberOfOutputNeurons, 1); // Out x
																		// 1

		if (max != 0 && mid != 0) {
			weights = weights.sub(mid).mul(max);
			bias = bias.sub(mid).mul(max);
		}
		return new Layer(weights, bias, f);
	}
}
