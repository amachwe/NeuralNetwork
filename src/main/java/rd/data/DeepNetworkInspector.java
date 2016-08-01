package rd.data;

import org.jblas.FloatMatrix;
/**
 * To provide deep network inspection capabilities
 * @author azahar
 *
 */
public interface DeepNetworkInspector {

	/**
	 * For inspecting delta for weights
	 * @param layer - layer number
	 * @param count - count of iteration
	 * @param delta - delta value
	 */
	void inspectWeightChange(int layer,int count,FloatMatrix delta);
	/**
	 * For inspecting delta for bias
	 * @param layer - layer number
	 * @param count - count of iteration
	 * @param delta - bias delta value
	 */
	void inspectBiasChange(int layer,int count,FloatMatrix bias);
	/**
	 * For inspecting delta for weights
	 * @param layer - layer number
	 * @param count - count of iteration
	 * @param neuron - neuron count in layer
	 * @param output - output value of neuron
	 */
	void inspectOutput(int layer,int count,int neuron,float output);
	
	/**
	 * For inspecting delta for weights
	 * @param layer - layer number
	 * @param count - count of iteration
	 * @param neuron - neuron count in layer
	 * @param input - input values of neuron
	 */
	void inspectInput(int layer,int count,int neuron, float...input);
	
}
