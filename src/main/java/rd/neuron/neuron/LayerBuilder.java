package rd.neuron.neuron;

import rd.neuron.neuron.Layer.Function;
/**
 * Builds different types of layers
 * @author azahar
 *
 */
public interface LayerBuilder {

	/**
	 * 
	 * @param noOfInputNeurons - number of input neurons
	 * @param noOfOutputNeurons - number of output neurons
	 * @param f - activation function
	 * @return Layer
	 */
	Layer build(int noOfInputNeurons,int noOfOutputNeurons,Function f);
	
}
