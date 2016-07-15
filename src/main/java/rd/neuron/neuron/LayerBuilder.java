package rd.neuron.neuron;

import rd.neuron.neuron.Layer.Function;

public interface LayerBuilder {

	Layer build(int noOfInputNeurons,int noOfOutputNeurons,Function f);
	
}
