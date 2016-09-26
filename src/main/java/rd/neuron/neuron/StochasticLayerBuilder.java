package rd.neuron.neuron;

import java.util.Random;

import org.jblas.FloatMatrix;

public class StochasticLayerBuilder {

	public static LayerIf build(int noOfInputNeurons, int noOfOutputNeurons, Random rnd) {
		FloatMatrix weights = FloatMatrix.zeros(noOfInputNeurons, noOfOutputNeurons);
		FloatMatrix bias = FloatMatrix.zeros(noOfOutputNeurons);
		FloatMatrix inputBias = FloatMatrix.zeros(noOfInputNeurons);

		return new StochasticLayer(weights, bias, inputBias, rnd);
	}
	
	public static LayerIf build(int noOfInputNeurons, int noOfOutputNeurons)
	{
		return build(noOfInputNeurons,noOfOutputNeurons,new Random(System.currentTimeMillis()));
	}

}
