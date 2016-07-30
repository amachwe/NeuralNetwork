package rd.neuron.neuron;

import org.jblas.FloatMatrix;

public class TrainNetwork {

	public static void train(SimpleNetwork network,FloatMatrix input,FloatMatrix output,float learningRate)
	{
		FloatMatrix outputLayerNewWts[] = network.trainOutputLayer(learningRate, output,
				network.io(input));
		FloatMatrix hiddenLayerNewWts[] = network.trainHiddenLayer(0,learningRate, output,
				network.io(input), input);

		network.setOutputWeights(outputLayerNewWts[0]);
		network.setOutputBias(outputLayerNewWts[1]);
	    network.setWeights(0, hiddenLayerNewWts[0]);
		network.setBias(0, hiddenLayerNewWts[1]);
	}
}
