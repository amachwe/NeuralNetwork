package rd.neuron.neuron;

import org.jblas.FloatMatrix;
/**
 * Train the Network 
 * @author azahar
 *
 */
public class TrainNetwork {

	/**
	 * Train the Simple Network
	 * @param network - network to be trained
	 * @param input - input value
	 * @param output -corresponding expected output value
	 * @param learningRate - constant learning rate (0.05 - 0.01)
	 */
	public static void trainBackprop(SimpleNetwork network,FloatMatrix input,FloatMatrix output,float learningRate)
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
	
	/**
	 * Train the Simple Network with L2
	 * @param network - network to be trained
	 * @param input - input value
	 * @param output -corresponding expected output value
	 * @param learningRate - constant learning rate (0.05 - 0.01)
	 */
	public static void trainBackpropWithL2(SimpleNetwork network,FloatMatrix input,FloatMatrix output,float learningRate,float beta)
	{
		FloatMatrix outputLayerNewWts[] = network.trainOutputLayerWithL2(learningRate, beta,output,
				network.io(input));
		FloatMatrix hiddenLayerNewWts[] = network.trainHiddenLayerWithL2(0,learningRate,beta, output,
				network.io(input), input);

		network.setOutputWeights(outputLayerNewWts[0]);
		network.setOutputBias(outputLayerNewWts[1]);
	    network.setWeights(0, hiddenLayerNewWts[0]);
		network.setBias(0, hiddenLayerNewWts[1]);
	}
}
