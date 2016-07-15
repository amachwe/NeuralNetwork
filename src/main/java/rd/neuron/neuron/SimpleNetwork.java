package rd.neuron.neuron;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.jblas.FloatMatrix;

import rd.neuron.neuron.Layer.Function;

public class SimpleNetwork  implements Iterable<Layer>{

	private final Integer numberOfLayers;
	private final List<Layer> network = new ArrayList<>();
	private final Function activationFunction;

	public SimpleNetwork(LayerBuilder builder, Function f, int... numberOfNeuronsInLayer) {
		activationFunction = f;
		numberOfLayers = numberOfNeuronsInLayer.length-1;
		for (int i = 0; i < numberOfNeuronsInLayer.length - 1; i++) {
			network.add(builder.build(numberOfNeuronsInLayer[i], numberOfNeuronsInLayer[i + 1], f));
		}
	}

	public Integer getNumberOfLayers() {
		return numberOfLayers;
	}

	/**
	 * Get Weight
	 * 
	 * @param layer
	 *            - 0 indexed
	 * @param neuron
	 *            - source neuron; 0 indexed
	 * @param targetNeuron
	 *            - target neuron; 0 indexed

	 * @return weight
	 */
	public float getWeight(int layer, int neuron, int targetNeuron) {
		Layer l = network.get(layer);

		return l.getWeight(targetNeuron, neuron);
	}
	
	public FloatMatrix getWeights(int layer)
	{
		return network.get(layer).getWeights();
	}
	public void setWeights(int layer,FloatMatrix weightsNew)
	{
		Layer l = network.get(layer);
		
		l.setAllWeights(weightsNew);
	}

	public void setBias(int layer,FloatMatrix biasNew)
	{
		Layer l = network.get(layer);
		
		l.setAllBias(biasNew);
	}
	
	public void setOutputWeights(FloatMatrix weightsNew)
	{
		Layer l = network.get(this.numberOfLayers-1);
		
		l.setAllWeights(weightsNew);
	}

	public void setOutputBias(FloatMatrix biasNew)
	{
		Layer l = network.get(this.numberOfLayers-1);
		
		l.setAllBias(biasNew);
	}
	/**
	 * Adjust Weight
	 * 
	 * @param layer
	 *            - 0 indexed
	 * @param neuron
	 *            - source neuron; 0 indexed
	 * @param targetNeuron
	 *            - target neuron; 0 indexed
	 * @param wt
	 *            - weight
	 * @return old weight
	 */
	public float adjustWeight(int layer, int neuron, int targetNeuron, float wt) {
		Layer l = network.get(layer);
		float oldWt = l.getWeight(targetNeuron, neuron);
		l.setWeight(targetNeuron, neuron, wt);
		return oldWt;
	}


	public float adjustBias(int layer, int neuron, float bias) {
		Layer l = network.get(layer);
		float oldBias = l.getBias(neuron);
		l.setBias(neuron, bias);

		return oldBias;

	}
	
	

	public FloatMatrix io(FloatMatrix input) {
		return io(input, -1);
	}

	public FloatMatrix io(FloatMatrix input, int tillLayer) {
		FloatMatrix temp = null;
		int layerCount = 0;
		for (Layer l : network) {
			if (layerCount == tillLayer && tillLayer > 0) {
				break;
			}
			if (temp == null) {
				temp = l.io(input);
			} else {
				temp = l.io(temp);
			}
			layerCount++;
		}
		return temp;
	}
	
	public FloatMatrix trainOutputLayerWeights(float learningRate,FloatMatrix expected,FloatMatrix actuals)
	{
		//Get Weights to be trained.
		Layer output = network.get(this.numberOfLayers-1);
		Layer lastHidden = network.get(this.numberOfLayers-2);
		
		FloatMatrix outputActualLastHidden = lastHidden.getActualOutput();
		FloatMatrix grad = null;
		
		FloatMatrix expActuals = actuals.sub(expected);
		//System.out.println("-(Expected-Actuals): "+expActuals);
		
		switch(activationFunction)
		{
			case LOGISTIC:
				FloatMatrix activeDef = actuals.mul(FloatMatrix.ones(actuals.rows,actuals.columns).sub(actuals));
				grad = expActuals.mul(activeDef).mul(learningRate);
				//System.out.println("Activ Def: "+activeDef+"\nGrad: "+grad);
				break;
			case ReLU:
				grad = expActuals.mul(learningRate);
				break;
		}	
		//System.out.println(" \nHidden Layer Output: "+outputActual+"\nGrad': "+grad+"\nWeights to Output: "+weightsToOutput.getWeights());
		FloatMatrix delta = outputActualLastHidden.mmul(grad.transpose());
	
		return output.getWeights().sub(delta);
		//System.out.println(weightsToOutput.getWeights());
	}
	
	public FloatMatrix[] trainOutputLayer(float learningRate,FloatMatrix expected,FloatMatrix actuals)
	{
		//Get Weights to be trained.
		Layer output = network.get(this.numberOfLayers-1);
		Layer lastHidden = network.get(this.numberOfLayers-2);
		
		FloatMatrix outputActualLastHidden = lastHidden.getActualOutput();
		FloatMatrix grad = null;
		
		FloatMatrix expActuals = actuals.sub(expected);
		//System.out.println("-(Expected-Actuals): "+expActuals);
		
		switch(activationFunction)
		{
			case LOGISTIC:
				FloatMatrix activeDef = actuals.mul(FloatMatrix.ones(actuals.rows,actuals.columns).sub(actuals));
				grad = expActuals.mul(activeDef).mul(learningRate);
			
				break;
			case ReLU:
				grad = expActuals.mul(learningRate);
				break;
		}	
		FloatMatrix delta = outputActualLastHidden.mmul(grad.transpose());
		FloatMatrix newBias = output.getAllBias().sub(grad);
		
		
		return new FloatMatrix[]{output.getWeights().sub(delta),newBias};
	
	}
	
	public FloatMatrix[] trainHiddenLayer(int layer,float learningRate,FloatMatrix expected,FloatMatrix actuals,FloatMatrix input)
	{
		//Get Weights to be trained.
		Layer output = network.get(layer+1);
		Layer lastHidden = network.get(layer);
		
		
		FloatMatrix outputActualLastHidden = lastHidden.getActualOutput();
		FloatMatrix grad = null;
		FloatMatrix activeDef = null;
		FloatMatrix expActuals = actuals.sub(expected);
		
		switch(activationFunction)
		{
			case LOGISTIC:
				activeDef = actuals.mul(FloatMatrix.ones(actuals.rows,actuals.columns).sub(actuals));
				grad = expActuals.mul(activeDef);
				//System.out.println("Activ Def: "+activeDef+"\nGrad: "+grad);
				break;
			case ReLU:
				grad = expActuals;	
				break;
		}	
		
		FloatMatrix alpha = output.getWeights().mmul(grad);
		
		FloatMatrix delta = alpha.mul(outputActualLastHidden.mul(FloatMatrix.ones(outputActualLastHidden.rows,outputActualLastHidden.columns).sub(outputActualLastHidden)));
		
		FloatMatrix update = input.mmul(delta.transpose()).mul(learningRate);

	
		return new FloatMatrix[]{lastHidden.getWeights().sub(update),lastHidden.getAllBias().sub(grad.mul(learningRate))};
	}
	
	public FloatMatrix trainHiddenLayerWeights(int layer,float learningRate,FloatMatrix expected,FloatMatrix actuals,FloatMatrix input)
	{
		//Get Weights to be trained.
		Layer output = network.get(layer+1);
		Layer lastHidden = network.get(layer);
		
		
		FloatMatrix outputActualLastHidden = lastHidden.getActualOutput();
		FloatMatrix grad = null;
		FloatMatrix activeDef = null;
		FloatMatrix expActuals = actuals.sub(expected);
		
		switch(activationFunction)
		{
			case LOGISTIC:
				activeDef = actuals.mul(FloatMatrix.ones(actuals.rows,actuals.columns).sub(actuals));
				grad = expActuals.mul(activeDef);
				//System.out.println("Activ Def: "+activeDef+"\nGrad: "+grad);
				break;
			case ReLU:
				grad = expActuals;	
				break;
		}	
		
		FloatMatrix alpha = output.getWeights().mmul(grad);
		
		FloatMatrix delta = alpha.mul(outputActualLastHidden.mul(FloatMatrix.ones(outputActualLastHidden.rows,outputActualLastHidden.columns).sub(outputActualLastHidden)));
		delta = input.mmul(delta.transpose()).mul(learningRate);
		
	
		return lastHidden.getWeights().sub(delta);
	}
	
	public Layer getOutputLayer()
	{
		return network.get(this.numberOfLayers-1);
	}
	

	

	@Override
	public String toString()
	{
		StringBuilder strNetwork = new StringBuilder();
		for(Layer l : network)
		{
			strNetwork.append(l.toString());
		}
		
		return strNetwork.toString();
	}

	@Override
	public Iterator<Layer> iterator() {
		return network.iterator();
	}
}
