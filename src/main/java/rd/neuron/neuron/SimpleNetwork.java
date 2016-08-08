package rd.neuron.neuron;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.jblas.FloatMatrix;

import rd.data.DeepNetworkInspector;
import rd.neuron.neuron.Layer.Function;

/**
 * A simple feed forward Artificial Neural Network
 * 
 * @author azahar
 *
 */
public class SimpleNetwork implements Iterable<Layer> {

	private final Integer numberOfLayers;
	private final List<Layer> network = new ArrayList<>();
	private final Function activationFunction;
	private DeepNetworkInspector dni;
	private int trainCount = 0;

	/**
	 * 
	 * @param builder
	 *            - layer builder
	 * @param f
	 *            - activation function
	 * @param numberOfNeuronsInLayer
	 *            - number of neurons in each layer (array): index 0 is input
	 *            layer and last entry (array.length-1) the number of neurons in
	 *            the output layer
	 */
	public SimpleNetwork(LayerBuilder builder, Function f, int... numberOfNeuronsInLayer) {
		activationFunction = f;
		numberOfLayers = numberOfNeuronsInLayer.length - 1;
		for (int i = 0; i < numberOfNeuronsInLayer.length - 1; i++) {
			network.add(builder.build(numberOfNeuronsInLayer[i], numberOfNeuronsInLayer[i + 1], f));
		}
	}

	/**
	 * Activate deep network inspection
	 * 
	 * @param dni
	 */
	public void activateDeepNetworkInspection(DeepNetworkInspector dni) {
		this.dni = dni;
	}

	/**
	 * Get number of layers
	 * 
	 * @return
	 */
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
	 * 
	 * @return weight
	 */
	public float getWeight(int layer, int neuron, int targetNeuron) {
		Layer l = network.get(layer);

		return l.getWeight(targetNeuron, neuron);
	}

	/**
	 * Get Weights
	 * 
	 * @param layer
	 *            - 0 indexed
	 * @return
	 */
	public FloatMatrix getWeights(int layer) {
		return network.get(layer).getWeights();
	}

	/**
	 * Set Weights
	 * 
	 * @param layer
	 *            - 0 indexed
	 * @param weightsNew
	 *            - new weights matrix
	 */
	public void setWeights(int layer, FloatMatrix weightsNew) {
		Layer l = network.get(layer);

		l.setAllWeights(weightsNew);
	}

	/**
	 * Set Bias
	 * 
	 * @param layer
	 *            - 0 indexed
	 * @param biasNew
	 *            - new Bias vector
	 */
	public void setBias(int layer, FloatMatrix biasNew) {
		Layer l = network.get(layer);

		l.setAllBias(biasNew);
	}

	/**
	 * Set Weights of output layer
	 * 
	 * @param weightsNew
	 *            - new weights matrix for output layer
	 */
	public void setOutputWeights(FloatMatrix weightsNew) {
		Layer l = network.get(this.numberOfLayers - 1);

		l.setAllWeights(weightsNew);
	}

	/**
	 * Set New Output Bias
	 * 
	 * @param biasNew
	 *            - new Bias vector
	 */
	public void setOutputBias(FloatMatrix biasNew) {
		Layer l = network.get(this.numberOfLayers - 1);

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

	/**
	 * Adjust bias
	 * 
	 * @param layer
	 *            - 0 indexed
	 * @param neuron
	 *            - neuron number (0 indexed)
	 * @param bias
	 * @return
	 */
	public float adjustBias(int layer, int neuron, float bias) {
		Layer l = network.get(layer);
		float oldBias = l.getBias(neuron);
		l.setBias(neuron, bias);

		return oldBias;

	}

	/**
	 * Propagate from input to output - through the full network
	 * 
	 * @param input
	 *            - input values
	 * @return output of network
	 */
	public FloatMatrix io(FloatMatrix input) {
		return io(input, -1);
	}

	/**
	 * Propagate from input to output - through the full network
	 * 
	 * @param input
	 *            - input values
	 * @param outputs - to record the outputs
	 * @return output of network
	 */
	public FloatMatrix io(FloatMatrix input, Map<Integer, FloatMatrix> outputs) {
		return io(input, -1, outputs);
	}

	/**
	 * Partial propagation till end layer
	 * 
	 * @param input
	 *            - input values
	 * @param tillLayer
	 *            - 0 indexed layer index
	 * @return partial output
	 */
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

	/**
	 * Partial propagation till end layer
	 * 
	 * @param input
	 *            - input values
	 * @param tillLayer
	 *            - 0 indexed layer index
	 * @param outputs
	 *            - outputs from each layer - indexed by layer number
	 * @return partial output
	 */
	public FloatMatrix io(FloatMatrix input, int tillLayer, Map<Integer, FloatMatrix> outputs) {
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
			outputs.put(layerCount + 1, temp);
			layerCount++;
		}

		return temp;
	}

	/**
	 * Train output layer
	 * 
	 * @param learningRate
	 *            - learning rate value (around 0.01 - 0.05)
	 * @param expected
	 *            - expected output
	 * @param actuals
	 *            - actual output
	 * @return new weights (index 0) and biases (index 1)
	 */
	public FloatMatrix[] trainOutputLayer(float learningRate, FloatMatrix expected, FloatMatrix actuals) {
		// Get Weights to be trained.
		Layer output = network.get(this.numberOfLayers - 1);
		Layer lastHidden = network.get(this.numberOfLayers - 2);

		FloatMatrix outputActualLastHidden = lastHidden.getActualOutput();
		FloatMatrix grad = null;

		FloatMatrix expActuals = actuals.sub(expected);
		// System.out.println("-(Expected-Actuals): "+expActuals);

		switch (activationFunction) {
		case LOGISTIC:
			FloatMatrix activeDef = actuals.mul(FloatMatrix.ones(actuals.rows, actuals.columns).sub(actuals));
			grad = expActuals.mul(activeDef).mul(learningRate);

			break;
		case ReLU:
			grad = expActuals.mul(learningRate);
			break;
		}
		FloatMatrix delta = outputActualLastHidden.mmul(grad.transpose());
		FloatMatrix newBias = output.getAllBias().sub(grad);
		if (dni != null) {
			dni.inspectWeightChange(this.numberOfLayers - 1, trainCount, delta);
		}
		return new FloatMatrix[] { output.getWeights().sub(delta), newBias };

	}

	/**
	 * 
	 * @param layer
	 *            - train layer
	 * @param learningRate
	 *            - learning rate value (around 0.01 - 0.05)
	 * @param expected
	 *            - expected output (at output layer)
	 * @param actuals
	 *            - actual output (at output layer)
	 * @param input
	 *            - input
	 * @return new weights (index 0) and biases (index 1)
	 */
	public FloatMatrix[] trainHiddenLayer(int layer, float learningRate, FloatMatrix expected, FloatMatrix actuals,
			FloatMatrix input) {
		// Get Weights to be trained.
		Layer output = network.get(layer + 1);
		Layer lastHidden = network.get(layer);

		FloatMatrix outputActualLastHidden = lastHidden.getActualOutput();
		FloatMatrix grad = null;
		FloatMatrix activeDef = null;
		FloatMatrix expActuals = actuals.sub(expected);

		switch (activationFunction) {
		case LOGISTIC:
			activeDef = actuals.mul(FloatMatrix.ones(actuals.rows, actuals.columns).sub(actuals));
			grad = expActuals.mul(activeDef);
			// System.out.println("Activ Def: "+activeDef+"\nGrad: "+grad);
			break;
		case ReLU:
			grad = expActuals;
			break;
		}

		FloatMatrix alpha = output.getWeights().mmul(grad);

		FloatMatrix delta = alpha.mul(outputActualLastHidden.mul(FloatMatrix
				.ones(outputActualLastHidden.rows, outputActualLastHidden.columns).sub(outputActualLastHidden)));

		FloatMatrix update = input.mmul(delta.transpose()).mul(learningRate);
		FloatMatrix newWts = lastHidden.getWeights().sub(update);

		FloatMatrix newBias = lastHidden.getAllBias().sub(grad.mean() * learningRate);
		if (dni != null) {
			dni.inspectWeightChange(this.numberOfLayers - 2, trainCount, update);
		}
		trainCount++;

		return new FloatMatrix[] { newWts, newBias };

	}

	/**
	 * Get output Layer
	 * 
	 * @return
	 */
	public Layer getOutputLayer() {
		return network.get(this.numberOfLayers - 1);
	}

	@Override
	public String toString() {
		StringBuilder strNetwork = new StringBuilder();
		for (Layer l : network) {
			strNetwork.append(l.toString());
		}

		return strNetwork.toString();
	}

	@Override
	public Iterator<Layer> iterator() {
		return network.iterator();
	}
}
