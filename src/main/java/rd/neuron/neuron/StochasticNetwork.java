package rd.neuron.neuron;

import java.util.List;

import org.jblas.FloatMatrix;

import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.LayerIf.LayerType;

public class StochasticNetwork implements Network {

	private final List<LayerIf> network;
	private final int numberOfLayers;
	private final Function outputLayerFn, lastHiddenLayerFn;

	public StochasticNetwork(List<LayerIf> network, Function outputLayerFn, Function lastHiddenLayerFn) {
		this.network = network;
		this.numberOfLayers = network.size();
		this.outputLayerFn = outputLayerFn;
		this.lastHiddenLayerFn = lastHiddenLayerFn;
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
	@Override
	public FloatMatrix[] trainOutputLayer(float learningRate, FloatMatrix expected, FloatMatrix actuals) {
		// Get Weights to be trained.
		LayerIf output = network.get(this.numberOfLayers - 1);
		LayerIf lastHidden = network.get(this.numberOfLayers - 2);

		FloatMatrix outputActualLastHidden = lastHidden.getActualOutput();
		FloatMatrix grad = null;

		FloatMatrix expActuals = actuals.sub(expected);
		// System.out.println("-(Expected-Actuals): "+expActuals);

		switch (outputLayerFn) {
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

		return new FloatMatrix[] { output.getWeights().sub(delta), newBias };

	}

	public void fineTuneOutputLayer(FloatMatrix expectedOutput, FloatMatrix actualOutput, FloatMatrix input) {

		FloatMatrix[] outputLayerNewWtsB = trainOutputLayer(0.05f, expectedOutput, actualOutput);
		network.get(network.size() - 1).setAllWeights(outputLayerNewWtsB[0]);
		network.get(network.size() - 1).setAllBias(outputLayerNewWtsB[1]);

		FloatMatrix layerInput = input;
		if (network.size() >= 3) {
			layerInput = Propagate.up(input, network, network.size() - 2);
		}
		FloatMatrix[] hiddenLayerNewWtsB = trainHiddenLayer(network.size() - 2, 0.05f, expectedOutput, actualOutput,
				layerInput);
		network.get(network.size() - 2).setAllWeights(hiddenLayerNewWtsB[0]);
		network.get(network.size() - 2).setAllBias(hiddenLayerNewWtsB[1]);

	}

	public void preTrain(FloatMatrix in) {


		for (LayerIf layer : network) {

			if (layer.getLayerType() == LayerType.FIRST_HIDDEN) {
			

				layer.train(in, 10, 0.02f);

				

			} else {

				if (layer.getLayerType() != LayerType.OUTPUT) {
			

					FloatMatrix _result = null;

					_result = Propagate.up(in, network, layer.getLayerIndex());

					layer.train(_result, 10, 0.02f);

			
				}

			}
		}
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
	@Override
	public FloatMatrix[] trainHiddenLayer(int layer, float learningRate, FloatMatrix expected, FloatMatrix actuals,
			FloatMatrix input) {
		// Get Weights to be trained.
		LayerIf output = network.get(this.numberOfLayers - 1);
		LayerIf lastHidden = network.get(layer);

		FloatMatrix outputActualLastHidden = lastHidden.getActualOutput();
		FloatMatrix grad = null;
		FloatMatrix activeDef = null;
		FloatMatrix expActuals = actuals.sub(expected);

		switch (lastHiddenLayerFn) {
		case LOGISTIC:
			activeDef = actuals.mul(FloatMatrix.ones(actuals.rows, actuals.columns).sub(actuals));
			grad = expActuals.mul(activeDef);

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

		return new FloatMatrix[] { newWts, newBias };

	}

	@Override
	public FloatMatrix[] trainOutputLayerWithL2(float learningRate, float beta, FloatMatrix expected,
			FloatMatrix actuals) {
		throw new Error("Method not supported.");

	}

	@Override
	public FloatMatrix[] trainHiddenLayerWithL2(int layer, float learningRate, float beta, FloatMatrix expected,
			FloatMatrix actuals, FloatMatrix input) {

		throw new Error("Method not supported.");
	}
}
