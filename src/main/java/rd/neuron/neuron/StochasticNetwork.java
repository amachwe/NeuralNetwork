package rd.neuron.neuron;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;

import org.jblas.FloatMatrix;

import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.LayerIf.LayerType;

public class StochasticNetwork implements Network {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7807952804454760098L;
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

	public void preTrain(FloatMatrix[] in, int steps, float learningRate) {

		for (LayerIf layer : network) {
			for (FloatMatrix _in : in) {
				if (layer.getLayerType() == LayerType.FIRST_HIDDEN) {

					layer.train(_in, steps, learningRate);

				} else {

					if (layer.getLayerType() != LayerType.OUTPUT) {

						FloatMatrix _result = null;

						_result = Propagate.up(_in, network, layer.getLayerIndex());

						layer.train(_result, steps, learningRate);

					}

				}
			}
		}
	}

	public void preTrain(List<FloatMatrix> in, int steps, float learningRate) {

		int iterC = 0;
		for (LayerIf layer : network) {
			for (FloatMatrix _in : in) {
				if (layer.getLayerType() == LayerType.FIRST_HIDDEN) {

					layer.train(_in, steps, learningRate);

				} else {

					// if (layer.getLayerType() != LayerType.OUTPUT)
					{

						FloatMatrix _result = null;

						_result = Propagate.up(_in, network, layer.getLayerIndex());

						layer.train(_result, steps, learningRate);

					}

				}
			}
			iterC++;
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

	public static void save(String file, List<LayerIf> network) {
		try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
			oos.writeObject(network);
		} catch (Exception e) {
			e.printStackTrace(System.out);
		} finally {
			System.out.println("Written network.");
		}
	}

	public static List<LayerIf> load(String file) {
		try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
			Object obj = ois.readObject();
			if (obj instanceof List) {
				return (List<LayerIf>) obj;
			}
		} catch (Exception e) {
			e.printStackTrace(System.out);

		}
		return null;
	}
}
