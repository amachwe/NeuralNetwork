package rd.neuron.neuron;

import java.util.ArrayList;
import java.util.List;

import org.jblas.FloatMatrix;

public class Propagate {

	/**
	 * Do up-down for given layer and collect samples.
	 * 
	 * @param input
	 * @param layer
	 * @param iter
	 * @return
	 */
	public static final List<FloatMatrix> upDown(FloatMatrix input, LayerIf layer, int iter) {
		List<FloatMatrix> result = new ArrayList<>();
		result.add(input);
		FloatMatrix tempResult = layer.io(input);
		result.add(tempResult);
		for (int i = 0; i < iter; i++) {
			tempResult = layer.oi(tempResult);
			result.add(tempResult);
			tempResult = layer.io(tempResult);
			result.add(tempResult);
		}

		return result;
	}

	public static final List<FloatMatrix> upWithIntermediateResults(FloatMatrix input, List<LayerIf> network) {
		List<FloatMatrix> result = new ArrayList<>();
		result.add(input);
		FloatMatrix temp = null;
		for (LayerIf l : network) {
			if (temp == null) {
				temp = l.io(input);
				result.add(temp);
			} else {
				temp = l.io(temp);
				result.add(temp);
			}
		}

		return result;
	}

	/**
	 * Up prop from Input to Output
	 * 
	 * @param input
	 * @param network
	 * @param maxIndex
	 *            - max layer to prop to
	 * @return
	 */
	public static final FloatMatrix up(FloatMatrix input, List<LayerIf> network, int maxIndex) {

		if (maxIndex <= 0) {
			maxIndex = network.size() + 1;
		}
		FloatMatrix temp = null;
		for (LayerIf l : network) {
			if (l.getLayerIndex() < maxIndex) {
				if (temp == null) {
					temp = l.io(input);

				} else {
					temp = l.io(temp);
				}
			} else {
				break;
			}
		}

		return temp;
	}

	/**
	 * Up prop one layer up
	 * 
	 * @param input
	 * @Param layer
	 * @return
	 */
	public static final FloatMatrix upOne(FloatMatrix input, LayerIf layer) {

		if (layer != null) {
			return layer.io(input);
		}

		return null;
	}

	/**
	 * Full prop from input to output
	 * 
	 * @param input
	 * @param network
	 * @return
	 */
	public static final FloatMatrix up(FloatMatrix input, List<LayerIf> network) {

		return up(input, network, -1);
	}

	/**
	 * Down Prop from Output towards Input
	 * 
	 * @param input
	 * @param network
	 * @param minIndex
	 *            - limit down prop
	 * @return
	 */
	public static final FloatMatrix down(FloatMatrix input, List<LayerIf> network, int minIndex) {

		FloatMatrix temp = null;
		for (int i = network.size() - 1; i >= 0; i--) {
			LayerIf l = network.get(i);

			if (l.getLayerIndex() >= minIndex) {

				if (temp == null) {

					temp = l.oi(input);

				} else {

					temp = l.oi(temp);
				}
			} else {
				break;
			}
		}

		return temp;
	}

	/**
	 * Full Down Prop from Output to Input
	 * 
	 * @param input
	 * @param network
	 * @return
	 */
	public static final FloatMatrix down(FloatMatrix input, List<LayerIf> network) {

		return down(input, network, -1);
	}
}
