package rd.data;

import java.util.HashMap;
import java.util.Map;

import org.jblas.FloatMatrix;
/**
 * Delta inspector that samples weight change.
 * To prevent massive data overload use a low sample rate (~ 1e-5)
 * @author azahar
 *
 */
public class DeltaInspector implements DeepNetworkInspector {

	private final DataWriter dw;

	private final double sampleRate;
/**
 * Constructor with sample rate - we use uniformly distributed random numbers so lower the sampling rate, lower the number of samples generated.
 * To prevent data overload for large scale networks use ~1e-6 to 1e-5
 * @param sampleRate
 * @param dw
 */
	public DeltaInspector(double sampleRate, DataWriter dw) {
		this.dw = dw;
		this.sampleRate = sampleRate;
	}

	public DeltaInspector(DataWriter dw) {
		this.dw = dw;
		this.sampleRate = 1;
	}
	
	@Override
	public void inspectWeightChange(int layer, int count, FloatMatrix delta) {

		for (float item : delta.elementsAsList()) {
			if (Math.random() <= sampleRate) {
				Map<String, Object> data = new HashMap<>();
				data.put("Count", count);
				data.put("Layer", layer);
				data.put("Delta", item);
				dw.write(data);
			}
		}
	}

	@Override
	public void inspectBiasChange(int layer, int count, FloatMatrix delta) {
		// TODO Auto-generated method stub

	}

	@Override
	public void inspectOutput(int layer, int count, int neuron, float output) {
		// TODO Auto-generated method stub

	}

	@Override
	public void inspectInput(int layer, int count, int neuron, float... input) {
		// TODO Auto-generated method stub

	}

}
