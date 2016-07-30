package rd.data;

import java.util.HashMap;
import java.util.Map;

import org.jblas.FloatMatrix;

public class DeltaInspector implements DeepNetworkInspector {

	private final DataWriter dw;

	private final double sampleRate;

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
