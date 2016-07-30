package rd.data;

import org.jblas.FloatMatrix;

public interface DeepNetworkInspector {

	void inspectWeightChange(int layer,int count,FloatMatrix delta);
	void inspectBiasChange(int layer,int count,FloatMatrix bias);
	void inspectOutput(int layer,int count,int neuron,float output);
	void inspectInput(int layer,int count,int neuron, float...input);
	
}
