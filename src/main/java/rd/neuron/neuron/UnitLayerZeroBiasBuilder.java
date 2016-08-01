package rd.neuron.neuron;

import org.jblas.FloatMatrix;

import rd.neuron.neuron.Layer.Function;

/**
 * Build so that weights are set to 1.0 and bias to 0.0
 * @author azahar
 *
 */
public class UnitLayerZeroBiasBuilder implements LayerBuilder{

	@Override
	public Layer build(int numberOfInputNeurons,int numberOfOutputNeurons,Function f)
	{
		if(numberOfInputNeurons<=0 || numberOfOutputNeurons<=0)
		{
			throw new IllegalArgumentException("Number of neurons cannont be <=0");
		}
		
		FloatMatrix weights = FloatMatrix.ones(numberOfInputNeurons,numberOfOutputNeurons);  //Out x In
		FloatMatrix bias = FloatMatrix.zeros(numberOfOutputNeurons,1);  //Out x 1
		
		return new Layer(weights,bias,f);
	}
}
