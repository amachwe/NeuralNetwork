/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rd.neuron.neuron;

import org.jblas.FloatMatrix;

/**
 *
 * @author azahar
 */
public class Layer {

	private FloatMatrix weights;
	private FloatMatrix bias;
	private final Function function;

	private FloatMatrix outputNet;
	private FloatMatrix outputActual;

	public static enum Function {

		LOGISTIC, ReLU
	};

	public Layer(FloatMatrix weights, FloatMatrix bias, Function function) {
		this.function = function;

		this.weights = weights;
		this.bias = bias;
	}

	public FloatMatrix getWeights() {
		return this.weights;
	}

	public void setWeight(int inNeuron, int outNeuron, float newWt) {
		weights.put(inNeuron, outNeuron, newWt);
	}

	public void setAllWeights(FloatMatrix newWeights) {
		this.weights = newWeights;
	}

	public float getWeight(int inNeuron, int outNeuron) {
		return weights.get(inNeuron, outNeuron);
	}

	public void setBias(int inNeuron, float newBias) {
		bias.put(inNeuron, 0, newBias);
	}

	public void setAllBias(FloatMatrix bias)
	{
		this.bias = bias;
	}
	public float getBias(int inNeuron) {
		return bias.get(inNeuron, 0);
	}

	public FloatMatrix getAllBias()
	{
		return bias;
	}
	public FloatMatrix io(FloatMatrix input) {
		
		FloatMatrix output = weights.transpose().mmul(input);
		output = output.add(bias);
		this.outputNet = output;
		for (int i = 0; i < output.getRows(); i++) {
			if (function == Function.LOGISTIC) {
				output.put(i, 0, 1f / (float) (1 + Math.exp(-output.get(i, 0))));
			} else {
				output.put(i, 0, output.get(i, 0) > 0 ? output.get(i, 0) : 0f);
			}
		}
		this.outputActual = output;
		return output;
	}

	public FloatMatrix getNetOutput() {
		return this.outputNet;
	}

	public FloatMatrix getActualOutput() {
		return this.outputActual;
	}

	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder("Layer Neuron Count: ");
		sb.append(weights.getRows());
		sb.append("    next Layer Count: ");
		sb.append(weights.getColumns());
		sb.append("\n"+"Weights: ["+weights.length+"] ");
		sb.append(weights);
		sb.append("\nBias: ["+bias.length+"] ");
		sb.append(bias);
		sb.append("\n\n");
		
		return sb.toString();
	
	}
}
