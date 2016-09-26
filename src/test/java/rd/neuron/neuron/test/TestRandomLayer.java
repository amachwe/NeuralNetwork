/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rd.neuron.neuron.test;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.LayerIf;
import rd.neuron.neuron.UnitLayerBuilder;

/**
 *
 * @author azahar
 */
public class TestRandomLayer {

	@Test
	public void doLayer() {

		FloatMatrix input1 = new FloatMatrix(3, 1);
		input1.put(0, 0, 1);
		input1.put(1, 0, 1);
		input1.put(2, 0, 1);
		FloatMatrix input2 = new FloatMatrix(3, 1);
		input2.put(0, 0, 0);
		input2.put(1, 0, 0);
		input2.put(2, 0, 0);
		FloatMatrix input3 = new FloatMatrix(3, 1);
		input3.put(0, 0, 2);
		input3.put(1, 0, 1);
		input3.put(2, 0, 2);
		FloatMatrix input4 = new FloatMatrix(3, 1);
		input4.put(0, 0, 0);
		input4.put(1, 0, 1);
		input4.put(2, 0, 0);
		LayerIf l1 = new UnitLayerBuilder().build(3, 2, Function.ReLU);
		LayerIf l2 = new UnitLayerBuilder().build(2, 1, Function.ReLU);

		/*
		 * ->o 
		 * 			o 
		 * ->o 	
		 * 			o --> o 
		 * ->o
		 */
		System.out.println(input1 + " - " + l2.io(l1.io(input1)));
		System.out.println(input2 + " - " + l2.io(l1.io(input2)));
		System.out.println(input3 + " - " + l2.io(l1.io(input3)));
		System.out.println(input4 + " - " + l2.io(l1.io(input4)));
	}
}
