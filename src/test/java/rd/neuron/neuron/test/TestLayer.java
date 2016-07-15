/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rd.neuron.neuron.test;

import org.jblas.FloatMatrix;
import org.junit.Test;
import rd.neuron.neuron.Layer;
import rd.neuron.neuron.Layer.Function;

/**
 *
 * @author azahar
 */
public class TestLayer {

    @Test
    public void doLayer() {
        FloatMatrix fm1 = new FloatMatrix(2, 2);
        fm1.put(0, 0,1f);
        fm1.put(0, 1,0.2f);
        fm1.put(1, 0,0.5f);
        fm1.put(1, 1,4f);
        
        FloatMatrix bias = new FloatMatrix(2,1);
        bias.put(0,0, 1);
        bias.put(1,0, 1);
        
        FloatMatrix input1 = new FloatMatrix(2,1);
        input1.put(0, 0,1);
        input1.put(1,0, 1);
        
        FloatMatrix input2 = new FloatMatrix(2,1);
        input2.put(0,0, 0);
        input2.put(1,0, 0);
        
        FloatMatrix input3 = new FloatMatrix(2,1);
        input3.put(0,0, 1);
        input3.put(1,0, 0);
        
        FloatMatrix input4 = new FloatMatrix(2,1);
        input4.put(0,0, 0);
        input4.put(1,0, 1);
        Layer l1 = new Layer(fm1, bias, Function.ReLU);

        System.out.println(input1 +" - "+l1.io(input1));
        System.out.println(input2+" - "+l1.io(input2));
        System.out.println(input3+ " - "+l1.io(input3));
        System.out.println(input4+" - "+l1.io(input4));
        
    }
}
