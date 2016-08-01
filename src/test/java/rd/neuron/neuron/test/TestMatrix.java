/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rd.neuron.neuron.test;

import static org.junit.Assert.*;

import org.apache.spark.mllib.linalg.DenseMatrix;
import org.junit.Test;

/**
 * Matrix operations test
 * @author azahar
 */
public class TestMatrix {

	@Test
    public void doTest() {
        DenseMatrix trans = new DenseMatrix(2, 2, new double[]{0.6, 0.15, 0.4, 0.85}, false);
        DenseMatrix init = new DenseMatrix(1, 2, new double[]{0.1, 0.9}, false);
        System.out.println(init+" \n------------- \n"+trans);
        for (int i = 0; i < 1; i++) {
            trans = trans.multiply(trans);
        }

        assertNotNull(init.multiply(trans));

    }
}
