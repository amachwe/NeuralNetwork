package rd.data;

import java.util.Arrays;
/**
 * Softmax implementation.
 * Converts any vector of real values into probabilistic equivalent so that sum = 1.
 * @author azahar
 *
 */
public final class Softmax {
	
	/**
	 * Test Implementation
	 * @param args
	 */
	public static void main(String...args)
	{
		double[] score = {1,2,4,3,0};
		double[] prob = softmax(score);
		printA(prob);
		validate(prob);
	
		
		
	}
	/**
	 * Validate - sum of probabilities generated must be = 1
	 * @param prob
	 */
	private static void validate(double[] prob)
	{
		double sum = 0;
		for(double v : prob)
		{
			sum+=v;
		}
		
		if(sum!=1.0)
		{
			throw new IllegalStateException("Probabilities are not correct. Total not equal to 1.0, value: "+sum);
		}
	}
	/**
	 * Print the resulting array of probabilities
	 * @param a
	 */
	private static void printA(double[] a)
	{
		System.out.println(Arrays.toString(a));
	}
	
	/**
	 * Softmax function
	 * @param input - to be converted to probabilities
	 * @return
	 */
	public static double[] softmax(double[] input) {
		double prob[] = new double[input.length];

		double sum = 0;
		for (double val : input) {
			sum += Math.exp(val);
		}
		if (sum == 0) {
			throw new IllegalStateException("Sum cannot be zero");
		}
		for (int i = 0; i < input.length; i++) {
			prob[i] = Math.exp(input[i]) / sum;
		}

		return prob;
	}
}


