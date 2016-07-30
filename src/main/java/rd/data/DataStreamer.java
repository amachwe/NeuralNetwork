package rd.data;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.jblas.FloatMatrix;

public class DataStreamer implements Iterable<FloatMatrix> {

	private final Map<FloatMatrix, FloatMatrix> trainData = new HashMap<>();

	private final int inDataWidth, outDataWidth;

	/**
	 * 
	 * @param inDataWidth - input data width
	 * @param outDataWidth - output data width
	 */
	public DataStreamer(int inDataWidth, int outDataWidth) {
		this.inDataWidth = inDataWidth;
		this.outDataWidth = outDataWidth;
	}

	/**
	 * 
	 * @param data - Map of input and output float matrix (rows)
	 */
	public DataStreamer(Map<FloatMatrix, FloatMatrix> data) {
		FloatMatrix key = data.keySet().iterator().next();
		this.inDataWidth = key.getRows();
		this.outDataWidth = data.get(key).getRows();
		trainData.putAll(data);
	}
	

	/**
	 * Add input output pair
	 * @param data
	 * @param output
	 */
	public void add(float[] data, float... output) {
		if (data.length == inDataWidth) {
			FloatMatrix inFm = new FloatMatrix(inDataWidth, 1);
			FloatMatrix outFm = new FloatMatrix(outDataWidth, 1);
			int i = 0;
			for (float item : data) {
				inFm.put(i++, 0, item);
			}
			i = 0;
			for (float item : output) {
				outFm.put(i++, item);
			}
			trainData.put(inFm, outFm);
		} else {
			System.err.println("Error data width does not match");
		}
	}
	
	/**
	 * Add input output pair
	 * @param data
	 * @param output
	 */
	public void add(int[] data, int... output) {
		if (data.length == inDataWidth) {
			FloatMatrix inFm = new FloatMatrix(inDataWidth, 1);
			FloatMatrix outFm = new FloatMatrix(outDataWidth, 1);
			int i = 0;
			for (int item : data) {
				inFm.put(i++, 0, (float)item);
			}
			i = 0;
			for (int item : output) {
				outFm.put(i++, (float)item);
			}
			trainData.put(inFm, outFm);
		} else {
			System.err.println("Error data width does not match");
		}
	}
	
	
	/**
	 * Randomly return an input
	 * @return randomly selected input - use getOutput to obtain corresponding output
	 */
	public FloatMatrix getRandom()
	{
		int index = (int)(Math.random()*trainData.keySet().size());
		FloatMatrix fm=null;
		int i=0;
		for(FloatMatrix key: trainData.keySet())
		{
			if(i>=index)
			{
				return key;
			}
			i++;
			fm = key;
		}
		
		return fm;
	}

	/**
	 * Iterator
	 */
	@Override
	public Iterator<FloatMatrix> iterator() {
		return trainData.keySet().iterator();
	}

	/**
	 * Get output from input
	 * @param input - the input for which we need to get the output
	 * @return get output
	 */
	public FloatMatrix getOutput(FloatMatrix input) {
		return trainData.get(input);
	}
}
